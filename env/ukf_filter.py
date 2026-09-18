"""
ukf.py — Dynamics-Centric UKF (12-state, 9-observation)
=========================================================
순수 추정 모듈: UKF 상태 추정 + 잔차(Residual) + NIS 계산만 수행.

State  x = [pos_ned(3), euler(3), vel_ned(3), gyro(3)]  → 12차원
Obs    z = [gps_pos_ned(3), gps_vel_ned(3), gyro(3)]    → 9차원
"""

import numpy as np
import os
from env.knobs import knob   # ★09-18 env 변수 → YAML 노브
import json


def load_calibration(path='calibration.json'):
    """캘리브레이션 파일 로드. 주어진 경로가 없으면 흔한 위치들을 자동 탐색.
    탐색 순서: (1) 준 경로 그대로 (2) ./calibration/<파일명>
    (3) 이 모듈 기준 ../calibration/<파일명> (4) cwd/<파일명)."""
    import glob
    base = os.path.basename(path)
    here = os.path.dirname(os.path.abspath(__file__))   # .../env
    repo = os.path.dirname(here)                          # repo root
    candidates = [
        path,
        os.path.join('calibration', base),
        os.path.join(repo, 'calibration', base),
        os.path.join(repo, base),
        os.path.join(os.getcwd(), base),
    ]
    # 마지막 폴백: 레포 어디든 같은 이름 파일
    candidates += sorted(glob.glob(os.path.join(repo, '**', base), recursive=True))

    for c in candidates:
        if c and os.path.exists(c):
            with open(c) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"[!] '{base}' 를 찾지 못했습니다. 다음 위치를 확인했습니다:\n  - "
        + "\n  - ".join(dict.fromkeys(candidates))
        + "\n  calibrate_sysld.py로 생성하거나 calibration/ 폴더에 두세요.")


def to_physical_u(thrust, torque, calib):
    N = len(thrust)
    u = np.zeros((N, 4))
    # C_torque_x/y 가 있으면 축별로 쓴다(2026-07-28: 실측 롤/피치 게인이 ~15% 다름).
    # 없으면 기존 스키마(C_torque_xy 공통)로 폴백.
    cx = calib.get('C_torque_x', calib['C_torque_xy'])
    cy = calib.get('C_torque_y', calib['C_torque_xy'])
    u[:, 0] = np.abs(thrust[:, 2]) * calib['C_thrust']
    u[:, 1] = torque[:, 0] * cx
    u[:, 2] = torque[:, 1] * cy
    u[:, 3] = torque[:, 2] * calib['C_torque_z']
    # 총추력에 비례하는 기하 토크 오프셋 (2026-07-28).
    #  Iris 는 로터 중심이 바디 COM 에서 (x+6.7mm, y−1.9mm) 어긋나 있어 총추력이 상시 토크를 만든다.
    #  PX4 는 이를 트림 명령으로 상쇄하지만, 모델이 모르면 그 트림을 실제 토크로 오해해
    #  ω̇ 예측에 상시 편향이 생긴다(수정 전 실측: ω̇y −3.2 rad/s², 호버·순항·급기동 공통).
    K = calib.get('torque_thrust_coupling')
    if K:
        u[:, 1] += K[0] * u[:, 0]
        u[:, 2] += K[1] * u[:, 0]
        u[:, 3] += K[2] * u[:, 0]
    return u


_NIS_CLIP_DEFAULT = float(knob('NIS_CLIP', '3.0'))   # ★2026-09-12 클립 env (확정 4.0: 밴드 최대 3.75 무손실)
def compute_nis_scaled(r_sub, Pzz_sub, nz, offset=1.0, clip=None):
    clip = _NIS_CLIP_DEFAULT if clip is None else clip
    """관측 압축 ε̃ = min( log(1 + √NIS) , clip ),  NIS = rᵀS⁻¹r / nz.
    ★2026-08-20 v5: 마할라노비스 '거리'(√NIS = 몇 σ) → log1p → clip3. 범위 [0,3].
      · √: NIS(마할라 제곱)은 스케일 폭발(raw124) → 거리는 선형(√124≈11σ). OOD/AD 표준.
      · log1p: 표준 압축(스케일 파라미터 없음). clip3: RL 관측클리핑 관행(온셋 2.6이라 거의 안 걸림).
      실측 온셋 d′(공격 vs 바람/기동) 2.48 ≈ tanh√/7(2.56) ≫ 기존 NIS²+log/(1+log)(1.73).
      z-score 배제: 아핀이라 d′ 못 올리고, 공격이 러닝통계 오염+비정상성으로 RHUKF 해침.
    offset 인자는 하위호환용(미사용). 채널별 동일 적용(gyro 주 + vel 보조)."""
    try:
        nis_raw = r_sub @ np.linalg.solve(Pzz_sub, r_sub) / nz
    except np.linalg.LinAlgError:
        nis_raw = 0.0
    nis_scaled = min(float(np.log1p(np.sqrt(max(nis_raw, 0.0)))), clip)
    return nis_raw, nis_scaled


class DynamicsUKF:
    def __init__(self, dt=0.02, calib=None, ff=1.00, q_gate=0.0, stale_gps_inflate=1e6):
        # ff=1.0: fading-memory 끔(P 재팽창=이득↑+NIS분모↑ 둘 다 손해). 고집은 low Q로.
        # q_gate>0: 명령 토크에 비례해 gyro Q 인플레(정상 기동 FP 억제). 0=off.
        #
        # stale_gps_inflate (2026-08-05): 멀티레이트 게이트.
        #   predict + gyro update = 50Hz(루프),  GPS update = 10Hz(센서 실제 레이트).
        #   GPS 가 낡은 스텝에서 R 의 GPS 블록(0:6)에 이 배수를 곱해 보정을 사실상 끈다
        #   (극한에서 'GPS update 스킵'과 수학적으로 동일). 코드 경로를 하나로 유지하려는 선택.
        #   ⚠ 이전 구현은 10Hz GPS 를 50Hz 로 5회 재사용(ZOH)해 GPS 채널을 과대가중했다
        #     — 실효 R 이 약 1/5 로 작아져 "고집(R↑·Q↓)" 설계 의도와 반대 방향이었다.
        self.nx = 12
        self.nz = 9
        self.dt = dt
        self.ff = ff
        self.q_gate = float(q_gate)
        self.stale_gps_inflate = float(stale_gps_inflate)
        d = calib['drone']
        self.m = d['mass']
        self.g = d['g']
        self.I = [d['Ixx'], d['Iyy'], d['Izz']]
        self.drag = np.array(calib['drag'])

        n = self.nx
        lam = 0.5**2 * n - n
        self.lam = lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - 0.5**2 + 2.0)

        # ★★ 2026-08-27 config E 채택 (RESULTS.md Ⅸ · 아티팩트 09장).
        #   목표 4개 동시 달성: ①고집 유지(공격이 u 로 들어와 R↓에도 잔차 유지, sustain 0.99)
        #   ②온셋 τon 3 유지 ③오프셋 τoff gyro 8→7 / vel 44→18 ④비자명(SNR 7.74→3.86,
        #   δ0.1-0.25 공격의 49% 가 평시에 묻힘 = 약공격 POMDP 화, δ≥0.55 는 여전히 분리).
        #   ── 롤백: env (UKF_Q_GYRO=5e-3 UKF_Q_EULER=5e-4 UKF_R_GYRO=0.2 UKF_R_VEL=0.1)
        #      또는 파일 복원: env/ukf_filter.py.bak_configA_20260827
        #   구 config A: Q_eul 5e-4 / Q_gyr 5e-3 / R_vel 0.1 / R_gyr 0.2
        self.Q = np.diag([
        1e-3, 1e-3, 1e-3,    # pos    (무효 실증 — OFAT 1e-4~5e-2 전 지표 불변, 2026-08-26)
        2e-3, 2e-3, 2e-3,    # euler  ★E: 5e-4→2e-3 (vel 채널 애매화 보조. vel d′ 1.21→0.93)
        5e-3, 5e-3, 5e-3,    # vel    (유지)
        2e-2, 2e-2, 2e-2,    # gyro   ★E: 5e-3→2e-2 (바닥 유지 + 고원 2.35→1.84 = 계조 확보, δ0.4/0.7 이 1.99/2.49 로 구분)
        ])
        # ── env 오버라이드: Q/R 전 블록 (2026-08-26 전수 노출) ─────────────────
        #   상태 12D: pos(0:3) euler(3:6) vel(6:9) gyro(9:12)
        #   측정  9D: pos(0:3) vel(3:6)  gyro(6:9)
        #   ⚠ 2026-08-26 버그수정: 구 UKF_R_VEL 은 R[6:9](=gyro 측정블록)에 썼다 —
        #     이름과 실제가 어긋났다. 이제 UKF_R_VEL→R[3:6], UKF_R_GYRO→R[6:9] 로 분리.
        #     (구 UKF_R_VEL 을 쓴 스크립트는 없어 과거 실험은 오염되지 않았다.)
        def _envset(mat, lo, hi, *names):
            for nm in names:
                v = knob(nm, '')
                if v:
                    for _i in range(lo, hi):
                        mat[_i, _i] = float(v)
                    return
        _envset(self.Q, 0, 3,  'UKF_Q_POS')
        _envset(self.Q, 3, 6,  'UKF_Q_EULER')
        _envset(self.Q, 6, 9,  'UKF_Q_VEL')
        _envset(self.Q, 9, 12, 'UKF_Q_GYRO')

        self.R = np.diag([
        0.5, 0.5, 0.5,       # pos    (무효 실증 — OFAT 1e-3~5 전 지표 불변)
        0.01, 0.01, 0.01,    # vel    ★E: 0.1→0.01 (≈GPS vel 수평 σ² 0.006. vel 잔향 τoff 44→18스텝 = 복귀판정 가능)
        0.02, 0.02, 0.02,    # gyro   ★E: 0.2→0.02 (σ² 0.002~0.005 의 4~10배 — 구 40~100배 인플레보다 정석에 가깝다.
                             #   ★공격이 z 아닌 u 로 들어와 R↓에도 고집 불변(실증: 40배↓ 에도 sustain 0.98) = R↓는 공짜 방향)
        ])
        _envset(self.R, 0, 3, 'UKF_R_POS')
        _envset(self.R, 3, 6, 'UKF_R_VEL')
        _envset(self.R, 6, 9, 'UKF_R_GYRO')

        self.x = np.zeros(12)
        self.P = np.eye(12) * 0.1

    def _f(self, x, u):
        s = x.copy()
        sdt = 0.005
        n_sub = int(self.dt / sdt)
        for _ in range(n_sub):
            phi, th, psi = s[3:6]
            vx, vy, vz = s[6:9]
            p, q, r = s[9:12]
            # 자세 클립 0.8 → 1.05 (2026-07-29). 1.05 rad = 60.2° = crash_flip 판정과 동일 →
            # **비행 가능 영역 전체에서 모델이 유효**해진다.
            #  구 값 0.8(45.8°)은 45.8~60.2° 구간에서 상태가 상한에 붙어 실제 자세를 못 따라갔고,
            #  그만큼 모델오차가 인위적으로 커졌다(수평추력 오차 50°:0.66N / 60°:2.00N=1.46 m/s²).
            #  공격이 성공해 기우는 구간이 정확히 여기라, 탐지 신호에 비물리적 성분이 섞였다.
            #  ※ 클립의 목적은 특이점 회피인데, 특이점 항은 tan(th)/1/cos(th) 로 **th 만** 들어간다.
            #    phi 는 cp/sp 로만 쓰여 특이점이 없다 = 롤 클립은 원래 잉여. 안전 상한으로만 유지.
            #    1.05 에서 cos=0.498, tan=1.74 — 수치 안전.
            #  ⚠ 모델 변경 = NIS 기준선/d′ 변경. 기준선·밴드 측정 전에 넣을 것(2026-07-29 반영).
            limit = 1.05
            phi = np.clip(phi, -limit, limit)
            th = np.clip(th, -limit, limit)
            s[3:6] = [phi, th, psi]
            cp = np.cos(phi); sp = np.sin(phi)
            ct = np.cos(th);  st = np.sin(th); tt = np.tan(th)
            cps = np.cos(psi); sps = np.sin(psi)
            vbx = cps * ct * vx + sps * ct * vy - st * vz
            vby = (cps * st * sp - sps * cp) * vx + \
                  (sps * st * sp + cps * cp) * vy + ct * sp * vz
            vbz = (cps * st * cp + sps * sp) * vx + \
                  (sps * st * cp - cps * sp) * vy + ct * cp * vz
            fd = np.array([-self.drag[0]*vbx, -self.drag[1]*vby, -self.drag[2]*vbz])
            f_thrust_body = np.array([0, 0, -u[0]])
            f_total_body = f_thrust_body + fd
            R_mat = np.array([
                [ct * cps, sp * st * cps - cp * sps, cp * st * cps + sp * sps],
                [ct * sps, sp * st * sps + cp * cps, cp * st * sps - sp * cps],
                [-st,      sp * ct,                   cp * ct]
            ])
            f_total_ned = R_mat @ f_total_body
            accel_ned = (f_total_ned / self.m) + np.array([0, 0, self.g])
            s[0:3] += np.array([vx, vy, vz]) * sdt
            s[3:6] += np.array([
                p + sp * tt * q + cp * tt * r,
                cp * q - sp * r,
                sp / (ct + 1e-10) * q + cp / (ct + 1e-10) * r
            ]) * sdt
            s[6:9] += accel_ned * sdt
            s[9]  += ((self.I[1]-self.I[2])/self.I[0]*q*r + u[1]/self.I[0]) * sdt
            s[10] += ((self.I[2]-self.I[0])/self.I[1]*p*r + u[2]/self.I[1]) * sdt
            s[11] += ((self.I[0]-self.I[1])/self.I[2]*p*q + u[3]/self.I[2]) * sdt
        return s

    def _h(self, x):
        return np.concatenate([x[0:3], x[6:9], x[9:12]])

    def step(self, z, u, gps_fresh=True):
        """gps_fresh=False 면 z[0:6](GPS pos/vel)을 무시한다 — R 블록 인플레로 구현.
        gyro(z[6:9])와 predict 는 매 호출 정상 수행된다."""
        n = self.nx
        self.P = 0.5 * (self.P + self.P.T)
        try:
            S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-7 * np.eye(n))
        except np.linalg.LinAlgError:
            self.P += np.eye(n) * 1e-6
            S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-7 * np.eye(n))
        pts = np.vstack([self.x, self.x + S_root.T, self.x - S_root.T])
        pts_f = np.array([self._f(p, u) for p in pts])
        x_bar = self.Wm @ pts_f
        cov_spread = sum(self.Wc[i] * np.outer(pts_f[i]-x_bar, pts_f[i]-x_bar) for i in range(2*n+1))
        # maneuver-gated Q: 명령 토크(|u[1:4]|)만큼 gyro 프로세스노이즈 인플레(off면 q_gate=0).
        #   명령된 각가속은 '설명'하고(FP↓), 명령 안 한 이탈(공격)만 NIS로 남김.
        Q_eff = self.Q
        if self.q_gate > 0.0:
            Q_eff = self.Q.copy()
            tau_cmd = np.abs(u[1:4])
            Q_eff[9, 9]   += self.q_gate * tau_cmd[0]
            Q_eff[10, 10] += self.q_gate * tau_cmd[1]
            Q_eff[11, 11] += self.q_gate * tau_cmd[2]
        P_bar = Q_eff + (self.ff ** 2) * cov_spread
        z_pts = np.array([self._h(p) for p in pts_f])
        z_bar = self.Wm @ z_pts
        # 멀티레이트: GPS 가 낡은 스텝은 R[0:6] 인플레 → 그 채널 이득 ≈ 0 (update 스킵과 동치)
        R_eff = self.R
        if not gps_fresh:
            R_eff = self.R.copy()
            R_eff[0:6, 0:6] *= self.stale_gps_inflate
        Pzz = R_eff + sum(self.Wc[i]*np.outer(z_pts[i]-z_bar, z_pts[i]-z_bar) for i in range(2*n+1))
        Pxz = sum(self.Wc[i]*np.outer(pts_f[i]-x_bar, z_pts[i]-z_bar) for i in range(2*n+1))
        K = Pxz @ np.linalg.inv(Pzz)
        res = z - z_bar
        self.x = x_bar + K @ res
        self.P = P_bar - K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        if np.isnan(self.P).any() or np.isinf(self.P).any():
            self.P = np.eye(n) * 0.1
        matrix_to_decompose = (n + self.lam) * self.P + 1e-6 * np.eye(n)
        matrix_to_decompose = 0.5 * (matrix_to_decompose + matrix_to_decompose.T)
        try:
            np.linalg.cholesky(matrix_to_decompose)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix_to_decompose)
            eigvals = np.maximum(eigvals, 1e-8)
            matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            matrix_pd = 0.5 * (matrix_pd + matrix_pd.T) + 1e-6 * np.eye(n)
            try:
                np.linalg.cholesky(matrix_pd)
                self.P = (matrix_pd - 1e-6 * np.eye(n)) / (n + self.lam)
            except np.linalg.LinAlgError:
                self.P = np.eye(n) * 0.1
        return res, Pzz
