"""
ukf.py — Dynamics-Centric UKF (12-state, 9-observation)
=========================================================
순수 추정 모듈: UKF 상태 추정 + 잔차(Residual) + NIS 계산만 수행.

State  x = [pos_ned(3), euler(3), vel_ned(3), gyro(3)]  → 12차원
Obs    z = [gps_pos_ned(3), gps_vel_ned(3), gyro(3)]    → 9차원
"""

import numpy as np
import os
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


def compute_nis_scaled(r_sub, Pzz_sub, nz, offset=1.0):
    """NIS 압축. offset=1.0 → log1p(기존, gyro 유지). offset<1.0 → 저압축(vel).
    채널별 offset이 다르므로 호출부에서 vel=0.5 / gyro=1.0 로 분리 적용한다."""
    try:
        nis_raw = r_sub @ np.linalg.solve(Pzz_sub, r_sub) / nz
    except np.linalg.LinAlgError:
        nis_raw = 0.0
    log_nis = np.log(nis_raw + offset)   # offset=1.0: log1p / offset=0.5: log(x+0.5) 저압축
    nis_scaled = log_nis / (log_nis + 1.0)
    return nis_raw, nis_scaled


class DynamicsUKF:
    def __init__(self, dt=0.02, calib=None, ff=1.00, q_gate=0.0):
        # ff=1.0: fading-memory 끔(P 재팽창=이득↑+NIS분모↑ 둘 다 손해). 고집은 low Q로.
        # q_gate>0: 명령 토크에 비례해 gyro Q 인플레(정상 기동 FP 억제). 0=off.
        self.nx = 12
        self.nz = 9
        self.dt = dt
        self.ff = ff
        self.q_gate = float(q_gate)
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

        self.Q = np.diag([
        1e-3, 1e-3, 1e-3,    # pos    (무효, 유지)
        5e-4, 5e-4, 5e-4,    # euler  1e-3 → 5e-4 (d'v↑)
        5e-3, 5e-3, 5e-3,    # vel    (유지; velQ는 R로 대체)
        5e-4, 5e-4, 5e-4,    # gyro   1e-3 → 5e-4 (3박자 개선)
        ])
        
        self.R = np.diag([
        0.5, 0.5, 0.5,       # pos    0.1→0.5 (2026-07-22: pos NIS 신규 로깅용 시작값, 사후 검증)
        0.3, 0.3, 0.3,       # vel    (현행 튜닝값 유지 — 변경 금지)
        0.5, 0.5, 0.5,       # gyro   (현행 튜닝값 유지 — 변경 금지)
        ])

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

    def step(self, z, u):
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
        Pzz = self.R + sum(self.Wc[i]*np.outer(z_pts[i]-z_bar, z_pts[i]-z_bar) for i in range(2*n+1))
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
