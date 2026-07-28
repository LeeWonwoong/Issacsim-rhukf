#!/usr/bin/env python3
"""fit_static_from_rotor.py — 로터 각속도 기반 C_thrust / C_torque 재캘리브레이션.

왜 이 경로인가 (2026-07-28 진단):
  자세 rate loop 는 폐루프라서 "IMU 자이로 미분 vs 토크명령" 회귀가 편향된다.
  실제로 그 회귀는 R²=0.03~0.15, 계수가 필터 차단주파수에 따라 2배씩 흔들렸다.
  여기서는 run_sim 이 기록한 **실제 적용 로터 각속도 ω_i** 와 **시뮬레이터 실제 각속도**를 써서
  PX4 명령 → 실제 응답 게인을 직접 잰다. 관측 노이즈·UKF 무관.

  Pegasus 플랜트 모델(quadratic_thrust_curve.py):
    T_i = k·ω_i²            (k=8.54858e-6)      → 로터 위치에 +z(FLU) 로 인가
    τ_z(항력모멘트) = Σ c_m·ω_i²·dir_i           (c_m=1e-6, dir=[-1,-1,1,1])
  로터 위치가 x/y 로 비대칭(Σx=+0.0267, Σy=-0.0077)이라 총추력·요 명령이 롤/피치로
  새어든다 → 회귀에 교차항(3축 명령 + 총추력)을 반드시 넣어야 한다(안 넣으면 롤 R²=0.4).

산출 C_torque 는 **UKF 가 쓰는 I 와 짝**이다: ω̇_pred = C·τ_cmd/I 가 실제와 맞도록
I·ω̇ 를 τ_cmd 에 회귀한다. 따라서 I 를 바꾸면 C 도 함께 재적합해야 한다.

사용:
  ~/isaacsim/python.sh calibration/fit_static_from_rotor.py results_calib_mass/rotor_log.npz [--write]
"""
import argparse
import json
import sys

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

K_ROTOR = 8.54858e-6
C_ROLL_MOMENT = 1e-6
ROT_DIR = np.array([-1, -1, 1, 1])
ROTOR_POS = np.array([                     # 바디(FLU) 기준 로터 위치 [m] (iris.usd)
    [0.1379854530096054, -0.206716388463974, 0.023],
    [-0.1251116842031479, 0.2187541425228119, 0.023],
    [0.1379999965429306, 0.20257577300071716, 0.023],
    [-0.12415359914302826, -0.22234579920768738, 0.023],
])
AXES = ('roll(x)', 'pitch(y)', 'yaw(z)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--calib', default='calibration/calibration.json')
    ap.add_argument('--fc', type=float, default=5.0,
                    help='영위상 저역통과 [Hz]. 명령/응답 양쪽에 동일 적용')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    A = d['data']
    dt = float(d['dt'])
    fs = 1.0 / dt
    calib = json.load(open(args.calib))
    drone = calib['drone']
    I = np.array([float(drone['Ixx']), float(drone['Iyy']), float(drone['Izz'])])
    m, g = float(drone['mass']), float(drone['g'])

    def zp(x, fc=args.fc):
        b, a = butter(4, fc / (0.5 * fs), btype='low')
        return filtfilt(b, a, x, axis=0)

    # 시간 갭으로 연속구간 분할 (리셋/재이륙)
    t = A[:, 0]
    bnd = np.r_[0, np.flatnonzero(np.diff(t) > 2 * dt) + 1, len(t)]
    segs = [A[bnd[i]:bnd[i + 1]] for i in range(len(bnd) - 1) if bnd[i + 1] - bnd[i] > 3000]
    print(f"[*] {args.npz}: {len(A)} rows @ {1/dt:.0f}Hz → 연속구간 {len(segs)}개 "
          f"({sum(len(s) for s in segs)} 샘플), fc={args.fc}Hz")
    print(f"[*] 모델 m={m:.6f} I={I}")
    if not segs:
        print("[!] 사용할 구간이 없습니다.")
        return 1

    thr_a, thr_c = [], []                      # 적용 추력 / 명령
    hov_u = []
    Yt = [[], [], []]                          # I·ω̇ - 자이로항
    Xt = [[], [], []]                          # [cmd(3, FLU), thrust, 1]
    Yi = [[], [], []]                          # τ_applied - 자이로항 (유효관성 검증용)
    Xi = [[], [], []]
    for S in segs:
        w = S[:, 7:11]
        ok = (w.min(axis=1) > 101) & (w.max(axis=1) < 1099)     # 포화 제외
        if ok.sum() < 3000:
            continue
        S, w = S[ok], w[ok]
        T = K_ROTOR * w ** 2
        thrust = T.sum(axis=1)
        tau_app = np.column_stack([(T * ROTOR_POS[:, 1]).sum(axis=1),
                                   -(T * ROTOR_POS[:, 0]).sum(axis=1),
                                   (C_ROLL_MOMENT * ROT_DIR * w ** 2).sum(axis=1)])   # FLU
        u_thr = np.abs(S[:, 3])
        cmd_flu = np.column_stack([S[:, 4], -S[:, 5], -S[:, 6]])   # PX4 FRD → FLU
        om = zp(S[:, 14:17])                                        # GT 각속도(FLU)
        wd = savgol_filter(om, 11, 3, deriv=1, delta=dt, axis=0)
        cf, tf = zp(cmd_flu), zp(u_thr)
        taf = zp(tau_app)

        thr_a.append(thrust); thr_c.append(u_thr)
        lvl = (np.abs(om).max(axis=1) < 0.05)
        if lvl.any():
            hov_u.append(u_thr[lvl])
        for k in range(3):
            a, b = (k + 1) % 3, (k + 2) % 3
            gyro_term = (I[a] - I[b]) * om[:, a] * om[:, b]
            reg = np.column_stack([cf, tf, np.ones(len(cf))])
            Yt[k].append(I[k] * wd[:, k] - gyro_term); Xt[k].append(reg)
            Yi[k].append(taf[:, k] - gyro_term)
            Xi[k].append(np.column_stack([wd[:, k], tf, np.ones(len(cf))]))

    def fit(Y, X, col=0, full=False):
        y, Xm = np.concatenate(Y), np.vstack(X)
        th, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        r2 = 1 - ((y - Xm @ th).var() / y.var())
        return (th, r2) if full else (float(th[col]), r2)

    # ── 추력 ──
    ta, tc = np.concatenate(thr_a), np.concatenate(thr_c)
    C_thrust = float(np.sum(tc * ta) / np.sum(tc * tc))
    r2t = 1 - ((ta - C_thrust * tc).var() / ta.var())
    print("\n── 추력 ──")
    print(f"  C_thrust(정적, 원점통과) = {C_thrust:8.4f}   R²={r2t:.4f}")
    if hov_u:
        hu = float(np.median(np.concatenate(hov_u)))
        print(f"  C_thrust(호버평형 m·g/u) = {m*g/hu:8.4f}   (u_hover={hu:.5f})")

    # ── 유효 관성 검증 (적용토크 → 실제 각가속도; PX4 명령 무관) ──
    print("\n── 유효 관성 검증 (적용토크 = I·ω̇) ──")
    for k in range(3):
        Ik, r2 = fit(Yi[k], Xi[k])
        print(f"  {AXES[k]:9s} I_eff={Ik:.6f}  설정 {I[k]:.6f}  비={Ik/I[k]:.3f}  R²={r2:+.3f}")
    print("  ※ yaw 는 로터 스핀업 반작용(모델 외 토크)이 커서 비가 1에서 크게 벗어남 — 정상.")

    # ── C_torque (최종): I·ω̇ 를 명령에 회귀 = UKF 예측식과 동일한 형태 ──
    print("\n── C_torque (UKF 예측식 기준, 교차항 포함) ──")
    C_tq, K_tq = [], []
    for k in range(3):
        th, r2 = fit(Yt[k], Xt[k], full=True)
        C_tq.append(float(th[k]))
        K_tq.append(float(th[3]) / max(C_thrust, 1e-9))   # 총추력[N] 당 토크 결합
        print(f"  C_torque_{AXES[k]:9s} = {th[k]:8.4f}   R²={r2:+.3f}   "
              f"추력결합 {th[3]/C_thrust:+.6f} N·m/N")
    print("  ※ 추력결합 = 로터 중심이 바디 COM 에서 (x+6.7mm, y−1.9mm) 어긋나 총추력이 만드는")
    print("     상시 토크. PX4 는 트림으로 상쇄하지만 UKF 가 모르면 그 트림명령을 실제 토크로 오해해")
    print("     ω̇ 예측에 상시 편향이 생긴다(실측 ω̇y −3.2 rad/s²).")

    print("\n── 현행 calibration.json 대비 ──")
    print(f"  C_thrust    {calib['C_thrust']:9.4f} → {C_thrust:9.4f}")
    print(f"  C_torque_xy {calib['C_torque_xy']:9.4f} → x={C_tq[0]:.4f} / y={C_tq[1]:.4f}")
    print(f"  C_torque_z  {calib['C_torque_z']:9.4f} → {C_tq[2]:9.4f}")

    if args.write:
        import shutil
        shutil.copy2(args.calib, args.calib + '.bak_prestatic')
        calib['C_thrust'] = C_thrust
        calib['C_torque_x'] = C_tq[0]
        calib['C_torque_y'] = C_tq[1]
        calib['C_torque_xy'] = 0.5 * (C_tq[0] + C_tq[1])   # 구 스키마 호환
        calib['C_torque_z'] = C_tq[2]
        # ⚠ 추력결합은 여기서 쓰지 않는다: 이 회귀는 FLU 프레임 + 총추력이 기동과 상관돼 있어
        #   계수가 오염된다(요축 −0.048 은 로터 스핀업 반작용을 흡수한 값). UKF 프레임(FRD)에서
        #   예측 편향으로부터 직접 추정하는 `validate_by_regime.py --fit-coupling` 을 쓸 것.
        calib.pop('torque_thrust_coupling', None)
        calib['note'] = ("2026-07-28 총질량 1.372kg 정합 + 로터각속도 기반 재캘리브레이션 "
                         f"(fit_static_from_rotor.py, fc={args.fc}Hz). "
                         "C_torque 는 drone.I 와 짝 — I 변경 시 재적합 필수.")
        with open(args.calib, 'w') as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)
        print(f"\n[*] {args.calib} 기록 (백업 .bak_prestatic)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
