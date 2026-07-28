#!/usr/bin/env python3
"""fit_gains.py — 비행 로그로부터 UKF 게인(C_thrust/C_torque/drag) 직접 재적합.

배경(2026-07-28):
  Iris USD 는 body 1.5kg + rotor 4개 0.1186kg = **총 1.6186kg** 인데 UKF 모델은 1.372kg 를
  쓰고 있었고, calibrate_sysld.py 는 가정질량 1.5 로 적합돼 있었다(3중 불일치).
  총 비행질량을 1.372kg 로 정합한 뒤에는 호버 동작점이 이동하므로 계수 재적합이 필수다.

이 스크립트는 **UKF 가 실제로 쓰는 예측식 그 자체**에 OLS 를 건다 → 가정질량 상쇄 문제가
원천적으로 없다. calibrate_sysld.py(가속도계 기반, 가정질량 스케일) 와 독립적인 경로.

  병진: m*(a_ned - [0,0,g]) 를 바디로 회전 → w
        w_x = -drag_x*vbx,  w_y = -drag_y*vby,  w_z = -C_thrust*u_norm - drag_z*vbz
  회전: I_a*ω̇_a - (I_b-I_c)*ω_b*ω_c = C_torque_a * τ_norm_a       (오일러 방정식, UKF _f 와 동일)

사용:
  ~/isaacsim/python.sh calibration/fit_gains.py results_calib_mass/sysid_log.npz [--write]
  (시스템 python3 는 numpy2/scipy 불일치 → isaacsim python 사용)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

COLS = ('episode reset attack action z0_gpsN z1_gpsE z2_gpsD z3_velN z4_velE z5_velD '
        'z6_gyrx z7_gyry z8_gyrz u0_thrust u1_tx u2_ty u3_tz euler_phi euler_th euler_psi '
        'atk_scale atk_delay').split()
IDX = {n: i for i, n in enumerate(COLS)}


def zero_phase(x, fs, fc=8.0):
    nyq = 0.5 * fs
    b, a = butter(4, min(fc, 0.8 * nyq) / nyq, btype='low')
    return filtfilt(b, a, x, axis=0)


def rot_ned_from_body(phi, th, psi):
    """ZYX 오일러 → 바디(FRD)에서 NED 로의 회전행렬. UKF _f 와 동일한 식."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(th), np.sin(th)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.empty((len(phi), 3, 3))
    R[:, 0, 0] = ct * cps
    R[:, 0, 1] = sp * st * cps - cp * sps
    R[:, 0, 2] = cp * st * cps + sp * sps
    R[:, 1, 0] = ct * sps
    R[:, 1, 1] = sp * st * sps + cp * cps
    R[:, 1, 2] = cp * st * sps - cps * sp
    R[:, 2, 0] = -st
    R[:, 2, 1] = sp * ct
    R[:, 2, 2] = cp * ct
    return R


def ols(Phi, y):
    """열별 계수 + 표준오차 + R^2."""
    Phi = np.atleast_2d(Phi.T).T if Phi.ndim == 1 else Phi
    theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    resid = y - Phi @ theta
    dof = max(1, len(y) - Phi.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(Phi.T @ Phi)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float('nan')
    return theta, se, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--calib', default='calibration/calibration.json',
                    help='캡처 당시 사용된 calibration.json (u 를 정규화 명령으로 되돌리는 데 필요)')
    ap.add_argument('--min-alt', type=float, default=1.0, help='이 고도(m) 미만 샘플 제외')
    ap.add_argument('--trim', type=int, default=100, help='에피소드 앞뒤로 버릴 샘플 수(이륙/착지 과도)')
    ap.add_argument('--fc', type=float, default=8.0,
                    help='영위상 저역통과 차단주파수(Hz). 명령/응답 양쪽에 동일 적용')
    ap.add_argument('--write', action='store_true', help='결과를 calibration.json 에 기록')
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    dt = float(d['dt'])
    fs = 1.0 / dt
    calib = json.load(open(args.calib))
    drone = calib['drone']

    # ── 두 가지 소스 지원 ──
    #   zu_log.npz   : UKF 관측(z) 기반 — GPS 노이즈 포함(미분 SNR 낮음, 참고용)
    #   sysid_log.npz: GT 속도 + IMU + PX4 명령 setpoint — 재캘리브레이션 기본 소스
    T = d['t'] if 't' in d else None
    if 'data' in d:                                # zu_log
        A = d['data']
        C_thr_cap = float(calib['C_thrust'])       # u 를 정규화 명령으로 역산
        C_txy_cap = float(calib['C_torque_xy'])
        C_tz_cap = float(calib['C_torque_z'])
        src = 'zu(관측)'
    else:                                          # sysid_log (GT)
        n = len(d['velocity'])
        A = np.zeros((n, len(COLS) + 3))           # +3 = IMU 비력(바디) 열
        A[:, IDX['episode']] = d['episode']
        A[:, IDX['attack']] = d['attack']
        A[:, [IDX['z3_velN'], IDX['z4_velE'], IDX['z5_velD']]] = d['velocity']
        A[:, [IDX['z6_gyrx'], IDX['z7_gyry'], IDX['z8_gyrz']]] = d['gyro']
        A[:, [IDX['euler_phi'], IDX['euler_th'], IDX['euler_psi']]] = d['euler']
        A[:, IDX['u0_thrust']] = np.abs(d['thrust'][:, 2])     # 이미 정규화 명령
        A[:, [IDX['u1_tx'], IDX['u2_ty'], IDX['u3_tz']]] = d['torque']
        A[:, IDX['z2_gpsD']] = -50.0               # 고도 필터 무력화(로깅 시 이미 alt>1 적용)
        A[:, len(COLS):len(COLS)+3] = d['accelerometer']
        C_thr_cap = C_txy_cap = C_tz_cap = 1.0
        src = 'sysid(GT)'
    m = float(drone['mass'])
    g = float(drone['g'])
    I = np.array([float(drone['Ixx']), float(drone['Iyy']), float(drone['Izz'])])

    print(f"[*] {args.npz} [{src}]: {A.shape[0]} rows @ {fs:.0f}Hz, "
          f"{len(np.unique(A[:, 0]))} episodes")
    print(f"[*] 모델 질량/관성: m={m:.6f}  I={I}")
    print(f"[*] 캡처 당시 계수(역산용): C_thrust={C_thr_cap:.4f} C_tq_xy={C_txy_cap:.4f} "
          f"C_tq_z={C_tz_cap:.4f}")

    # ── 에피소드별로 필터·미분 후 유효구간만 모음 ──
    Wz, Uz, Vbz = [], [], []          # 수직: w_z = -C*u - drag_z*vbz
    Wx, Vbx, Wy, Vby = [], [], [], []
    Tq = [[], [], []]                 # (LHS, τ_norm) per axis
    Rhs = [[], [], []]
    hover_u, hover_ct = [], []

    # ── 구간 분할: 시간 갭 + 에피소드 번호 + 속도 불연속(리셋 텔레포트/이착륙 갭) ──
    #   sweep 모드에선 episode 열이 증가하지 않을 수 있어 번호만으로는 갈라지지 않는다.
    v_all = A[:, [IDX['z3_velN'], IDX['z4_velE'], IDX['z5_velD']]]
    brk = (np.abs(np.diff(v_all, axis=0)).max(axis=1) > 2.0) | \
          (np.diff(A[:, IDX['episode']]) != 0)
    if T is not None:
        brk = brk | (np.diff(T) > 5 * dt)          # 로깅 중단(착지/리셋) 구간
    jump = np.r_[True, brk]
    bounds = np.flatnonzero(jump)
    segments = [A[s:e] for s, e in zip(bounds, np.r_[bounds[1:], len(A)])]
    print(f"[*] 연속구간 {len(segments)}개 (속도 불연속/에피소드 경계 기준)")

    for seg in segments:
        if len(seg) < 4 * args.trim:
            continue
        seg = seg[args.trim:-args.trim]
        alt = -seg[:, IDX['z2_gpsD']]
        seg = seg[alt > args.min_alt]
        if len(seg) < 200:
            continue

        v_ned = zero_phase(seg[:, [IDX['z3_velN'], IDX['z4_velE'], IDX['z5_velD']]], fs, args.fc)
        gyr = zero_phase(seg[:, [IDX['z6_gyrx'], IDX['z7_gyry'], IDX['z8_gyrz']]], fs, args.fc)
        eul = seg[:, [IDX['euler_phi'], IDX['euler_th'], IDX['euler_psi']]]
        a_ned = savgol_filter(v_ned, 11, 3, deriv=1, delta=dt, axis=0)
        w_dot = savgol_filter(gyr, 11, 3, deriv=1, delta=dt, axis=0)

        # 명령에도 응답과 **동일한** 영위상 필터를 건다. 한쪽만 필터링하면 대역이 달라져
        # 게인이 감쇠 편향된다(errors-in-variables).
        u_norm = zero_phase(seg[:, IDX['u0_thrust']] / C_thr_cap, fs, args.fc)
        tau_norm = zero_phase(np.column_stack([seg[:, IDX['u1_tx']] / C_txy_cap,
                                               seg[:, IDX['u2_ty']] / C_txy_cap,
                                               seg[:, IDX['u3_tz']] / C_tz_cap]), fs, args.fc)

        R = rot_ned_from_body(eul[:, 0], eul[:, 1], eul[:, 2])
        v_body = np.einsum('kji,kj->ki', R, v_ned)
        if seg.shape[1] > len(COLS):
            # IMU 비력(specific force, 바디 FRD) — 중력이 이미 빠져 있어 미분/회전 불필요(최고 SNR)
            w_body = m * zero_phase(seg[:, len(COLS):len(COLS)+3], fs, args.fc)
        else:
            f_ned = m * (a_ned - np.array([0.0, 0.0, g]))        # 중력 제외 총 힘(NED)
            w_body = np.einsum('kji,kj->ki', R, f_ned)           # R^T @ f  (NED→body)

        Wx.append(w_body[:, 0]); Vbx.append(v_body[:, 0])
        Wy.append(w_body[:, 1]); Vby.append(v_body[:, 1])
        Wz.append(w_body[:, 2]); Uz.append(u_norm); Vbz.append(v_body[:, 2])

        p, q, r = gyr[:, 0], gyr[:, 1], gyr[:, 2]
        gyro_terms = [(I[1] - I[2]) * q * r, (I[2] - I[0]) * p * r, (I[0] - I[1]) * p * q]
        for k in range(3):
            Rhs[k].append(I[k] * w_dot[:, k] - gyro_terms[k])
            Tq[k].append(tau_norm[:, k])

        # 호버 동작점(추력 평형) — 준정상·수평 구간
        hv = (np.abs(v_ned).max(axis=1) < 0.15) & (np.abs(eul[:, 0]) < 0.05) & (np.abs(eul[:, 1]) < 0.05)
        if hv.any():
            hover_u.append(u_norm[hv])
            hover_ct.append(np.cos(eul[hv, 0]) * np.cos(eul[hv, 1]))

    if not Wz:
        print("[!] 유효 구간이 없습니다.")
        return 1

    cat = lambda L: np.concatenate(L)
    Wx, Vbx = cat(Wx), cat(Vbx)
    Wy, Vby = cat(Wy), cat(Vby)
    Wz, Uz, Vbz = cat(Wz), cat(Uz), cat(Vbz)
    print(f"[*] 적합 샘플: {len(Wz)}")

    # ── 수직: w_z = -C_thrust*u - drag_z*vbz  →  [-u, -vbz] 회귀 ──
    th_z, se_z, r2_z = ols(np.column_stack([-Uz, -Vbz]), Wz)
    C_thrust, drag_z = float(th_z[0]), float(th_z[1])
    # ── 수평 항력 ──
    th_x, se_x, r2_x = ols(np.column_stack([-Vbx]), Wx)
    th_y, se_y, r2_y = ols(np.column_stack([-Vby]), Wy)
    drag_x, drag_y = float(th_x[0]), float(th_y[0])

    print("\n── 병진 ──")
    print(f"  C_thrust = {C_thrust:9.4f}  (±{se_z[0]:.4f})   R²={r2_z:.3f}")
    print(f"  drag_z   = {drag_z:9.4f}  (±{se_z[1]:.4f})")
    print(f"  drag_x   = {drag_x:9.4f}  (±{se_x[0]:.4f})   R²={r2_x:.3f}")
    print(f"  drag_y   = {drag_y:9.4f}  (±{se_y[0]:.4f})   R²={r2_y:.3f}")

    if hover_u:
        hu = cat(hover_u); hc = cat(hover_ct)
        C_hover = float(np.median(m * g / (hu * hc)))
        print(f"\n── 호버 평형 교차검증 ({len(hu)} 샘플) ──")
        print(f"  u_norm(호버) = {np.median(hu):.5f}")
        print(f"  C_thrust(호버평형) = {C_hover:9.4f}   ← 회귀값과 비교")
        print(f"  차이 = {100*(C_thrust-C_hover)/C_hover:+.2f}%")

    # ── 회전 ──
    print("\n── 회전 ──")
    C_tq = []
    for k, name in enumerate(('x(roll)', 'y(pitch)', 'z(yaw)')):
        y, x = cat(Rhs[k]), cat(Tq[k])
        th, se, r2 = ols(np.column_stack([x]), y)
        C_tq.append(float(th[0]))
        print(f"  C_torque_{name:9s} = {th[0]:9.4f}  (±{se[0]:.4f})   R²={r2:.3f}")
    asym = 100 * (C_tq[0] - C_tq[1]) / (0.5 * (C_tq[0] + C_tq[1]))
    print(f"  roll/pitch 비대칭 = {asym:+.1f}%  "
          f"(|비대칭|>10% 면 C_torque_x/y 분리 검토)")

    out = {
        'C_thrust': C_thrust,
        'C_torque_xy': 0.5 * (C_tq[0] + C_tq[1]),
        'C_torque_z': C_tq[2],
        'drag': [drag_x, drag_y, abs(drag_z)],
        'drone': drone,
        'note': f"zu_log 게인 직접적합 ({os.path.basename(args.npz)}, N={len(Wz)}) "
                f"— 총질량 1.372kg 정합 플랜트 기준 (2026-07-28)",
    }
    print("\n── 제안 calibration.json ──")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.write:
        import shutil
        bak = args.calib + '.bak_prefit'
        if not os.path.exists(bak):
            shutil.copy2(args.calib, bak)
            print(f"[*] 백업 → {bak}")
        with open(args.calib, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[*] {args.calib} 기록 완료")
    return 0


if __name__ == '__main__':
    sys.exit(main())
