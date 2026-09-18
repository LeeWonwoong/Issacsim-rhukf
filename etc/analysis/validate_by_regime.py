#!/usr/bin/env python3
"""validate_by_regime.py — 캘리브레이션이 비행 영역(호버/직진/급기동)에 무관하게 성립하는지 검증.

aggressive 데이터로 적합한 계수가 호버·저속직진에서도 맞는지 확인한다.
**재적합이 아니라 고정 계수의 예측잔차**를 본다 — 이게 UKF 가 실제로 겪는 오차다.

  병진(바디 FRD):  f_pred = [-drag_x·vbx, -drag_y·vby, -C_thrust·u_norm - drag_z·vbz]
                   f_meas = m · (IMU 비력)
  회전:            ω̇_pred = (C_torque_a·τ_a + (I_b-I_c)·ω_b·ω_c) / I_a
                   ω̇_meas = 자이로 미분

영역 구분(기체 상태 기준, 패턴 이름과 독립):
  hover  : |v|<0.3 m/s 이고 |ω|<0.15 rad/s
  cruise : |v_h|>1.0 m/s 이고 |ω|<0.5 rad/s   (직진/완만한 선회)
  agile  : |ω|>1.0 rad/s

사용:
  ~/isaacsim/python.sh validate_by_regime.py results_verify_frame [results_regime_waypoint ...]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def zp(x, fs, fc=8.0):
    b, a = butter(4, fc / (0.5 * fs), btype='low')
    return filtfilt(b, a, x, axis=0)


def rot(phi, th, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(th), np.sin(th)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.empty((len(phi), 3, 3))
    R[:, 0] = np.column_stack([ct * cps, sp * st * cps - cp * sps, cp * st * cps + sp * sps])
    R[:, 1] = np.column_stack([ct * sps, sp * st * sps + cp * cps, cp * st * sps - cps * sp])
    R[:, 2] = np.column_stack([-st, sp * ct, cp * ct])
    return R


def load(path, calib):
    d = np.load(os.path.join(path, 'sysid_log.npz'), allow_pickle=True)
    dt = float(d['dt']); fs = 1 / dt
    m = float(calib['drone']['mass'])
    I = np.array([float(calib['drone'][k]) for k in ('Ixx', 'Iyy', 'Izz')])
    C_t = float(calib['C_thrust'])
    C_q = np.array([float(calib.get('C_torque_x', calib['C_torque_xy'])),
                    float(calib.get('C_torque_y', calib['C_torque_xy'])),
                    float(calib['C_torque_z'])])
    drag = np.array(calib['drag'])

    v = zp(d['velocity'], fs); eul = d['euler']
    acc = zp(d['accelerometer'], fs); gyr = zp(d['gyro'], fs)
    u = zp(np.abs(d['thrust'][:, 2]), fs); tau = zp(d['torque'], fs)
    wd = savgol_filter(gyr, 11, 3, deriv=1, delta=dt, axis=0)
    vb = np.einsum('kji,kj->ki', rot(eul[:, 0], eul[:, 1], eul[:, 2]), v)

    f_meas = m * acc
    f_pred = np.column_stack([-drag[0] * vb[:, 0], -drag[1] * vb[:, 1],
                              -C_t * u - drag[2] * vb[:, 2]])
    # 총추력에 비례하는 기하 토크 오프셋(로터 중심이 COM 에서 어긋나 생김; PX4 는 트림으로 상쇄)
    K = np.array(calib.get('torque_thrust_coupling', [0.0, 0.0, 0.0]))
    T_tot = C_t * u
    wd_pred = np.empty_like(wd)
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        wd_pred[:, k] = (C_q[k] * tau[:, k] + K[k] * T_tot
                         + (I[a] - I[b]) * gyr[:, a] * gyr[:, b]) / I[k]
    return dict(v=v, vb=vb, gyr=gyr, f_meas=f_meas, f_pred=f_pred,
                wd=wd, wd_pred=wd_pred, m=m, I=I, T=T_tot)


def report(name, D, idx):
    if idx.sum() < 300:
        print(f"  {name:7s} 표본 {idx.sum():6d} — 부족, 생략")
        return
    print(f"  {name:7s} 표본 {idx.sum():6d}  "
          f"|v_h| 중앙 {np.median(np.linalg.norm(D['v'][idx, :2], axis=1)):.2f} m/s  "
          f"|ω| 중앙 {np.median(np.abs(D['gyr'][idx]).max(axis=1)):.2f} rad/s")
    for k, nm in enumerate(('Fx', 'Fy', 'Fz')):
        e = D['f_pred'][idx, k] - D['f_meas'][idx, k]
        s = D['f_meas'][idx, k]
        rel = 100 * e.std() / max(s.std(), 1e-9)
        print(f"      {nm}: 편향 {e.mean():+7.4f} N   RMS {np.sqrt((e**2).mean()):6.4f} N   "
              f"(신호 std {s.std():6.4f} N, 잔차/신호 {rel:5.1f}%)")
    for k, nm in enumerate(('ω̇x', 'ω̇y', 'ω̇z')):
        e = D['wd_pred'][idx, k] - D['wd'][idx, k]
        s = D['wd'][idx, k]
        rel = 100 * e.std() / max(s.std(), 1e-9)
        print(f"      {nm}: 편향 {e.mean():+7.3f}      RMS {np.sqrt((e**2).mean()):6.3f} rad/s² "
              f"(신호 std {s.std():6.3f}, 잔차/신호 {rel:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--calib', default='calibration/calibration.json')
    ap.add_argument('--fit-coupling', action='store_true',
                    help='ω̇ 예측 편향으로부터 추력-토크 결합계수를 추정(UKF 프레임에서 직접)')
    ap.add_argument('--write', action='store_true', help='--fit-coupling 결과를 calibration.json 에 기록')
    args = ap.parse_args()
    calib = json.load(open(args.calib))
    print(f"[*] 고정 계수: C_thrust={calib['C_thrust']:.3f} "
          f"C_tq=({calib.get('C_torque_x', calib['C_torque_xy']):.3f},"
          f"{calib.get('C_torque_y', calib['C_torque_xy']):.3f},{calib['C_torque_z']:.3f}) "
          f"drag={calib['drag']}")
    for path in args.dirs:
        if not os.path.exists(os.path.join(path, 'sysid_log.npz')):
            print(f"\n[!] {path}: sysid_log.npz 없음 — 생략")
            continue
        D = load(path, calib)
        spd = np.linalg.norm(D['v'][:, :2], axis=1)
        wmag = np.abs(D['gyr']).max(axis=1)
        print(f"\n=== {path} ({len(spd)} 샘플) ===")
        report('hover', D, (np.linalg.norm(D['v'], axis=1) < 0.3) & (wmag < 0.15))
        report('cruise', D, (spd > 1.0) & (wmag < 0.5))
        report('agile', D, wmag > 1.0)
        report('전체', D, np.ones(len(spd), bool))

        if args.fit_coupling:
            #  예측편향(pred−meas) 을 0 으로 만드는 추가 토크: Δτ = −bias·I,  K = Δτ / 총추력
            cur = np.array(calib.get('torque_thrust_coupling', [0.0, 0.0, 0.0]))
            bias = (D['wd_pred'] - D['wd']).mean(axis=0)
            K = cur - bias * D['I'] / D['T'].mean()
            print(f"\n  [fit-coupling] 현재 K={np.round(cur, 6).tolist()}  "
                  f"편향={np.round(bias, 3).tolist()} rad/s²")
            print(f"                 제안 K={np.round(K, 6).tolist()}  (τ_offset = K·총추력[N])")
            if args.write:
                import shutil
                shutil.copy2(args.calib, args.calib + '.bak_precoupling')
                calib['torque_thrust_coupling'] = [float(x) for x in K]
                with open(args.calib, 'w') as f:
                    json.dump(calib, f, indent=2, ensure_ascii=False)
                print(f"                 → {args.calib} 기록 (백업 .bak_precoupling)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
