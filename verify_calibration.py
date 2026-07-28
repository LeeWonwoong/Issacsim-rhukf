#!/usr/bin/env python3
"""verify_calibration.py — 질량정합·재캘리브레이션·프레임수정 검증 (2026-07-28).

검사 항목:
  (1) 호버 전환 시 요 슬루 — 프레임 버그 상태에서는 중앙 84.5°(전환 전 0.7°) 였다. 0에 가까워야 한다.
  (2) 항력 정합 — 오일러가 NED 이면 바디 x/y 항력이 Pegasus 설정 [0.50, 0.30] 로 나와야 한다.
  (3) 호버 동작점 — u_hover 로 환산한 C_thrust 가 calibration.json 과 일치해야 한다.

사용:
  ~/isaacsim/python.sh verify_calibration.py results_verify_frame
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

ZU = ('episode reset attack action z0_gpsN z1_gpsE z2_gpsD z3_velN z4_velE z5_velD '
      'z6_gyrx z7_gyry z8_gyrz u0_thrust u1_tx u2_ty u3_tz euler_phi euler_th euler_psi '
      'atk_scale atk_delay').split()


def zp(x, fs, fc=5.0):
    b, a = butter(4, fc / (0.5 * fs), btype='low')
    return filtfilt(b, a, x, axis=0)


def check_yaw(outdir):
    p = os.path.join(outdir, 'zu_log.npz')
    if not os.path.exists(p):
        return print(f"[!] {p} 없음 — 요 슬루 검사 생략")
    d = np.load(p, allow_pickle=True)
    A = d['data']
    c = {n: i for i, n in enumerate(ZU)}
    act, psi, ep = A[:, c['action']], A[:, c['euler_psi']], A[:, c['episode']]
    tr = [i for i in np.flatnonzero((act[1:] == 1) & (act[:-1] == 0)) + 1
          if i > 200 and i + 150 < len(A) and ep[i] == ep[i + 150]]
    print(f"\n(1) 호버 전환 요 슬루 — 전환 {len(tr)}회")
    if not tr:
        return print("    (전환 없음)")
    after = np.array([np.abs(np.unwrap(psi[i:i + 150] - psi[i])).max() for i in tr])
    before = np.array([np.abs(np.unwrap(psi[i - 150:i] - psi[i - 150])).max() for i in tr])
    print(f"    전환 후 3s |Δψ|max: 중앙 {np.degrees(np.median(after)):6.1f}°  "
          f"90pct {np.degrees(np.percentile(after, 90)):6.1f}°")
    print(f"    전환 전 3s |Δψ|max: 중앙 {np.degrees(np.median(before)):6.1f}°   (기준선)")
    verdict = "OK (슬루 소멸)" if np.median(after) < np.radians(15) else "★ 여전히 슬루 발생"
    print(f"    → {verdict}   [수정 전 기록: 중앙 84.5° / 전환전 0.7°]")


def check_drag_thrust(outdir, calib):
    p = os.path.join(outdir, 'sysid_log.npz')
    if not os.path.exists(p):
        return print(f"[!] {p} 없음 — 항력/추력 검사 생략")
    d = np.load(p, allow_pickle=True)
    dt = float(d['dt']); fs = 1 / dt
    v, eul, acc = d['velocity'], d['euler'], d['accelerometer']
    m, g = float(calib['drone']['mass']), float(calib['drone']['g'])
    vf, af = zp(v, fs), zp(acc, fs)
    phi, th, psi = eul[:, 0], eul[:, 1], eul[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(th), np.sin(th)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.empty((len(phi), 3, 3))
    R[:, 0] = np.column_stack([ct * cps, sp * st * cps - cp * sps, cp * st * cps + sp * sps])
    R[:, 1] = np.column_stack([ct * sps, sp * st * sps + cp * cps, cp * st * sps - cps * sp])
    R[:, 2] = np.column_stack([-st, sp * ct, cp * ct])
    vb = np.einsum('kji,kj->ki', R, vf)
    sp_h = np.linalg.norm(v[:, :2], axis=1)
    msk = sp_h > 1.5
    print(f"\n(2) 항력 정합 — |v_h|>1.5 샘플 {msk.sum()}  (Pegasus 설정 [0.50, 0.30])")
    if msk.sum() < 500:
        print("    표본 부족 — aggressive 구간이 더 필요")
    else:
        for k, nm in enumerate('xy'):
            x, y = vb[msk, k], m * af[msk, k]
            C = -np.sum(x * y) / np.sum(x * x)
            print(f"    drag_{nm} = {C:6.3f}   corr={np.corrcoef(x, y)[0, 1]:+.3f}  "
                  f"(목표 {[0.50, 0.30][k]:.2f})")

    lvl = (np.abs(v).max(axis=1) < 0.15) & (np.abs(phi) < 0.05) & (np.abs(th) < 0.05)
    print(f"\n(3) 호버 동작점 — 준정상 샘플 {lvl.sum()}")
    if lvl.sum() > 100:
        u = np.median(np.abs(d['thrust'][lvl, 2]))
        print(f"    u_hover = {u:.5f}  →  C_thrust(호버평형) = {m*g/u:.4f}  "
              f"(calibration.json {calib['C_thrust']:.4f}, "
              f"차이 {100*(m*g/u - calib['C_thrust'])/calib['C_thrust']:+.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--calib', default='calibration/calibration.json')
    args = ap.parse_args()
    calib = json.load(open(args.calib))
    print(f"[*] {args.outdir} 검증 (calib: m={calib['drone']['mass']}, "
          f"C_thrust={calib['C_thrust']:.3f})")
    check_yaw(args.outdir)
    check_drag_thrust(args.outdir, calib)
    return 0


if __name__ == '__main__':
    sys.exit(main())
