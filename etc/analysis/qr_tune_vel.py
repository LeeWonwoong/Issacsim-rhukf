#!/usr/bin/env python3
"""qr_tune_vel.py — vel 채널 Q·R 오프라인 튜닝 (gyro는 고정) (2026-08-15)

  ~/isaacsim/python.sh qr_tune_vel.py <zu_log.npz> [--rg 0.2 --qg 5e-3]

qr_tune_offline.py 가 gyro 채널만 튜닝 → roll/pitch 공격은 표류로 **vel 채널에도** 실리므로
vel R·Q 도 튜닝해 탐지 채널을 둘로(해상도↑). gyro 는 --rg/--qg 로 고정(기본=격자 최적 0.2/5e-3).
측정: vel NIS(res[3:6]) 의 공격 vs 평시 분리도 d′ + 평시 바닥.
  R_vel  → R[3,4,5] (GPS vel 측정노이즈),  Q_vel → Q[6,7,8] (vel 상태 프로세스노이즈)
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration  # noqa: E402

R_VEL = [0.1, 0.3, 0.5, 1.0]
Q_VEL = [5e-3, 1e-3, 5e-4, 1e-4]


def rebuild(calib, rg, qg, rv, qv):
    u = DynamicsUKF(dt=0.02, calib=calib)
    for i in (6, 7, 8):           # gyro 고정
        u.R[i, i] = rg; u.Q[i + 3, i + 3] = qg
    for i in (3, 4, 5):           # vel 스윕
        u.R[i, i] = rv; u.Q[i + 3, i + 3] = qv
    return u


def replay(Z, U, FRESH, EP, ATK, calib, rg, qg, rv, qv):
    benign, onset, sustain, atk_all, peaks = [], [], [], [], []
    for a, b in EP:
        z, u, fr, atk = Z[a:b], U[a:b], FRESH[a:b], ATK[a:b]
        if z.shape[0] < 20:
            continue
        f = rebuild(calib, rg, qg, rv, qv)
        f.x[0:3] = z[0, 0:3]; f.x[6:9] = z[0, 3:6]; f.x[9:12] = z[0, 6:9]
        atk_start = None; ep = []
        for k in range(z.shape[0]):
            res, Pzz = f.step(z[k], u[k], gps_fresh=bool(fr[k]))
            v, _ = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3.0)   # ★ vel NIS
            if not fr[k]:
                continue
            if atk[k] > 0.5:
                atk_all.append(v); ep.append(v)
                if atk_start is None:
                    atk_start = k
                d = sum(1 for j in range(atk_start, k + 1) if fr[j])
                if 1 <= d <= 5:
                    onset.append(v)
                elif 6 <= d <= 30:
                    sustain.append(v)
            else:
                benign.append(v)
        if ep:
            peaks.append(np.percentile(ep, 95))
    return (np.array(benign), np.array(onset), np.array(sustain),
            np.array(atk_all), np.array(peaks))


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return 0.0
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zu')
    ap.add_argument('--rg', type=float, default=0.2, help='gyro R 고정(기본 격자최적 0.2)')
    ap.add_argument('--qg', type=float, default=5e-3, help='gyro Q 고정(기본 5e-3)')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.dirname(a.zu) or '.'

    d = np.load(a.zu)
    arr = d['data'] if 'data' in d else d[d.files[0]]
    RESET, ATK = arr[:, 1], arr[:, 2]
    Z, U = arr[:, 4:13], arr[:, 13:17]
    dz = np.abs(np.diff(Z[:, 0:6], axis=0)).sum(axis=1)
    FRESH = np.concatenate([[True], dz > 1e-9])
    calib = load_calibration()
    starts = list(np.where(RESET > 0.5)[0])
    if not starts or starts[0] != 0:
        starts = [0] + starts
    EP = [(starts[i], starts[i + 1] if i + 1 < len(starts) else arr.shape[0])
          for i in range(len(starts))]
    print(f"zu_log: {arr.shape[0]} 스텝 · {len(EP)} 에피소드 · gyro 고정 R={a.rg} Q={a.qg:.0e}")
    print(f"\n{'R_vel':>5s} {'Q_vel':>7s} | {'평시p50':>7s} {'평시p95':>7s} | "
          f"{'d′온셋':>7s} {'d′peak':>7s} {'d′평균':>7s} | {'지속비':>6s} {'채택':>4s}")
    print('-' * 74)
    FLOOR_MAX = 2.0
    results = []
    for rv, qv in itertools.product(R_VEL, Q_VEL):
        ben, ons, sus, atk, peaks = replay(Z, U, FRESH, EP, ATK, calib, a.rg, a.qg, rv, qv)
        d_on = dprime(ons, ben); d_pk = dprime(peaks, ben) if peaks.size >= 3 else 0.0
        d_mn = dprime(atk, ben)
        pers = (sus.mean() / ons.mean()) if (ons.size and ons.mean() > 1e-6 and sus.size) else np.nan
        p95 = float(np.percentile(ben, 95)); ok = p95 <= FLOOR_MAX
        results.append(dict(r_vel=rv, q_vel=qv, benign_p50=float(np.median(ben)), benign_p95=p95,
                            d_onset=float(d_on), d_peak=float(d_pk), d_mean=float(d_mn),
                            persist=float(pers) if np.isfinite(pers) else None, floor_ok=bool(ok)))
        print(f"{rv:5.2f} {qv:7.0e} | {np.median(ben):7.2f} {p95:7.2f} | "
              f"{d_on:7.2f} {d_pk:7.2f} {d_mn:7.2f} | "
              f"{pers if np.isfinite(pers) else float('nan'):6.2f} {'✓' if ok else '✗':>4s}")
    cands = [r for r in results if r['floor_ok']] or results
    cands.sort(key=lambda r: (r['d_onset'], r['d_peak']), reverse=True)
    best = cands[0]
    cur = next((r for r in results if r['r_vel'] == 0.3 and abs(r['q_vel'] - 5e-3) < 1e-9), None)
    print(f"\n★ vel 최적(바닥 p95≤{FLOOR_MAX}): R_vel={best['r_vel']} Q_vel={best['q_vel']:.0e} "
          f"| d′온셋 {best['d_onset']:.2f} peak {best['d_peak']:.2f} 평시p50 {best['benign_p50']:.2f}")
    if cur:
        print(f"   현행(R_vel0.3 Q5e-3): d′온셋 {cur['d_onset']:.2f} peak {cur['d_peak']:.2f} 평시p50 {cur['benign_p50']:.2f}")
    print("\n※ vel d′가 gyro d′(온셋2.03/peak3.41)보다 낮으면 vel은 보조채널. 둘 다 높으면 2채널 탐지 가능.")
    with open(os.path.join(out, 'qr_tune_vel_result.json'), 'w') as f:
        json.dump({'grid': results, 'best': best, 'gyro_fixed': [a.rg, a.qg]}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
