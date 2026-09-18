#!/usr/bin/env python3
"""analyze_pulse_recovery.py — 펄스 공격의 복귀 시상수 + 슬라이딩 윈도우 효과 (2026-08-16)

  ~/isaacsim/python.sh analyze_pulse_recovery.py <pulse_zu.npz> [--rg 0.2 --qg 5e-3 --rv 0.1]

두 가지:
  ① 온셋/복귀 시상수: 버스트 OFF→ON(온셋) / ON→OFF(복귀) 전환 후 gyro NIS 시간거동.
     복귀 τ = 공격 OFF 후 NIS 가 절반으로 떨어지는 fresh 스텝 수 (양방향 stopping 전제).
  ② 슬라이딩 윈도우(4) 효과: 캠페인(step≥30, ON/OFF 혼재) 를 탐지할 때
     단일 NIS[t] vs 윈도우-max(최근4) 의 분리도 d′.  펄스는 OFF 간격에 단일값이 꺼져
     평시와 aliasing → 윈도우-max 가 최근 스파이크를 기억해 캠페인을 유지 탐지하는지.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration  # noqa: E402


def rebuild(calib, rg, qg, rv):
    u = DynamicsUKF(dt=0.02, calib=calib)
    for i in (6, 7, 8):
        u.R[i, i] = rg; u.Q[i + 3, i + 3] = qg
    for i in (3, 4, 5):
        u.R[i, i] = rv
    return u


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return 0.0
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zu')
    ap.add_argument('--rg', type=float, default=0.2)
    ap.add_argument('--qg', type=float, default=5e-3)
    ap.add_argument('--rv', type=float, default=0.1)
    a = ap.parse_args()
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
    EP = [(starts[i], starts[i + 1] if i + 1 < len(starts) else arr.shape[0]) for i in range(len(starts))]
    print(f"zu_log: {arr.shape[0]} 스텝 · {len(EP)} 에피소드 · gyro R={a.rg} Q={a.qg:.0e} vel R={a.rv}")

    onset_prof = defaultdict(list)     # ON 후 fresh경과 → NIS
    recov_prof = defaultdict(list)     # OFF 후 fresh경과 → NIS
    # 윈도우 효과: 캠페인(공격 첫 ON 이후) 의 단일 vs 윈도우-max
    single_camp, winmax_camp, benign = [], [], []
    recov_tau = []

    for a0, b0 in EP:
        z, u, fr, atk = Z[a0:b0], U[a0:b0], FRESH[a0:b0], ATK[a0:b0]
        if z.shape[0] < 20:
            continue
        f = rebuild(calib, a.rg, a.qg, a.rv)
        f.x[0:3] = z[0, 0:3]; f.x[6:9] = z[0, 3:6]; f.x[9:12] = z[0, 6:9]
        nis_hist = []          # fresh 스텝별 gyro NIS
        atk_hist = []          # fresh 스텝별 atk
        for k in range(z.shape[0]):
            res, Pzz = f.step(z[k], u[k], gps_fresh=bool(fr[k]))
            if not fr[k]:
                continue
            g, _ = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)
            nis_hist.append(g); atk_hist.append(atk[k] > 0.5)
        nis_hist = np.array(nis_hist); atk_hist = np.array(atk_hist)
        n = len(nis_hist)
        if n < 8:
            continue
        camp_start = np.argmax(atk_hist) if atk_hist.any() else None
        # 전환 프로파일
        for i in range(1, n):
            if atk_hist[i] and not atk_hist[i - 1]:          # OFF→ON 온셋
                for dd in range(0, 6):
                    if i + dd < n:
                        onset_prof[dd].append(nis_hist[i + dd])
            if (not atk_hist[i]) and atk_hist[i - 1]:        # ON→OFF 복귀
                base = nis_hist[i - 1]
                half = base * 0.5
                tau = None
                for dd in range(0, 10):
                    if i + dd < n:
                        recov_prof[dd].append(nis_hist[i + dd])
                        if tau is None and nis_hist[i + dd] <= half:
                            tau = dd
                if tau is not None:
                    recov_tau.append(tau)
        # 윈도우 효과: 캠페인 vs 평시(캠페인 전)
        for i in range(n):
            if camp_start is not None and i >= camp_start:
                single_camp.append(nis_hist[i])
                winmax_camp.append(nis_hist[max(0, i - 3):i + 1].max())
            elif camp_start is None or i < camp_start:
                benign.append(nis_hist[i])

    print("\n① 온셋/복귀 시상수 (gyro NIS, fresh 스텝 기준 평균)")
    print("  경과fresh:  0     1     2     3     4     5")
    print("  온셋(ON후):", ' '.join('%5.0f' % np.mean(onset_prof.get(dd, [np.nan])) for dd in range(6)))
    print("  복귀(OFF후):", ' '.join('%5.0f' % np.mean(recov_prof.get(dd, [np.nan])) for dd in range(6)))
    if recov_tau:
        print(f"  복귀 τ(절반까지): 중앙 {np.median(recov_tau):.1f} fresh스텝 · 평균 {np.mean(recov_tau):.1f} "
              f"(n={len(recov_tau)})  → OFF 후 반감에 {np.median(recov_tau):.0f}스텝")
    print(f"  복귀 프로파일 길이 10: ", ' '.join('%5.0f' % np.mean(recov_prof.get(dd, [np.nan])) for dd in range(10)))

    print("\n② 슬라이딩 윈도우(4) 효과 — 캠페인(ON/OFF혼재) 탐지 분리도 d′ (vs 평시)")
    sc, wc, bn = np.array(single_camp), np.array(winmax_camp), np.array(benign)
    print(f"  표본: 캠페인 {sc.size} · 평시 {bn.size}")
    print(f"  단일 NIS[t]     : 캠페인 p50 {np.median(sc):6.0f} · d′ {dprime(sc, bn):.2f}")
    print(f"  윈도우-max(4)   : 캠페인 p50 {np.median(wc):6.0f} · d′ {dprime(wc, bn):.2f}")
    # OFF 간격에서 단일이 꺼지는 비율(aliasing) vs 윈도우가 유지하는 비율
    thr = np.percentile(bn, 95) if bn.size else 1.0
    off_single_low = np.mean(sc < thr) if sc.size else np.nan     # 캠페인 중 단일이 평시수준으로 떨어진 비율
    off_win_low = np.mean(wc < thr) if wc.size else np.nan
    print(f"  평시 p95 임계 {thr:.1f} 아래로 떨어진 캠페인 비율(=놓침): "
          f"단일 {off_single_low*100:.0f}%  vs  윈도우 {off_win_low*100:.0f}%")
    print("  → 윈도우 d′ > 단일 d′ 이고 놓침 비율↓ 이면: 슬라이딩 윈도우가 펄스 aliasing 완화(관측구조 정당화)")


if __name__ == '__main__':
    main()
