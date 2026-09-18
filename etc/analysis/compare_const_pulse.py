#!/usr/bin/env python3
"""compare_const_pulse.py — 상수 bias vs FDI 펄스 공격 비교 (2026-08-14)

  ~/isaacsim/python.sh compare_const_pulse.py <const_dir> <pulse_dir> [--auth 4.36]

같은 peak ‖a‖·같은 궤적에서 상수 주입 vs 펄스(on/off) 주입을 비교:
  결과성(생존·표류) + 탐지 시그니처(NIS p95=스파이크 / mean=지속).
hover 패턴(baseline≈0) 기준. 펄스는 mean이 낮아도 p95(온셋 스파이크)는 유지되는지가 관건.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load(outdir):
    surv = defaultdict(list); drift = defaultdict(list); nis = defaultdict(list)
    sp = os.path.join(outdir, 'sweep_summary.csv'); dp = os.path.join(outdir, 'sweep_detail.csv')
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp)):
            try: surv[(r['pattern'], float(r['bias']), r['policy'])].append(int(r['survived']))
            except (ValueError, KeyError): pass
    if os.path.exists(dp):
        for r in csv.DictReader(open(dp)):
            try:
                if r['policy'] != 'track': continue
                if r['attack_active'] != '1': continue
                k = (r['pattern'], float(r['bias']))
                drift[k].append(float(r['gt_err'])); nis[k].append(float(r['nis_g_raw']))
            except (ValueError, KeyError): pass
    return surv, drift, nis


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    auth = 4.36
    for a in sys.argv[1:]:
        if a.startswith('--auth'): auth = float(a.split('=', 1)[1])
    if len(args) < 2:
        print("사용: compare_const_pulse.py <const_dir> <pulse_dir>"); return
    cdir, pdir = args[0], args[1]
    cs, cd, cn = load(cdir)
    ps, pd, pn = load(pdir)
    pat = 'hover'
    biases = sorted({k[1] for k in cs if k[0] == pat} | {k[1] for k in ps if k[0] == pat})
    print(f"\n{'='*76}\n 상수({cdir}) vs 펄스({pdir})  [hover, track=무방어]\n{'='*76}")
    print(f" {'δ':>4s} | {'상수:생존 표류p95 NISp95 NIS평균':>34s} | {'펄스:생존 표류p95 NISp95 NIS평균':>34s}")
    for b in biases:
        d = b / auth
        def stat(sv, dr, ns):
            s = np.mean(sv.get((pat, b, 'track'), [np.nan]))
            dd = np.nanpercentile(dr.get((pat, b), [np.nan]), 95)
            n95 = np.nanpercentile(ns.get((pat, b), [np.nan]), 95)
            nmu = np.nanmean(ns.get((pat, b), [np.nan]))
            return s, dd, n95, nmu
        cs_, cd_, cn95, cnmu = stat(cs, cd, cn)
        ps_, pd_, pn95, pnmu = stat(ps, pd, pn)
        print(f" {d:4.2f} | {cs_:6.2f} {cd_:7.2f} {cn95:7.0f} {cnmu:7.0f}       | "
              f"{ps_:6.2f} {pd_:7.2f} {pn95:7.0f} {pnmu:7.0f}")
    print("\n 읽는 법: 표류↓+생존↑ = 펄스가 덜 결과적.  NISp95 유지 but NIS평균↓ = 펄스=스파이크형(CUSUM 누적 불리·RL 윈도우 유리).")


if __name__ == '__main__':
    main()
