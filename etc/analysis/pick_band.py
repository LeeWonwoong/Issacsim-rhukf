#!/usr/bin/env python3
"""pick_band.py — 측정 결과에서 하이재킹 밴드를 산출해 E 스윕용 bias(Nm) 를 emit (2026-08-13)

  ~/isaacsim/python.sh pick_band.py <outdir1> [<outdir2> ...] [--auth 4.36] [--n 5]

밴드 정의 (hover 패턴 기준 — baseline≈0 이라 표류=순수 하이재킹량):
  track 미추락(survive=1) ∧ track표류p95 > DRIFT_MIN ∧ dhover표류p95 < DHOVER_MAX ∧ dhover survive=1
  = 결과성(끌림) ∧ 대응가능(호버로 억제·생존). flip 구간은 track survive<1 로 자동 제외.

stdout 마지막 줄 = E 용 쉼표구분 Nm bias (band 안에서 최대 N점 균등). 밴드 없으면 fallback.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

DRIFT_MIN = 0.5     # track 표류가 이보다 크면 "끌림"(결과성)
DHOVER_MAX = 0.3    # dhover 표류가 이보다 작으면 "억제됨"(대응가능)


def load(outdirs):
    surv = defaultdict(list); drift = defaultdict(list)
    for od in outdirs:
        sp = os.path.join(od, 'sweep_summary.csv'); dp = os.path.join(od, 'sweep_detail.csv')
        if os.path.exists(sp):
            for r in csv.DictReader(open(sp)):
                try: surv[(r['pattern'], float(r['bias']), r['policy'])].append(int(r['survived']))
                except (ValueError, KeyError): pass
        if os.path.exists(dp):
            for r in csv.DictReader(open(dp)):
                try:
                    if r['attack_active'] != '1': continue
                    drift[(r['pattern'], float(r['bias']), r['policy'])].append(float(r['gt_err']))
                except (ValueError, KeyError): pass
    return surv, drift


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    auth = 4.36; npick = 5
    for a in sys.argv[1:]:
        if a.startswith('--auth'): auth = float(a.split('=', 1)[1]) if '=' in a else auth
        if a.startswith('--n'):    npick = int(a.split('=', 1)[1]) if '=' in a else npick
    surv, drift = load(args)
    pat = 'hover'   # 깨끗한 판정 패턴
    biases = sorted({k[1] for k in surv if k[0] == pat})
    delays = sorted({int(k[2][6:]) for k in surv if k[2].startswith('dhover')})
    band = []
    print(f"# pick_band  patterns-source={args}  auth={auth}", file=sys.stderr)
    print(f"#  δ    Nm  | trkSv trkDrift dhDrift dhSv | in-band", file=sys.stderr)
    for b in biases:
        tsv = np.mean(surv.get((pat, b, 'track'), [0]))
        tdr = np.percentile(drift.get((pat, b, 'track'), [0]), 95)
        dh_all = [x for d in delays for x in drift.get((pat, b, f'dhover{d}'), [])] or [9]
        ddr = np.percentile(dh_all, 95)
        dsv = np.mean([s for d in delays for s in surv.get((pat, b, f'dhover{d}'), [])] or [0])
        inb = (tsv >= 0.99) and (tdr > DRIFT_MIN) and (ddr < DHOVER_MAX) and (dsv >= 0.99)
        if inb: band.append(b)
        print(f"# {b/auth:4.2f} {b:5.2f} | {tsv:5.2f} {tdr:7.2f} {ddr:6.2f} {dsv:5.2f} | {'★' if inb else ''}",
              file=sys.stderr)
    if band:
        lo, hi = min(band), max(band)
        print(f"# BAND δ∈[{lo/auth:.2f},{hi/auth:.2f}]  Nm∈[{lo:.2f},{hi:.2f}]  ({len(band)}점)", file=sys.stderr)
        # 밴드 안에서 최대 npick 점 균등 선택
        if len(band) <= npick:
            pick = band
        else:
            idx = np.linspace(0, len(band) - 1, npick).round().astype(int)
            pick = [band[i] for i in sorted(set(idx))]
    else:
        pick = [1.74, 2.18, 2.62, 3.05]   # fallback: δ0.4~0.7
        print(f"# ⚠ 밴드 비어있음 — fallback {pick}", file=sys.stderr)
    print(','.join(f'{x:.2f}' for x in pick))   # stdout: E 용 bias


if __name__ == '__main__':
    main()
