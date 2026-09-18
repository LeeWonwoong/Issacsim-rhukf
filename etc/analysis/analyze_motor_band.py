#!/usr/bin/env python3
"""analyze_motor_band.py — 모터 커플링 공격의 3조건 밴드 판정 (2026-08-13)

  ~/isaacsim/python.sh analyze_motor_band.py <outdir> [<outdir2> ...]

프레임워크 3조건 (동시 필요):
  ① 결과성    track(무방어) 추락
  ② 탐지가능  공격 중 NIS 상승 (온셋)
  ③ 대응가능  dhover(d≤3) 생존 ≈ dhover0(즉시 호버)
밴드 = track 추락 ∧ dhover(d≤3) 생존 인 크기 구간.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load(outdir):
    surv = defaultdict(list)      # (bias, pattern, policy) → [survived]
    nis = defaultdict(list)       # (bias, pattern) → gyro NIS 공격중
    nis_pre = defaultdict(list)   # 공격전
    sp = os.path.join(outdir, 'sweep_summary.csv')
    dp = os.path.join(outdir, 'sweep_detail.csv')
    if os.path.exists(sp):
        with open(sp, newline='') as f:
            for r in csv.DictReader(f):
                try:
                    surv[(float(r['bias']), r['pattern'], r['policy'])].append(int(r['survived']))
                except (ValueError, KeyError):
                    pass
    if os.path.exists(dp):
        with open(dp, newline='') as f:
            for r in csv.DictReader(f):
                try:
                    if r['policy'] != 'track':
                        continue
                    k = (float(r['bias']), r['pattern'])
                    g = float(r['nis_g_raw'])
                    (nis if r['attack_active'] not in ('0', '0.0', 'False', '') else nis_pre)[k].append(g)
                except (ValueError, KeyError):
                    pass
    return surv, nis, nis_pre


def main():
    for outdir in sys.argv[1:] or ['results_motor_band']:
        surv, nis, nis_pre = load(outdir)
        if not surv:
            print(f"[{outdir}] 데이터 없음"); continue
        biases = sorted({k[0] for k in surv if abs(k[0]) > 1e-9})
        patterns = sorted({k[1] for k in surv})
        atype = os.path.basename(outdir).replace('results_motor_band_', '').replace('results_', '')
        print(f"\n{'='*72}\n {outdir}  (공격유형 {atype})\n{'='*72}")
        for pat in patterns:
            print(f"\n [{pat}]")
            print(f"  {'크기N':>6s} {'track':>6s} {'dhov0':>6s} {'dhov3':>6s} {'dhov5':>6s} "
                  f"{'NIS전':>6s} {'NIS공':>6s} {'판정':>16s}")
            for b in biases:
                tr = np.mean(surv.get((b, pat, 'track'), [np.nan]))
                d0 = np.mean(surv.get((b, pat, 'dhover0'), [np.nan]))
                d3 = np.mean(surv.get((b, pat, 'dhover3'), [np.nan]))
                d5 = np.mean(surv.get((b, pat, 'dhover5'), [np.nan]))
                np_pre = np.median(nis_pre.get((b, pat), [np.nan]))
                np_atk = np.median(nis.get((b, pat), [np.nan]))
                # 판정
                band = (tr <= 0.5) and (d3 >= 0.7)      # track 추락 ∧ dhover3 생존
                if np.isnan(tr):
                    verd = '표본없음'
                elif tr >= 0.9:
                    verd = '①미성립(안죽음)'
                elif d3 < 0.5:
                    verd = '③실패(구제못함)'
                elif band:
                    verd = '★밴드(①②③)'
                else:
                    verd = '경계'
                print(f"  {b:6.1f} {tr:6.2f} {d0:6.2f} {d3:6.2f} {d5:6.2f} "
                      f"{np_pre:6.2f} {np_atk:6.2f} {verd:>16s}")
        # 요약: 밴드 크기 구간
        band_b = [b for b in biases
                  for pat in patterns
                  if np.mean(surv.get((b, pat, 'track'), [1])) <= 0.5
                  and np.mean(surv.get((b, pat, 'dhover3'), [0])) >= 0.7]
        if band_b:
            print(f"\n  ★ 밴드 존재: 크기 {sorted(set(band_b))} N (track추락 ∧ dhover3생존)")
        else:
            print(f"\n  ✗ 밴드 없음 — 이 공격유형은 프레임워크 미성립 "
                  f"(추락 크기에서 호버 구제 안 됨 = 추력지배)")


if __name__ == '__main__':
    main()
