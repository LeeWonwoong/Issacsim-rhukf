#!/usr/bin/env python3
"""pick_qr_threshold.py — roll 공격 스윕에서 '추락 직전' 세기를 고른다 (2026-08-13)

기준
  · track 정책(무방어) 생존율로 추락을 판정한다.
  · '추락 직전' = track 생존율이 아직 높은(≥0.7) **가장 큰** bias.
    그보다 세면 무방어로 추락 → 탐지-대응이 의미 없어진다. 그 직전에서 Q·R 을 튜닝한다.
  · 패턴마다 취약도가 다르므로 **전 패턴 종합**(가장 취약한 패턴이 살아남는 최대 bias)으로 정한다.
  · 결과를 chosen_bias.txt 로 저장(오케스트레이터가 읽는다).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SURV_MIN = 0.7          # 이 이상이면 '생존'으로 본다


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    detail = os.path.join(out, 'sweep_detail.csv')
    summ = os.path.join(out, 'sweep_summary.csv')

    # ── 생존율: summary 에서 (bias, pattern, policy) 별 survived 평균 ──
    surv = defaultdict(list)
    with open(summ, newline='') as f:
        for r in csv.DictReader(f):
            try:
                surv[(float(r['bias']), r['pattern'], r['policy'])].append(int(r['survived']))
            except (ValueError, KeyError):
                pass
    biases = sorted({k[0] for k in surv if k[0] > 0})
    patterns = sorted({k[1] for k in surv})

    print("\n[생존율] track 정책 (무방어) — bias × 패턴")
    print(f"  {'bias':>7s} " + ' '.join(f'{p[:9]:>10s}' for p in patterns) + f" {'최소':>7s}")
    worst_by_bias = {}
    for b in biases:
        row = []
        for p in patterns:
            v = surv.get((b, p, 'track'), [])
            row.append(np.mean(v) if v else np.nan)
        mn = np.nanmin(row) if any(np.isfinite(row)) else np.nan
        worst_by_bias[b] = mn
        print(f"  {b:7.3f} " + ' '.join(f'{x:10.2f}' for x in row) + f" {mn:7.2f}")

    # ── 추락 직전 = worst 생존율이 아직 ≥ SURV_MIN 인 최대 bias ──
    survivable = [b for b in biases if np.isfinite(worst_by_bias[b]) and worst_by_bias[b] >= SURV_MIN]
    if survivable:
        chosen = max(survivable)
        reason = f"worst-pattern 생존율 {worst_by_bias[chosen]:.2f} ≥ {SURV_MIN}"
    else:
        # 전부 취약하면 가장 약한 공격을 택함(그래도 탐지 대상은 됨)
        chosen = min(biases)
        reason = f"모든 세기가 worst<{SURV_MIN} — 최소 세기 채택(전 패턴 취약)"

    delta = chosen / 4.36
    print(f"\n[NIS] bias={chosen:.3f} (δ≈{delta:.2f}) 에서 gyro NIS — 공격 vs 평시")
    # detail 에서 이 bias 의 track 셀: attack_active 0(공격전)/1(공격중)
    ben, atk = defaultdict(list), defaultdict(list)
    with open(detail, newline='') as f:
        for r in csv.DictReader(f):
            try:
                if abs(float(r['bias']) - chosen) > 1e-6 or r['policy'] != 'track':
                    continue
                g = float(r['nis_g_raw'])
                (atk if r['attack_active'] not in ('0', '0.0', 'False', '') else ben)[r['pattern']].append(g)
            except (ValueError, KeyError):
                pass
    print(f"  {'패턴':12s} {'평시 p50':>9s} {'공격 p50':>9s} {'공격 p95':>9s} {'비':>6s}")
    for p in patterns:
        b0, a0 = np.array(ben.get(p, [np.nan])), np.array(atk.get(p, [np.nan]))
        ratio = np.median(a0)/np.median(b0) if np.median(b0) > 0 else np.nan
        print(f"  {p:12s} {np.median(b0):9.2f} {np.median(a0):9.2f} {np.percentile(a0,95):9.2f} {ratio:6.1f}")

    with open(os.path.join(out, 'chosen_bias.txt'), 'w') as f:
        f.write(f'{chosen:.3f}')
    print(f"\n★ 선정: bias={chosen:.3f}  (정규화 δ≈{delta:.2f}, 제어권한의 {delta*100:.0f}%)")
    print(f"  근거: {reason}")

    # 표를 파일로도
    with open(os.path.join(out, 'roll_crash_analysis.txt'), 'w') as f:
        f.write(f"roll 공격 크래시 스윕 분석 (2026-08-13)\n")
        f.write(f"선정 bias={chosen:.3f} (δ≈{delta:.2f}) — {reason}\n\n")
        f.write("track 생존율 (bias × 패턴):\n")
        f.write(f"  {'bias':>7s} " + ' '.join(f'{p[:9]:>10s}' for p in patterns) + "\n")
        for b in biases:
            row = [surv.get((b, p, 'track'), []) for p in patterns]
            f.write(f"  {b:7.3f} " + ' '.join(f'{np.mean(v) if v else float("nan"):10.2f}' for v in row) + "\n")


if __name__ == '__main__':
    main()
