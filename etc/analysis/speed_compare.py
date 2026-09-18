#!/usr/bin/env python3
"""speed_compare.py — 속도 1× vs 1.5× 의 raw innovation·NIS 를 패턴·구간별 비교 (2026-08-17)

  ~/isaacsim/python.sh speed_compare.py <dir_1x> <dir_1p5x>

핵심 질문: 기동(circle/fig8/aggressive)의 평시 innovation(res_g/res_v)이 속도↑에 커지나?
  = 플랜트 비선형(2차추력+모터지연)을 UKF 선형근사가 못 따라가는 오차가 속도의존인가.
구간: clean(<80,바람·공격 없는 순수기동) / overlap(140~220,공격+바람). track 정책, 강풍셀 기준.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def region(step):
    if step < 80: return 'clean'
    if step < 140: return 'wind'
    if step < 220: return 'overlap'
    if step < 260: return 'atk'
    return 'recov'


def load(outdir):
    # (pattern, dtype, bias, region) -> [(res_v,res_g,nis_v,nis_g,speed_xy)]
    agg = defaultdict(list)
    dp = os.path.join(outdir, 'sweep_detail.csv')
    if not os.path.exists(dp):
        return None
    for r in csv.DictReader(open(dp)):
        if r['policy'] != 'track':
            continue
        try:
            k = (r['pattern'], r['disturbance_type'], float(r['bias']), region(int(r['step'])))
            agg[k].append((float(r['res_v']), float(r['res_g']),
                           float(r['nis_v_raw']), float(r['nis_g_raw']), float(r['speed_xy'])))
        except (ValueError, KeyError):
            pass
    return agg


def p50(agg, k, col):
    v = agg.get(k, [])
    return np.median([x[col] for x in v]) if v else float('nan')


def main():
    d1 = sys.argv[1] if len(sys.argv) > 1 else 'results_speed_1p0'
    d2 = sys.argv[2] if len(sys.argv) > 2 else 'results_speed_1p5'
    a1, a2 = load(d1), load(d2)
    if a1 is None or a2 is None:
        print("detail 없음"); return
    pats = sorted({k[0] for k in a1})

    print(f"\n{'='*94}")
    print(f" 속도 1.0×({d1}) vs 1.5×({d2}) — 평시기동 innovation & 속도 & 공격 (p50)")
    print(f"{'='*94}")
    print("  [clean 구간 = 순수 기동(바람·공격 없음)] — 여기 res_g 가 속도↑에 커지면 = 기동 모델오차 속도의존")
    print(f"  {'패턴':>10s} | {'speed 1.0×→1.5×':>16s} | {'res_g 1.0×→1.5×':>18s} | {'res_v 1.0×→1.5×':>18s} | {'NIS_g 1.0→1.5':>14s}")
    for pat in pats:
        k = (pat, 'none', 0.0, 'clean')   # 무풍·평시·순수기동
        sp1, sp2 = p50(a1, k, 4), p50(a2, k, 4)
        rg1, rg2 = p50(a1, k, 1), p50(a2, k, 1)
        rv1, rv2 = p50(a1, k, 0), p50(a2, k, 0)
        ng1, ng2 = p50(a1, k, 3), p50(a2, k, 3)
        print(f"  {pat:>10s} | {sp1:6.2f}→{sp2:6.2f} m/s | {rg1:7.3f}→{rg2:7.3f} | {rv1:7.3f}→{rv2:7.3f} | {ng1:5.2f}→{ng2:5.2f}")

    print("\n  [overlap 구간 = 공격δ0.7+강풍] — 공격 신호(참고)")
    print(f"  {'패턴':>10s} | {'res_g 1.0×→1.5×':>18s} | {'NIS_g 1.0×→1.5×':>16s}")
    for pat in pats:
        k = (pat, 'wind_turbulence', 3.05, 'overlap')
        rg1, rg2 = p50(a1, k, 1), p50(a2, k, 1)
        ng1, ng2 = p50(a1, k, 3), p50(a2, k, 3)
        print(f"  {pat:>10s} | {rg1:7.2f}→{rg2:7.2f} | {ng1:7.1f}→{ng2:7.1f}")

    print("\n─ 읽는 법 ────────────────────────────────────────────")
    print("  · clean res_g 가 속도↑에 커지면 = 급기동 모델오차(모터지연·2차추력)가 속도의존 → 원웅님 직관 ✓")
    print("  · 안 커지면 = 이 속도대(≤2 m/s)에선 모터지연(14ms<20ms스텝)이 여전히 정착 → 속도로는 부족.")


if __name__ == '__main__':
    main()
