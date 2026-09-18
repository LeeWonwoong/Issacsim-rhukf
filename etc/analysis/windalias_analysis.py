#!/usr/bin/env python3
"""windalias_analysis.py — 공격강도 × 바람 aliasing 맵 + 윈도우(1/4/8) 효과 (2026-08-18)

  ~/isaacsim/python.sh windalias_analysis.py <outdir>

각 (외란,ws,공격δ): 겹침(공격+바람) vs 바람만 의 분리도 d′ (gyro·vel), 단일/윈4/윈8.
  약공격+강풍서 d′ 낮으면(공격이 바람에 묻힘) = aliasing. 윈도우가 올리면 = 해소.
시간창: 바람 80~220, 공격 140~260.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

A_TQ = 4.36


def region(step):
    if step < 80: return 'clean'
    if step < 140: return 'wind'
    if step < 220: return 'overlap'
    if step < 260: return 'atk'
    return 'recov'


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return float('nan')
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def winmax(v, w):
    return [max(v[max(0, i-w+1):i+1]) for i in range(len(v))]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_windalias_const'
    seq = defaultdict(list)   # (dtype,ws,bias,pat,ep)->[(step,nis_v,nis_g)]
    for r in csv.DictReader(open(os.path.join(outdir, 'sweep_detail.csv'))):
        if r['policy'] != 'track':
            continue
        try:
            k = (r['disturbance_type'], float(r['wind_speed']), float(r['bias']), r['pattern'], r['episode'])
            seq[k].append((int(r['step']), float(r['nis_v_raw']), float(r['nis_g_raw'])))
        except (ValueError, KeyError):
            pass
    # (dtype,ws,bias) -> region -> channel -> {win: [vals]}
    ov = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # overlap: [w1,w4,w8] per ch
    wo = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # wind-only
    for (dtype, ws, bias, pat, ep), rows in seq.items():
        rows.sort(key=lambda x: x[0]); steps = [r[0] for r in rows]
        for ch, ci in [('g', 2), ('v', 1)]:
            vals = [r[ci] for r in rows]
            w4 = winmax(vals, 4); w8 = winmax(vals, 8)
            for i, st in enumerate(steps):
                rg = region(st)
                key = (dtype, ws)
                if bias > 1e-9 and rg == 'overlap':
                    ov[key][ch]['w1'].append((bias, vals[i]))
                    ov[key][ch]['w4'].append((bias, w4[i]))
                    ov[key][ch]['w8'].append((bias, w8[i]))
                elif bias <= 1e-9 and rg == 'wind':
                    wo[key][ch]['w1'].append(vals[i])
                    wo[key][ch]['w4'].append(w4[i])
                    wo[key][ch]['w8'].append(w8[i])

    biases = sorted({k[2] for k in seq if k[2] > 1e-9})
    print(f"\n{'='*104}\n {outdir} — 공격δ × 바람 aliasing: d′(겹침 vs 바람만) 단일/윈4/윈8\n{'='*104}")
    for ch, chn in [('v', 'vel(바람채널)'), ('g', 'gyro(공격채널)')]:
        print(f"\n [{chn}]")
        print(f"  {'외란':>15s} {'ws':>4s} | " + ' | '.join(f'δ{b/A_TQ:.1f}: 단일/윈4/윈8' for b in biases))
        for key in sorted(ov, key=lambda k: (k[0], k[1])):
            dtype, ws = key
            cells = []
            for b in biases:
                def dp(win):
                    at = [v for (bb, v) in ov[key][ch][win] if abs(bb-b) < 1e-6]
                    return dprime(at, wo[key][ch][win])
                cells.append(f"{dp('w1'):4.1f}/{dp('w4'):4.1f}/{dp('w8'):4.1f}")
            print(f"  {dtype:>15s} {ws:4.1f} | " + ' | '.join(cells))
    print("\n─ 읽는 법 ────────────────────────────────────────")
    print("  · vel 채널·강풍·약공격(δ0.3)서 단일 d′ 낮으면 = 공격이 바람에 묻힘(aliasing).")
    print("  · 윈4/윈8 d′ > 단일 이면 = 슬라이딩 윈도우가 그 aliasing 해소. 강풍서 격차 크면 강한 근거.")


if __name__ == '__main__':
    main()
