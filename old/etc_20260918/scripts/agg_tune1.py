#!/usr/bin/env python3
"""agg_tune1 — 1번(온라인 RL) SWIRL 튜닝 집계. per5n1c4 무대, 시드 42."""
import json, glob, numpy as np
Q = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/tonight/q'
rows = []
for f in sorted(glob.glob(f'{Q}/tune1_*_s42.json')):
    for k, m in json.load(open(f)).items():
        h = m['hist']
        def w(key, a=0, b=200, atk=False):
            v = [float(r[key]) for r in h[a:min(b,len(h))]
                 if r.get(key) is not None and not np.isnan(r.get(key, np.nan)) and (not atk or r.get('has_atk'))]
            return float(np.mean(v)) if v else float('nan')
        lv = m['lenv']
        rows.append((w('reward'), lv.get('RHUKF_PINIT'), lv.get('RHUKF_R'), lv.get('RHUKF_N'),
                     lv.get('RHUKF_ALPHA'), w('fpr'), w('f1', atk=True), w('delay')))
rows.sort(reverse=True)
print()
print('### 1번 온라인 RL — SWIRL 튜닝 (per5n1c4, 시드 42, 150ep)')
print()
print('| 순위 | P₀ | R | N | α | 보상 | 오탐 | F1 | 지연 |')
print('|---|---|---|---|---|---|---|---|---|')
for i, (r, p, R, n, a, fp, f1, d) in enumerate(rows, 1):
    print(f'| {i} | {p} | {R} | {n} | {a} | **{r:.2f}** | {fp:.4f} | {f1:.3f} | {d:.2f} |')
if rows:
    b = rows[0]
    print(f'\n**최적: P₀={b[1]} R={b[2]} N={b[3]} α={b[4]} → 보상 {b[0]:.2f}**')
