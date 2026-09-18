#!/usr/bin/env python3
"""v25(R4 Q×P0 격자: 이득·P·복구·F1) / v26(R2 초기 샘플효율: 에피소드별 greedy 프로브) 집계"""
import json, glob, numpy as np, collections
N = 'results/claudecodefortest/night'
print("== v25 R4: UKF/EKF-TD Q×P0 (frame, 200ep) — F1, fpr, 발산(F1<0.5), P복구횟수, |K| 후기, Pmax 후기")
d = {}
for f in glob.glob(f'{N}/v25_w*.json'): d.update(json.load(open(f)))
rows = collections.defaultdict(list)
for k, m in d.items():
    cell = k.split('_')[0]; h = m['hist']
    rows[cell].append(dict(F1=m['F1'], fpr=m['fpr'], div=int(m['F1'] < 0.5), prep=max(x.get('prep', 0) for x in h), kg=np.mean([x.get('kgain', 0) for x in h[150:]]), pm=np.mean([x.get('pmax', 0) for x in h[150:]])))
for key in sorted(rows):
    r = rows[key]; g = lambda k: np.nanmean([x[k] for x in r])
    print(f"  {key:16s} n{len(r)} F1 {g('F1'):.3f} fpr {g('fpr'):.3f} div {sum(x['div'] for x in r)}/{len(r)} prep {g('prep'):.0f} |K| {g('kg'):.3f} Pmax {g('pm'):.3f}")
print("\n== v26 R2: 초기 샘플효율 (greedy 프로브 F1, 에피소드 1..15 평균 곡선; t: probe F1>=0.85 첫 에피)")
d = {}
for f in glob.glob(f'{N}/v26*_w*.json'): d.update(json.load(open(f)))
rows = collections.defaultdict(list)
for k, m in d.items():
    cell, ag = k.split('_')[0], k.split('_')[1]; pf = [x.get('probe_f1', np.nan) for x in m['hist']]
    rows[(cell, ag)].append(pf)
for key in sorted(rows):
    a = np.array(rows[key]); mean = np.nanmean(a, 0); t = [next((i + 1 for i in range(len(p)) if p[i] >= 0.85), 99) for p in a]
    print(f"  {key[0]:6s}{key[1]:12s} n{len(a)} ep1-5: " + ' '.join(f'{v:.2f}' for v in mean[:5]) + f" | ep10 {mean[9]:.2f} ep15 {mean[14]:.2f} | t85 {np.mean(t):.1f}±{np.std(t):.1f}")
