#!/usr/bin/env python3
"""agg_tuneq — P₀ 빠른 스크린 판정. 기준선 = final1 의 SW(P0.03·R1·N7). 같은 무대·같은 시드(42)·같은 공격 시퀀스라 판별 짝 비교가 가능하다."""
import json, glob, numpy as np
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'; H = {}
for f in glob.glob(f'{N}/tonight/q/final1*_s42.json') + glob.glob(f'{N}/tonight/q/tuneq_*_s42.json'):
    for k, m in json.load(open(f)).items(): H[(m['cell'], m['learner'])] = m['hist']
NS = 'xfinal1'
def w(l, key, a, b):
    h = H.get((NS, l))
    if not h: return np.nan
    v = [float(h[i][key]) for i in range(a, min(b, len(h))) if h[i].get(key) is not None and not np.isnan(h[i].get(key, np.nan))]
    return float(np.mean(v)) if v else np.nan
def rr(l):
    h = H.get((NS, l)); return np.array([x['reward'] for x in h], float) if h else None
CFG = [('SW', '기준 P0.03·R1·N7')] + [(f'SWg_P{p}_R1_N6', f'P{p}·R1·N6') for p in ('0.01', '0.03', '0.05', '0.1', '0.2')]
base_u, base_a = rr('UKFp03'), rr('Adam3e-4')
print('| 설정 | 최종 160–199 | 강풍 120–199 | 중간 60–119 | 잔잔 0–59 | vs UKF-TD(강풍 판당) | vs Adam3e-4(강풍) | 공분산 | 오탐 |')
print('|---|---|---|---|---|---|---|---|---|')
for l, lab in CFG:
    r = rr(l)
    if r is None: print(f'| {lab} | (미완) |'); continue
    du = (r - base_u)[120:200].mean() if base_u is not None else np.nan
    da = (r - base_a)[120:200].mean() if base_a is not None else np.nan
    print(f'| {lab} | {w(l,"reward",160,200):7.2f} | {w(l,"reward",120,200):7.2f} | {w(l,"reward",60,120):7.2f} | {w(l,"reward",0,60):7.2f} | {du:+6.2f} | {da:+6.2f} | {w(l,"pmax",120,200):.3f} | {w(l,"fpr",120,200):.4f} |')
print()
print('참고 비교군: UKF-TD %.2f · EKF-TD %.2f · Adam3e-4 %.2f · Adam1e-3 %.2f (최종 160–199)'
      % tuple(w(x, 'reward', 160, 200) for x in ('UKFp03', 'EKFp03', 'Adam3e-4', 'Adam1e-3')))
