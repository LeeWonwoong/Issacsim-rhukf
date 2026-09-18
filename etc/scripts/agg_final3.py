#!/usr/bin/env python3
"""agg_final3 — 확정안 1시드 판정. 돌풍 off(final3) vs on(final3g).
   공격 δ 0.30–0.74(50%)+0.74–0.84(50%) · 버스트 1 · 평시조건 20판 주기 전환 · 보상 상수 −2.0 · 확약 없음 · 1스텝 TD · 200판."""
import json, glob, numpy as np
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'; H = {}
for f in glob.glob(f'{N}/tonight/q/final3*_s42.json'):
    for k, m in json.load(open(f)).items(): H[(m['cell'], m['learner'])] = m['hist']
NM = {'SW': 'SWIRL 기준(P0.03·N7)', 'SWg_P0.05_R1_N6': 'SWIRL 튜닝(P0.05·N6)', 'UKFp03': 'UKF-TD', 'EKFp03': 'EKF-TD',
      'Adam3e-4': 'Adam 3e-4', 'Adam1e-3': 'Adam 1e-3', 'SGD1e-2': 'SGD 1e-2'}
LS = ['SW', 'SWg_P0.05_R1_N6', 'UKFp03', 'EKFp03', 'Adam3e-4', 'Adam1e-3', 'SGD1e-2']
REF = 'SWg_P0.05_R1_N6'   # 짝 차이 기준은 튜닝 SWIRL
def w(c, l, key, a, b, atk=False):
    h = H.get((c, l))
    if not h: return np.nan
    v = [float(h[i][key]) for i in range(a, min(b, len(h)))
         if h[i].get(key) is not None and not np.isnan(h[i].get(key, np.nan)) and (not atk or h[i].get('has_atk'))]
    return float(np.mean(v)) if v else np.nan
def crashes(c, l, a, b):
    h = H.get((c, l)); return sum(int(h[i].get('crashed', 0)) for i in range(a, min(b, len(h)))) if h else np.nan
def strong(c, l, key, a, b):   # 절벽 구간(δ≥0.74) 판만
    h = H.get((c, l))
    if not h: return np.nan
    v = [float(h[i][key]) for i in range(a, min(b, len(h)))
         if h[i].get('has_atk') and (h[i].get('dmax', 0) or 0) >= 0.74 and h[i].get(key) is not None and not np.isnan(h[i].get(key, np.nan))]
    return float(np.mean(v)) if v else np.nan
for cell, lab in (('xfinal3', '돌풍 OFF'), ('xfinal3g', '돌풍 ON')):
    if not any((cell, l) in H for l in LS): continue
    print(f'== {lab}')
    print('| 학습기 | 최종 160–199 | 전체 0–199 | 추락(200판) | 절벽판 보상 | 지연 | 오탐 | 공분산 |')
    for l in LS:
        if (cell, l) not in H: continue
        print(f'| {NM[l]:22s} | {w(cell,l,"reward",160,200):7.2f} | {w(cell,l,"reward",0,200):7.2f} | {crashes(cell,l,0,200):3d} | '
              f'{strong(cell,l,"reward",0,200):7.2f} | {w(cell,l,"delay",0,200):5.2f} | {w(cell,l,"fpr",0,200):.4f} | {w(cell,l,"pmax",120,200):8.3f} |')
    sw = H.get((cell, REF)) or H.get((cell, 'SW'))
    if sw:
        print(f'  {NM[REF] if (cell, REF) in H else NM["SW"]} − 비교군 (판별 짝, 같은 판·같은 공격):')
        a = np.array([x['reward'] for x in sw], float)
        for l in [x for x in LS if x != (REF if (cell, REF) in H else 'SW')]:
            h = H.get((cell, l))
            if not h: continue
            d = a - np.array([x['reward'] for x in h], float)
            print(f'    vs {NM[l]:10s}: 전체 {d.mean():+6.2f}/판 · 후반 120–199 {d[120:200].mean():+6.2f} · 200판 누적 {d.sum():+8.1f}')
    print()
