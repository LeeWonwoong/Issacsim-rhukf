#!/usr/bin/env python3
"""agg_dwell — 확약(HOVER_DWELL) 축 판정. final4 무대 × 확약 {1,5,10} × 시드 42-44.
   핵심: 강제 호버가 만든 오탐(fp_forced)을 정책이 고른 오탐(fpr_chosen)과 분리해
   "격차가 래치 아티팩트냐"는 반박을 숫자로 막는다."""
import json, glob, numpy as np
from collections import defaultdict
Q = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/tonight/q'
H = defaultdict(dict)
for f in glob.glob(f'{Q}/final4d*_s4*.json') + glob.glob(f'{Q}/final4_*_s42.json'):
    for k, m in json.load(open(f)).items():
        cell = m['cell'].lstrip('x')
        if cell == 'final4': cell = 'final4d1'       # 확약 1 = 동일 설정
        H[cell][(m['learner'], m['seed'])] = m['hist']

def wm(h, key, a=0, b=200, atk=False):
    v = [float(r[key]) for r in h[a:min(b, len(h))]
         if r.get(key) is not None and not np.isnan(r.get(key, np.nan)) and (not atk or r.get('has_atk'))]
    return float(np.mean(v)) if v else np.nan

NM = {'final4d1': '확약 1 (없음)', 'final4d5': '확약 5', 'final4d10': '확약 10'}
for cell in ('final4d1', 'final4d5', 'final4d10'):
    if cell not in H: continue
    print(f'== {NM[cell]}')
    print('| 학습기 | 시드 | 보상 | 오탐(전체) | 오탐(정책선택) | 강제FP/판 | F1 | 지연 | 추락 |')
    for (l, sd) in sorted(H[cell]):
        h = H[cell][(l, sd)]
        print(f'| {l:10s} | {sd} | {wm(h,"reward"):7.2f} | {wm(h,"fpr"):.4f} | {wm(h,"fpr_chosen"):.4f} | '
              f'{wm(h,"fp_forced"):5.2f} | {wm(h,"f1",atk=True):.3f} | {wm(h,"delay"):5.2f} | '
              f'{sum(int(r.get("crashed",0)) for r in h):3d} |')
    # 짝 차이 (같은 시드·같은 시나리오)
    sws = {sd: h for (l, sd), h in H[cell].items() if l.startswith('SW')}
    for opp in sorted({l for (l, _) in H[cell] if not l.startswith('SW')}):
        ds = []
        for sd, a in sorted(sws.items()):
            b = H[cell].get((opp, sd))
            if b is None: continue
            n = min(len(a), len(b))
            ds.append(float(np.mean([a[i]['reward'] - b[i]['reward'] for i in range(n)])))
        if len(ds) < 2: continue
        m, sd_ = np.mean(ds), np.std(ds, ddof=1)
        t = m / (sd_ / np.sqrt(len(ds))) if sd_ > 0 else np.nan
        print(f'   SW − {opp:10s} {m:+7.3f}/판  t={t:6.2f}  승 {sum(1 for x in ds if x>0)}/{len(ds)}')
    print()
