#!/usr/bin/env python3
"""agg_stop — 정지 문제 본선 집계(다중 시드). 짝 차이 + 구간별."""
import json, glob, sys, numpy as np
from collections import defaultdict
pat, title = sys.argv[1], sys.argv[2]
H = defaultdict(dict)
for f in sorted(glob.glob(pat)):
    for nm, m in json.load(open(f)).items():
        _sd = (m.get('cfg') or {}).get('seed', f)
        H[nm][_sd] = m['hist']
if not H:
    print(f'\n### {title} — 결과 없음\n'); sys.exit()
print(f'\n### {title} — 본선 집계 (시드 {len(next(iter(H.values())))}개)\n')
print('| 학습기 | 후반100 보상 | 오경보 | 미탐 | 지연 | 전이 |')
print('|---|---|---|---|---|---|')
def avg(hs, fn): 
    v = [fn(h) for h in hs]; v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float('nan')
for nm in sorted(H):
    hs = list(H[nm].values())
    r  = avg(hs, lambda h: np.mean([x['reward'] for x in h[-100:]]))
    fa = avg(hs, lambda h: np.mean([x['fa'] for x in h[-100:]]))
    ms = avg(hs, lambda h: np.mean([x['miss'] for x in h[-100:]]))
    dl = avg(hs, lambda h: np.mean([x['delay'] for x in h[-100:] if x['delay'] is not None] or [np.nan]))
    tr = avg(hs, lambda h: h[-1]['n_trans'])
    print(f'| {nm} | **{r:.2f}** | {fa:.3f} | {ms:.3f} | {dl:.2f} | {tr:,.0f} |')
sw = H.get('SWIRL')
if sw:
    print('\n**SWIRL − 비교군 (짝, 같은 시드):**\n')
    for opp in sorted(k for k in H if k != 'SWIRL'):
        ds = []
        for s in sw:
            if s not in H[opp]: continue
            a = np.array([x['reward'] for x in sw[s]], float)
            b = np.array([x['reward'] for x in H[opp][s]], float)
            n = min(len(a), len(b)); ds.append(float(np.mean(a[:n]-b[:n])))
        if len(ds) < 2: continue
        m_, sd_ = np.mean(ds), np.std(ds, ddof=1)
        t = m_/(sd_/np.sqrt(len(ds))) if sd_ > 0 else float('nan')
        print(f'- vs {opp}: **{m_:+.3f}**/판 · t={t:+.2f} · 승 {sum(1 for x in ds if x>0)}/{len(ds)}')
