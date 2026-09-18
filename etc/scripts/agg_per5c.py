#!/usr/bin/env python3
"""agg_per5c — 가장 유망한 조건(per5 바람 불규칙 전환 + 1스텝 TD + 확약 5 + δ0.10-0.84)에
   오늘 찾은 관측 클립 제거를 적용한 팔의 판정.
   대조군 = xper5_n1 (같은 조건, 클립 3.0 유지, 시드 42-44).
   강제 호버(확약 5)가 지표를 오염시키므로 fpr_chosen / fp_forced 를 함께 본다."""
import json, glob, numpy as np
from collections import defaultdict
Q = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/tonight/q'
H = defaultdict(dict)
for f in glob.glob(f'{Q}/*.json'):
    try: d = json.load(open(f))
    except Exception: continue
    for k, m in d.items():
        if m.get('cell') in ('xper5n1c', 'xper5n1c4', 'xper5_n1') and m.get('hist'):
            H[m['cell']][(m['learner'], m['seed'])] = m['hist']

def wm(h, key, atk=False):
    v = [float(r[key]) for r in h
         if r.get(key) is not None and not np.isnan(r.get(key, np.nan)) and (not atk or r.get('has_atk'))]
    return float(np.mean(v)) if v else np.nan

NM = {'xper5_n1': '대조 (클립 3.0, 관측 [0,3])', 'xper5n1c': '클립 제거 (관측 [0,4])',
      'xper5n1c4': '★ 클립 제거 + /4 (관측 [0,1])'}
for cell in ('xper5_n1', 'xper5n1c', 'xper5n1c4'):
    D = H.get(cell)
    if not D: continue
    seeds = sorted({s for _, s in D})
    print(f'== {NM[cell]}  시드 {seeds}')
    print('| 학습기 | 보상 | 오탐(전체) | 오탐(정책) | 강제FP/판 | F1 | 지연 | 추락 |')
    for l in sorted({l for l, _ in D}):
        hs = [D[(l, s)] for s in seeds if (l, s) in D]
        if not hs: continue
        f = lambda k, a=False: np.nanmean([wm(h, k, a) for h in hs])
        cr = np.mean([sum(int(r.get('crashed', 0)) for r in h) for h in hs])
        print(f'| {l:14s} | {f("reward"):7.2f} | {f("fpr"):.4f} | {f("fpr_chosen"):.4f} | '
              f'{f("fp_forced"):5.2f} | {f("f1", True):.3f} | {f("delay"):5.2f} | {cr:4.1f} |')
    sw = 'SW'
    if not any(l == sw for l, _ in D): print(); continue
    print(f'   SWIRL − 비교군 (짝, 같은 시드·같은 시나리오):')
    for opp in sorted({l for l, _ in D if l.startswith('Adam')}):
        ds = []
        for s in seeds:
            a, b = D.get((sw, s)), D.get((opp, s))
            if a is None or b is None: continue
            n = min(len(a), len(b))
            ds.append(float(np.mean([a[i]['reward'] - b[i]['reward'] for i in range(n)])))
        if len(ds) < 2: continue
        m, sd_ = np.mean(ds), np.std(ds, ddof=1)
        t = m / (sd_ / np.sqrt(len(ds))) if sd_ > 0 else np.nan
        print(f'     vs {opp:14s} {m:+7.3f}/판  t={t:6.2f}  승 {sum(1 for x in ds if x>0)}/{len(ds)}')
    print()
