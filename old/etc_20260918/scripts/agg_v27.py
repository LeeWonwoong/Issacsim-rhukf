#!/usr/bin/env python3
"""v27 집계: (리플레이 모드 × 옵티마이저) F1·fpr·돌풍중 FP율·돌풍직후 recall·greedy 프로브 곡선·돌풍 에피소드 다음 프로브 F1 낙폭"""
import json, glob, numpy as np, collections, sys
N = 'results/claudecodefortest/night'; d = {}
for f in glob.glob(f'{N}/v27*_w*.json'): d.update(json.load(open(f)))
rows = collections.defaultdict(list)
for k, m in d.items():
    cell, ag = k.split('_')[0], k.split('_')[1]; h = m['hist']
    pf = np.array([x.get('probe_f1', np.nan) for x in h]); pfp = np.array([x.get('probe_fpr', np.nan) for x in h]); hg = np.array([x.get('had_gust', 0) for x in h], bool)
    fg = np.array([x.get('fpr_gust', np.nan) for x in h]); rag = np.array([x.get('rec_after_gust', np.nan) for x in h])
    # 돌풍 에피소드 직후 프로브 F1 vs 무돌풍 에피소드 직후 (같은 런 안에서 짝)
    nxt_g = [pf[i + 1] for i in range(len(h) - 1) if hg[i]]; nxt_n = [pf[i + 1] for i in range(len(h) - 1) if not hg[i]]
    rows[(cell, ag)].append(dict(F1=m['F1'], fpr=m['fpr'], dly=m['delay'], crash=m['crash'], fg=np.nanmean(fg) if np.isfinite(fg).any() else np.nan,
        rag=np.nanmean(rag) if np.isfinite(rag).any() else np.nan, p_early=np.nanmean(pf[:10]), p_mid=np.nanmean(pf[30:60]), p_late=np.nanmean(pf[100:150]),
        pfp_late=np.nanmean(pfp[100:150]), drop=(np.nanmean(nxt_n) - np.nanmean(nxt_g)) if nxt_g and nxt_n else np.nan,
        t80=next((i for i in range(len(pf)) if np.nanmean(pf[max(0, i - 4):i + 1]) >= 0.8), -1)))
hdr = f"{'cell':5s}{'agent':6s} n  F1          fpr    dly   crash fprGust recAfterG  probeF1 ep<10/30-60/100-150  probeFPR  drop(nextEp)  t80"
print(hdr)
for key in sorted(rows):
    r = rows[key]; g = lambda k: np.nanmean([x[k] for x in r]); sd = lambda k: np.nanstd([x[k] for x in r])
    print(f"{key[0]:5s}{key[1]:6s} {len(r)}  {g('F1'):.3f}±{sd('F1'):.3f} {g('fpr'):.3f} {g('dly'):5.2f} {g('crash'):4.1f}  {g('fg'):.3f}   {g('rag'):.3f}     {g('p_early'):.3f}/{g('p_mid'):.3f}/{g('p_late'):.3f}   {g('pfp_late'):.3f}    {g('drop'):+.3f}      {g('t80'):5.1f}")
