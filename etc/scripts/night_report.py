#!/usr/bin/env python3
"""night_report — night 스윕 JSON 집계 → results/claudecodefortest/night/night_report.txt"""
import json, glob, os, statistics as st
os.chdir('/home/acsl/projects/Issacsim-rhukf'); N = 'results/claudecodefortest/night'
L = []
def P(s=''): L.append(s); print(s)
dec = json.load(open(f'{N}/decision.json'))
P(f"결정: arm {dec['arm']} · 필터 {dec['filter']} · 약공격 δ[{dec['weak_lo']},{dec['weak_hi']}] ON{dec['weak_on']} · 치명 δ{dec['leth']} hold{dec['leth_hold']} · P(치명) {dec['p_lethal']}")
for tag in ('S1', 'S2'):
    res = {}
    for f in glob.glob(f'{N}/sweep_w*_{tag}.json'): res.update(json.load(open(f)))
    if not res: P(f"\n### {tag}: (없음)"); continue
    P(f"\n### {tag}  ({len(res)} runs)")
    P(f"  {'run':28s} {'F1':>6s} {'fpr':>6s} {'delay':>6s} {'rwd':>6s} {'e40':>6s} {'crash':>5s} {'relapse':>7s}")
    for k, v in sorted(res.items(), key=lambda kv: (-kv[1]['F1'])):
        P(f"  {k:28s} {v['F1']:6.3f} {v['fpr']:6.3f} {v['delay']:6.2f} {v['rwd']:6.0f} {v['e40']:6.0f} {v['crash']:5d} {v.get('relapse',0):7.2f}")
# 옵티마이저 비교 (S1+S3: best SWIRL vs Adam vs SGD, seeds 42~44)
allr = {}
for f in glob.glob(f'{N}/sweep_w*_S*.json'): allr.update(json.load(open(f)))
sw = sorted([(v['F1'], k) for k, v in allr.items() if k.startswith('SWIRL') and k.endswith('_s42') and '_pl' not in k], reverse=True)
if sw:
    best = sw[0][1].replace('_s42', '')
    P(f"\n### 옵티마이저 비교 (best SWIRL = {best}, seeds 42~44)")
    for nm, pref in (('SWIRL', best), ('Adam', 'Adam'), ('SGD', 'SGD')):
        rows = [allr[k] for k in allr if k.startswith(pref + '_s') and '_pl' not in k]
        if not rows: continue
        def ms(key): xs = [r[key] for r in rows]; return f"{st.mean(xs):.3f}±{st.stdev(xs) if len(xs)>1 else 0:.3f}"
        P(f"  {nm:6s} n{len(rows)}  F1 {ms('F1')}  fpr {ms('fpr')}  delay {ms('delay')}  rwd {ms('rwd')}  e40 {ms('e40')}  crash {st.mean([r['crash'] for r in rows]):.1f}")
P("\n### S2 치명 비율 P(lethal) 별 (seed 42): F1 / fpr / delay / rwd / crash")
res2 = {}
for f in glob.glob(f'{N}/sweep_w*_S2.json'): res2.update(json.load(open(f)))
names = sorted(set(k.split('_pl')[0] for k in res2))
for pl in ('0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1'):
    row = []
    for nm in names:
        v = res2.get(f'{nm}_pl{pl}_s42')
        row.append(f"{nm[:22]:22s} {v['F1']:.3f}/{v['fpr']:.3f}/{v['delay']:.1f}/{v['rwd']:.0f}/{v['crash']}" if v else f"{nm[:22]:22s} -")
    P(f"  pl {pl}: " + " | ".join(row))
open(f'{N}/night_report.txt', 'w').write("\n".join(L)); print(f'\n저장: {N}/night_report.txt')
