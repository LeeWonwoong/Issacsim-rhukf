"""단일축 vs balanced 비교: 생존 + gyro/vel NIS (δ×ws). 방향이 밴드에 영향 주나."""
import sys, os, csv
import numpy as np
from collections import defaultdict
comp=lambda x: min(float(np.log1p(np.sqrt(max(x,0.0)))),3.0)
WINDS=[0,4,5,6,7,8,9,10,11,12]; DS=[0.3,0.4,0.5,0.6,0.7,0.8]
DIV={'single':4.36,'balanced':3.0829}   # bias→δ 환산 (축모드별)

def load(tag):
    D=f'results_axis_{tag}'
    if not os.path.exists(f'{D}/sweep_detail.csv'): return None
    div=DIV[tag]; eps=defaultdict(list)
    for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
        try:
            if r['policy']!='track': continue
            ws=int(round(float(r['wind_speed']))); d=round(float(r['bias'])/div,1)
            if d<0.1: continue
            eps[(ws,d,r['episode']+r['cell_idx'])].append((int(r['step']),float(r['nis_g_raw']),float(r['nis_v_raw'])))
        except: pass
    surv=defaultdict(lambda:[0,0]); atg=defaultdict(list); atv=defaultdict(list)
    for (ws,d,ek),rows in eps.items():
        rows.sort(); a=np.array(rows); mx=int(a[-1,0])
        surv[(ws,d)][0]+=(1 if mx>=360 else 0); surv[(ws,d)][1]+=1
        for st,g,v in a:
            if 300<=int(st)<380: atg[(ws,d)].append(comp(g)); atv[(ws,d)].append(comp(v))
    return surv,atg,atv

S=load('single'); B=load('balanced')
L=[]; P=lambda s:(L.append(str(s)),print(s))
P('='*90); P(' 단일축(roll) vs balanced(roll=pitch) — 같은 magnitude δ. 생존 & 공격 NIS'); P('='*90)
for tag,dat in [('단일축',S),('balanced',B)]:
    if dat is None: P(f'\n【{tag}】 아직 데이터 없음'); continue
    surv,atg,atv=dat
    P(f'\n【{tag}】 생존율 (δ×ws):')
    P('  δ＼ws | '+' '.join(f'{w:>5d}' for w in WINDS))
    for d in DS:
        P(f'  {d:.1f}   | '+' '.join((lambda s:f'{s[0]/s[1]:.2f}' if s[1] else '  -  ')(surv.get((w,d),[0,0])).rjust(5) for w in WINDS))
    P(f'  [공격 gyro/vel 압축 중앙값, ws9]')
    for d in DS:
        g=atg.get((9,d),[]); v=atv.get((9,d),[])
        P(f'   δ{d}: gyro {np.median(g):.2f} vel {np.median(v):.2f}' if g else f'   δ{d}: -')
# 직접 비교 (생존 차이)
if S and B:
    P('\n【생존 차이: balanced − 단일축】 (양수=balanced가 더 생존)')
    P('  δ＼ws | '+' '.join(f'{w:>5d}' for w in WINDS))
    for d in DS:
        row=[]
        for w in WINDS:
            ss=S[0].get((w,d),[0,0]); bs=B[0].get((w,d),[0,0])
            if ss[1] and bs[1]: row.append(f'{bs[0]/bs[1]-ss[0]/ss[1]:+.2f}')
            else: row.append('  -  ')
        P(f'  {d:.1f}   | '+' '.join(x.rjust(5) for x in row))
    P('\n→ 차이 작으면(±0.1 내) 방향 무관 = roll≈pitch 확인. 크면 축 의존.')
open('results_axis_compare_summary.txt','w').write("\n".join(L))
print('\n[저장] results_axis_compare_summary.txt')
