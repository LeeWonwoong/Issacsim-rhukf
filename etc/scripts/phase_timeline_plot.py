#!/usr/bin/env python3
# phase_timeline_plot — 5단계(정상→바람→바람+공격→공격→정상) NIS/관측 타임라인.
#   3 기동(waypoint/aggressive/hover) × 2 채널(gyro/vel) 그리드, δ0.2 vs δ0.7 대조.
import csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import defaultdict

D={'0.872':'d0.2','3.052':'d0.7'}
eps=defaultdict(list)
with open('results_phase_timeline/sweep_detail.csv') as f:
    for r in csv.DictReader(f):
        pat=r['pattern'] if r['policy']=='track' else 'hover'
        eps[(pat,D.get(r['bias'],''),r['episode'])].append(r)

def med_series(pat,dl,col):
    """에피 3개의 step별 median 시계열."""
    series={}
    for (p,d,e),rows in eps.items():
        if p!=pat or d!=dl: continue
        for r in rows:
            s=int(r['step']); v=float(r[col])
            series.setdefault(s,[]).append(v)
    ss=sorted(series)
    return ss,[np.median(series[s]) for s in ss],[np.percentile(series[s],90) for s in ss]

PHASES=[(0,60,'clean','#f0f0f0'),(60,120,'wind','#dbeafe'),(120,180,'wind+atk','#fde2e2'),
        (180,240,'atk only','#fff3d6'),(240,300,'clean','#f0f0f0')]
fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
for col,pat in enumerate(['waypoint','aggressive','hover']):
    for row,(ch,chn) in enumerate([('nis_g_scaled','gyro obs'),('nis_v_scaled','vel obs')]):
        ax=axes[row][col]
        for a,b,nm,c in PHASES:
            ax.axvspan(a,b,color=c,alpha=0.8,zorder=0)
        for dl,color,lw in [('d0.2','#2b5fb3',1.3),('d0.7','#d1495b',1.3)]:
            ss,med,p90=med_series(pat,dl,ch)
            if not ss: continue
            ax.plot(ss,med,color=color,lw=lw,label=f'{dl} med')
            ax.plot(ss,p90,color=color,lw=0.7,ls=':',alpha=0.7)
        ax.set_ylim(-0.05,3.1); ax.grid(alpha=0.2,zorder=1)
        if row==0:
            ax.set_title(pat,fontsize=12,fontweight='bold')
            for a,b,nm,c in PHASES: ax.text((a+b)/2,2.95,nm,ha='center',fontsize=7,color='#555')
        if col==0: ax.set_ylabel(chn,fontsize=11,fontweight='bold')
        if row==1: ax.set_xlabel('step (10Hz)')
        ax.legend(fontsize=8,loc='center left')
plt.suptitle('arm0.07 · ws9 : clean -> wind -> wind+attack -> attack -> clean  (obs = min(log1p(sqrt NIS),3), median of 3 eps, dotted=p90)',fontsize=11)
plt.tight_layout(); plt.savefig('/tmp/phase_timeline.png',dpi=115,bbox_inches='tight')
print('saved /tmp/phase_timeline.png')

# 수치 요약: 페이즈별 gyro/vel obs median
print(f'\n{"pattern":11}{"δ":>5} | {"ch":4} | ' + ' | '.join(f'{nm:>8}' for _,_,nm,_ in PHASES))
for pat in ['waypoint','aggressive','hover']:
    for dl in ['d0.2','d0.7']:
        for ch,chn in [('nis_g_scaled','g'),('nis_v_scaled','v')]:
            ss,med,_=med_series(pat,dl,ch)
            if not ss: continue
            arr=dict(zip(ss,med)); out=[]
            for a,b,nm,_ in PHASES:
                vals=[arr[s] for s in ss if a<=s<b]
                out.append(f'{np.median(vals):>8.2f}' if vals else f'{"—":>8}')
            print(f'{pat:11}{dl:>5} | {chn:4} | ' + ' | '.join(out))
