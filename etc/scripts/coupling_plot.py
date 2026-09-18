# -*- coding: utf-8 -*-
"""결과성 ↔ 애매성 커플링 — 두 곡선이 δ 축에서 겹치지 않는다는 것을 한 장으로."""
import os,sys,csv,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
from before_after_plot import run, CFG_E
AUTH=4.36
R=list(csv.DictReader(open('results_axis_single/sweep_detail.csv')))
S=list(csv.DictReader(open('results_axis_single/sweep_summary.csv')))
base=np.median([float(r['gt_err']) for r in R if r['policy']=='track' and r['attack_active']=='0'])
bs=sorted(set(float(r['bias']) for r in R))
dev=[];srv=[];dd=[]
for b in bs:
    A=[r for r in R if float(r['bias'])==b and r['policy']=='track' and r['attack_active']=='1']
    T=[r for r in S if float(r['bias'])==b and r['policy']=='track']
    if len(A)<50: continue
    dd.append(b/AUTH); dev.append(np.median([float(r['gt_err']) for r in A])/base)
    srv.append(1-np.mean([float(r['survived']) for r in T]))
A=run('results_zu_v3',CFG_E)
f=A['fresh']; BEN=f&(~A['atk'])&(A['so']>15)&(A['act']==0); ST=f&A['atk']&(A['delay']>=3)
p90=np.percentile(A['g'][BEN],90)
dx=[];amb=[]
for lo,hi in [(0.10,0.25),(0.25,0.40),(0.40,0.55),(0.55,0.80)]:
    m=ST&(A['delta']>=lo)&(A['delta']<hi)
    if m.sum()<10: continue
    dx.append((lo+hi)/2); amb.append(float((A['g'][m]<p90).mean())*100)
fig,ax=plt.subplots(1,2,figsize=(15,5.4))
Ax=ax[0]
Ax.plot(dd,dev,'o-',c='#d62728',lw=2.4,ms=7,label='궤적이탈 배수 (무대응 track)')
Ax.axhline(1.0,c='#d62728',ls=':',lw=1.2)
Ax2=Ax.twinx(); Ax2.plot(dd,[s*100 for s in srv],'s--',c='#8c564b',lw=1.8,ms=6,label='추락률 [%]')
Ax2.set_ylabel('추락률 [%]',color='#8c564b')
Ax.plot(dx,[a/100*max(dev) for a in amb],'^-',c='#1f77b4',lw=2.4,ms=8,label='공격이 평시에 묻히는 비율 (AFTER)')
for x,a in zip(dx,amb): Ax.annotate(f'{a:.0f}%',(x,a/100*max(dev)),color='#1f77b4',fontsize=9,xytext=(0,7),textcoords='offset points',ha='center')
Ax.axvspan(0.10,0.30,color='#1f77b4',alpha=.10); Ax.text(0.20,max(dev)*.93,'애매하지만\n무해',ha='center',fontsize=10,color='#1f77b4')
Ax.axvspan(0.55,0.80,color='#d62728',alpha=.10); Ax.text(0.68,max(dev)*.93,'유해하지만\n자명',ha='center',fontsize=10,color='#d62728')
Ax.set_xlabel('공격 세기 δ'); Ax.set_ylabel('궤적이탈 배수 (1.0=평시와 동일)')
Ax.set_title('커플링 문제 — 두 곡선이 δ 축에서 만나지 않는다',fontsize=12)
Ax.grid(alpha=.25); Ax.legend(fontsize=9,loc='upper left'); Ax2.legend(fontsize=9,loc='center left')
Ax=ax[1]
ons=[(5,20),(4,12),(3,8),(3,5)]
frac=[np.mean([min(3,L)/L for L in range(a,b+1)])*100 for a,b in ons]
Ax.bar([f'U({a},{b})' for a,b in ons],frac,color=['#bbb','#7fb3d5','#2e86c1','#1b4f72'])
for i,v in enumerate(frac): Ax.text(i,v+1.5,f'{v:.0f}%',ha='center',fontsize=11,fontweight='bold')
Ax.axhline(frac[0],c='k',ls=':',lw=1.2)
Ax.set_ylabel('공격 스텝 중 온셋 과도구간 비율 [%]')
Ax.set_xlabel('버스트 ON 길이 분포 (현행 = U(5,20))')
Ax.set_title('시간 레버 — 버스트를 짧게 하면\n**결과성 있는 δ에서도** 애매구간이 생긴다',fontsize=12)
Ax.grid(alpha=.25,axis='y'); Ax.set_ylim(0,95)
fig.suptitle('진폭으로는 못 만들고, 시간으로는 만들 수 있다',fontsize=13.5)
fig.tight_layout(); fig.savefig('coupling_and_lever.png',dpi=110)
print('saved coupling_and_lever.png')
