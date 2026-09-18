# -*- coding: utf-8 -*-
"""config E 재구동 — zu_v3 전 15에피 time-step 관측(NIS) 플롯, 패턴×바람×δ 정렬."""
import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.'); sys.path.insert(0,'etc/scripts')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
from before_after_plot import run, CFG_E, PATS
R=run('results_zu_v3',CFG_E)
f=R['fresh']
# 평시 기준선 (전 에피 공통): 순수 평시 gyro/vel 90p
BEN=f&(~R['atk'])&(R['so']>15)
g90=np.percentile(R['g'][BEN],90); v90=np.percentile(R['v'][BEN],90)
eps=np.unique(R['ep'])
meta=[]
for e in eps:
    m=R['ep']==e
    pat=int(np.median(R['pat'][m])); ws=float(np.median(R['ws'][m])); af=float(R['atk'][m].mean())
    dd=R['delta'][m&R['atk']]; d=float(np.median(dd)) if len(dd) else 0.0
    meta.append((e,pat,ws,af,d))
ben=[x for x in meta if x[3]==0]; atk=[x for x in meta if x[3]>0]
ben.sort(key=lambda x:x[2])           # 평시: 바람 오름차순
atk.sort(key=lambda x:x[4])           # 공격: δ 오름차순
order=ben+atk
ncol=3; nrow=(len(order)+ncol-1)//ncol
fig,axes=plt.subplots(nrow,ncol,figsize=(17,2.9*nrow),sharey=True)
axes=axes.ravel()
for k,(e,pat,ws,af,d) in enumerate(order):
    A=axes[k]; m=(R['ep']==e)&f; t=np.arange(m.sum())*0.1
    aa=R['atk'][m]
    on=np.where(np.r_[aa[0],np.diff(aa.astype(int))]==1)[0]; off=np.where(np.diff(aa.astype(int))==-1)[0]+1
    if len(off)<len(on): off=np.r_[off,len(aa)-1]
    for s_,e_ in zip(on,off): A.axvspan(t[s_],t[min(e_,len(t)-1)],color='#d62728',alpha=.15,lw=0)
    A.plot(t,R['g'][m],c='#d62728',lw=1.3,label='gyro NIS')
    A.plot(t,R['v'][m],c='#1f77b4',lw=1.1,label='vel NIS')
    A.plot(t,R['w'][m],c='#888',lw=.8,ls='--',alpha=.8,label='|ω| [rad/s]')
    A.axhline(g90,c='#d62728',ls=':',lw=.9,alpha=.8)
    A.axhline(v90,c='#1f77b4',ls=':',lw=.9,alpha=.6)
    is_atk = af>0
    wind_lbl = '무풍' if ws<1 else ('약풍' if ws<3 else ('중풍' if ws<6 else '강풍'))
    ttl=f"ep{int(e)} · {PATS[pat]} · {wind_lbl} {ws:.1f} m/s" + (f" · 공격 δ={d:.2f}" if is_atk else " · 평시")
    A.set_title(ttl,fontsize=10, color=('#8a1f14' if is_atk else '#1a5c37'), fontweight='bold')
    A.set_ylim(0,3.2); A.grid(alpha=.22)
    if k%ncol==0: A.set_ylabel('압축 NIS')
    if k>=len(order)-ncol: A.set_xlabel('시간 [s]')
    if k==0: A.legend(fontsize=8,ncol=3,loc='upper right')
for k in range(len(order),len(axes)): axes[k].axis('off')
fig.suptitle('config E (Qg2e-2·Rg0.02·Rv0.01·Qe2e-3) — 전 15에피 관측 시계열  |  녹색제목=평시(바람↑ 순), 적색제목=공격(δ↑ 순)  |  점선=평시 90p (gyro 빨강 %.2f · vel 파랑 %.2f)'%(g90,v90),
             fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig('configE_facet_timeseries.png',dpi=105)
print('saved configE_facet_timeseries.png')
