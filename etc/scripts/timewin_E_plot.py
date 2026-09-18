# -*- coding: utf-8 -*-
"""config E 시간창 시나리오 플롯 — clean→강풍→(겹침)공격→바람off→공격off→복귀.
   행 = 패턴(track 셀), 열 = δ{0.4, 0.7}. 마지막 행 = hover 셀."""
import os,csv,numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
AUTH=4.36
R=list(csv.DictReader(open('results_timewin_E/sweep_detail.csv')))
for r in R:
    for k in ['bias','step','nis_v_scaled','nis_g_scaled','omega_norm','wind_speed']: r[k]=float(r[k])
    r['attack_active']=int(r['attack_active'])
pats=['circle','figure8','waypoint','aggressive']
bs=sorted(set(r['bias'] for r in R))
WIND=(100,280); ATK=(180,380)
rows=[(p,'track') for p in pats]+[('hover','hover')]
fig,axes=plt.subplots(len(rows),len(bs),figsize=(16,2.6*len(rows)),sharex=True,sharey=True)
for i,(pat,pol) in enumerate(rows):
    for j,b in enumerate(bs):
        A=axes[i,j]
        sel=[r for r in R if r['pattern']==pat and r['policy']==pol and r['bias']==b]
        sel.sort(key=lambda r:r['step'])
        if not sel:
            A.text(.5,.5,'(no data)',ha='center',transform=A.transAxes); continue
        t=np.array([r['step'] for r in sel])*0.1
        g=np.array([r['nis_g_scaled'] for r in sel]); v=np.array([r['nis_v_scaled'] for r in sel])
        w=np.array([r['omega_norm'] for r in sel])
        A.axvspan(WIND[0]*0.1,WIND[1]*0.1,color='#1f77b4',alpha=.10)
        A.axvspan(ATK[0]*0.1,ATK[1]*0.1,color='#d62728',alpha=.12)
        A.plot(t,g,c='#d62728',lw=1.3,label='gyro NIS')
        A.plot(t,v,c='#1f77b4',lw=1.1,label='vel NIS')
        A.plot(t,w,c='#888',lw=.8,ls='--',alpha=.8,label='|ω|')
        crash=[r for r in sel if r['crash_reason'] not in ('','timeout')]
        if sel[-1]['step']<440: A.axvline(t[-1],c='k',lw=1.2,ls='-'); A.text(t[-1],2.9,'✕추락',fontsize=8,ha='right')
        if i==0 and j==0: A.legend(fontsize=7.5,ncol=3,loc='upper left')
        if j==0: A.set_ylabel(f'{pat}\n({pol})',fontsize=10)
        if i==0: A.set_title(f'δ={b/AUTH:.2f}  (τ={b:.2f} N·m)',fontsize=11)
        A.set_ylim(0,3.2); A.grid(alpha=.2)
for j in range(len(bs)): axes[-1,j].set_xlabel('시간 [s]')
fig.suptitle('config E — 시간창 시나리오: clean(0-10s) → 강풍 ws8 ON(10s) → 공격 ON(18s, 겹침) → 바람 OFF(28s) → 공격 OFF(38s) → 복귀\n'
             '파란 음영=바람 창 · 붉은 음영=공격 창 · 겹침=두 음영 중첩(18-28s)',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.955])
fig.savefig('timewin_E_timeseries.png',dpi=105)
print('saved timewin_E_timeseries.png')
