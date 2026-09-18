# -*- coding: utf-8 -*-
"""waypoint 공격 궤적 4패널 (top-down 2D): δ{0.6,0.8} × {track(무대응), whover2(호버대응→복귀)}.
   zu_log 위치(z0 gpsN, z1 gpsE) 사용. 공격창 = SWEEP_ATK 150-250 (15-25s)."""
import os,sys,numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
AUTH=4.36
D=np.load('results_traj_wp/zu_log.npz',allow_pickle=True); d=D['data']
# cols: ep,reset,attack,action, z0N,z1E,z2D, z3-5 vel, z6-8 gyr, u0-3, euler3, atk_scale, atk_delay, pat, ws
import csv
S=list(csv.DictReader(open('results_traj_wp/sweep_detail.csv')))
# 에피 → (bias, policy) 매핑: zu ep 순서 = 셀 순서 (각 1에피)
cells=[]
seen=set()
for r in S:
    k=(float(r['bias']),r['policy'])
    if k not in seen: seen.add(k); cells.append(k)
print('셀 순서:',cells)
eps=sorted(np.unique(d[:,0]))
# 기준(공칭) waypoint 박스: R=1.6 → kk=0.32
kk=1.6/5.0
wps=np.array([[0.,0.],[5.,0.],[5.,5.],[-5.,5.],[-5.,0.],[0.,0.]])*kk
order=[(2.616,'track'),(2.616,'whover2'),(3.488,'track'),(3.488,'whover2')]
fig,axes=plt.subplots(2,2,figsize=(13.5,12))
for k,(b,pol) in enumerate(order):
    A=axes[k//2,k%2]
    try: ei=eps[cells.index((b,pol))]
    except ValueError: A.text(.5,.5,'no data',transform=A.transAxes); continue
    m=d[:,0]==ei
    N=d[m,4]; E=d[m,5]; atk=d[m,2]==1; act=d[m,3]
    x,y=N,E   # NED N=x, E=y (top-down)
    # 국면 분할: 공격 전 / 공격 중 / 공격 후
    n=len(x); idx=np.arange(n)
    a_on=np.where(atk)[0]
    s0,s1=(a_on[0],a_on[-1]) if len(a_on) else (n,n)
    A.plot(wps[:,0],wps[:,1],'k--',lw=1.2,alpha=.55,label='명령 궤적(waypoint 박스)')
    A.plot(x[:s0],y[:s0],c='#2ca02c',lw=1.6,label='공격 전')
    A.plot(x[s0:s1+1],y[s0:s1+1],c='#d62728',lw=2.0,label='공격 중 (10s)')
    A.plot(x[s1+1:],y[s1+1:],c='#1f77b4',lw=1.6,label='공격 후 (복귀)')
    hov=(act==1)
    if hov.sum()>3: A.scatter(x[hov],y[hov],s=14,c='#9467bd',alpha=.6,label='호버 구간',zorder=5)
    A.scatter(x[s0] if s0<n else x[-1],y[s0] if s0<n else y[-1],marker='X',s=130,c='#d62728',edgecolor='k',zorder=6,label='공격 ON')
    if s1<n-1: A.scatter(x[s1],y[s1],marker='o',s=110,c='#1f77b4',edgecolor='k',zorder=6,label='공격 OFF')
    dev=np.max(np.hypot(x[s0:s1+1]-np.clip(x[s0:s1+1],wps[:,0].min(),wps[:,0].max()),0)) if s1>s0 else 0
    # 최대 이탈: 명령 박스 경로와의 거리 근사(각 세그먼트 최소거리)
    def dist_to_path(px,py):
        dm=np.inf
        for i in range(len(wps)-1):
            a_,b_=wps[i],wps[i+1]; ab=b_-a_; t=np.clip(((np.array([px,py])-a_)@ab)/max(ab@ab,1e-9),0,1)
            dm=min(dm,np.hypot(*(np.array([px,py])-a_-t*ab)))
        return dm
    dmax=max(dist_to_path(px,py) for px,py in zip(x[s0:s1+1],y[s0:s1+1])) if s1>s0 else 0
    pol_lbl='track (무대응)' if pol=='track' else 'whover2 (호버 대응 후 복귀)'
    A.set_title(f'δ={b/AUTH:.1f} · {pol_lbl}   |   공격 중 최대 경로이탈 {dmax:.1f} m',fontsize=11.5)
    A.set_xlabel('N [m]'); A.set_ylabel('E [m]'); A.axis('equal'); A.grid(alpha=.3)
    if k==0: A.legend(fontsize=8.5,loc='best')
fig.suptitle('waypoint 공격 궤적 (위에서 본 2D) — 공격창 15~25s(10초), 무풍, config E\n'
             '좌열=무대응(track): 이탈 후 자력복귀 여부 · 우열=호버대응: 공격 감지 시 호버로 고정 후 종료 시 재기동',fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig('traj_wp_topdown.png',dpi=115)
print('saved traj_wp_topdown.png')
