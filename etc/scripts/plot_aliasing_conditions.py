"""기동 aliasing before/after + 조건별(궤적·공격·바람) NIS 분석·plot.
zu_before(가감속off) vs zu_after(가감속0.5) 재구동.
Fig1: NIS-vs-|ω| aliasing곡선 (before/after × benign/attack)
Fig2: 패턴별 시계열 (after)
표:  궤적·바람·공격별 gyro/vel NIS (before→after)
사용: python plot_aliasing_conditions.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
import sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
PATS=['circle','figure8','waypoint','aggressive']

def replay(path):
    if not os.path.exists(f'{path}/zu_log.npz'): return None
    D=np.load(f'{path}/zu_log.npz',allow_pickle=True); data=D['data']; dt=float(D['dt'])
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
    for i in (9,10,11): ukf.Q[i,i]=2e-3
    out=[]; sie=0
    for r in data:
        if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
        z=r[4:13].astype(float); u=r[13:17].astype(float)
        w=float(np.linalg.norm(z[6:9])); fresh=(sie%5==0)
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
        _,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)
        pat=int(r[22]) if len(r)>22 else -1; ws=float(r[23]) if len(r)>23 else 0.0
        out.append((int(r[0]),int(r[2]),w,gs,vs if fresh else np.nan,pat,ws)); sie+=1
    return np.array(out)  # ep,attack,|w|,gyro,vel,pat,ws

B=replay('results_zu_before'); A=replay('results_zu_after')
L=[]; P=lambda s='':(L.append(str(s)),print(s))

# ── 표: |ω|별 평시 gyro NIS (before vs after) ──
P('='*76); P(' 기동강도 |ω|별 평시 gyro NIS — before(가감속off) → after(on)'); P('='*76)
P(' |ω| 구간        | before | after | 공격(after)')
for nm,lo,hi in [('gentle 0-0.5',0,0.5),('중간 0.5-1.5',0.5,1.5),('급기동 1.5+',1.5,99)]:
    def med(X,atk):
        if X is None: return np.nan
        m=(X[:,1]==atk)&(X[:,2]>=lo)&(X[:,2]<hi); return np.median(X[m,3]) if m.sum() else np.nan
    P(f'  {nm:14s}| {med(B,0):5.2f}  | {med(A,0):5.2f} | {med(A,1):5.2f}')
P(' → after 급기동서 평시 gyro가 공격 수준까지↑ = aliasing (before는 낮게 유지)')

# ── 표: 패턴별 급기동 aliasing (after, 평시 |ω|>1.0 gyro) ──
P('\n'+'='*76); P(' 궤적별 aliasing (after 평시, |ω|>1.0 순간 gyro NIS 90pct)'); P('='*76)
P(' 궤적       | |ω| 90p | 평시gyro 90p(|ω|>1.0) | 공격gyro중앙')
for pi,pat in enumerate(PATS):
    if A is None: break
    m=A[:,5]==pi
    if not m.sum(): P(f'  {pat:10s}| (없음)'); continue
    ben=m&(A[:,1]==0); benhi=ben&(A[:,2]>1.0); atk=m&(A[:,1]==1)
    w90=np.percentile(A[ben,2],90) if ben.sum() else 0
    g90=np.percentile(A[benhi,3],90) if benhi.sum() else np.nan
    ag=np.median(A[atk,3]) if atk.sum() else np.nan
    P(f'  {pat:10s}| {w90:5.2f}  | {g90:5.2f}              | {ag:.2f}')

# ── 표: 바람별 (after 평시 gyro/vel) ──
P('\n'+'='*76); P(' 바람별 평시 NIS (after, 공격X)'); P('='*76)
P(' 바람        | gyro중앙 | vel중앙')
for nm,lo,hi in [('약풍<3',0,3),('중풍3-6',3,6),('강풍6+',6,99)]:
    if A is None: break
    m=(A[:,1]==0)&(A[:,6]>=lo)&(A[:,6]<hi)
    if m.sum(): P(f'  {nm:10s}| {np.median(A[m,3]):.2f}     | {np.nanmedian(A[m,4]):.2f}')
open('results_aliasing_conditions.txt','w').write("\n".join(L)); print('\n[저장] results_aliasing_conditions.txt')

# ══ Fig1: NIS-vs-|ω| aliasing 곡선 ══
def curve(X,atk,ch):
    xs=np.arange(0,3.0,0.3); ys=[]
    for x in xs:
        m=(X[:,1]==atk)&(X[:,2]>=x)&(X[:,2]<x+0.3); v=X[m,3 if ch=='g' else 4]; v=v[~np.isnan(v)]
        ys.append(np.median(v) if len(v) else np.nan)
    return xs+0.15,np.array(ys)
fig,axes=plt.subplots(1,2,figsize=(15,5))
for ax,ch,ttl in [(axes[0],'g','gyro NIS'),(axes[1],'v','vel NIS')]:
    if B is not None: x,y=curve(B,0,ch); ax.plot(x,y,'o-',color='#7c8b9c',label='before 평시(가감속off)')
    if A is not None:
        x,y=curve(A,0,ch); ax.plot(x,y,'o-',color='#C44E52',label='after 평시(가감속on)=aliasing')
        x,y=curve(A,1,ch); ax.plot(x,y,'s--',color='#bd4b3e',label='after 공격')
    ax.set_xlabel('기동강도 |ω|'); ax.set_ylabel(f'{ttl} (압축)'); ax.set_title(f'{ttl} vs 기동강도',fontsize=12)
    ax.set_ylim(0,3.15); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle('기동 aliasing: |ω|(기동강도)↑ → 평시 NIS↑ (가감속 on서). 급기동서 공격과 겹침',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('aliasing_curve.png',dpi=115,bbox_inches='tight'); plt.close(fig)
print('[plot] aliasing_curve.png')

# ══ Fig2: 패턴별 시계열 (after) ══
if A is not None:
    fig,axes=plt.subplots(4,1,figsize=(15,11),sharex=True)
    for pi,(ax,pat) in enumerate(zip(axes,PATS)):
        m=A[:,5]==pi
        if not m.sum(): ax.text(0.5,0.5,f'{pat}:없음',ha='center',transform=ax.transAxes); continue
        e=int(np.median(np.unique(A[m,0])))  # 대표 에피
        d=A[(A[:,0]==e)]; st=np.arange(len(d))
        for i in range(len(d)):
            if d[i,1]==1: ax.axvspan(i-0.5,i+0.5,color='red',alpha=0.10)
            if d[i,2]>1.5: ax.axvspan(i-0.5,i+0.5,color='orange',alpha=0.13)
        ax.plot(st,d[:,3],color='#C44E52',lw=1.2,label='gyro'); ax.plot(st,d[:,4],color='#4C72B0',lw=1.1,label='vel')
        ax.axhline(2.35,color='gray',ls=':',lw=0.8); ax.set_ylim(0,3.15); ax.set_ylabel(f'{pat}\nNIS',fontsize=10)
        ax.legend(loc='upper right',fontsize=8); ax.grid(alpha=0.25)
    axes[-1].set_xlabel('step (10Hz)')
    fig.suptitle('궤적별 NIS 시계열 (가감속 on) — 빨강=공격 주황=급기동. aggressive가 aliasing 최다',fontsize=13,y=1.005)
    fig.tight_layout(); fig.savefig('aliasing_by_pattern.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    print('[plot] aliasing_by_pattern.png')
