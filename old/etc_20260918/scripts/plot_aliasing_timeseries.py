"""기동 aliasing time-step plot: gyro/vel NIS + |ω|(기동강도) + 공격 구간.
zu_aggr 캡처(SPEED_MOD 가감속 + 급기동 + 바람)를 UKF 재구동 → NIS 시계열.
급기동/가감속 순간 gyro가 튀는(aliasing) vs 공격(지속)을 한눈에.
사용: python plot_aliasing_timeseries.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
import sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

import sys
_path=sys.argv[1] if len(sys.argv)>1 else 'results_zu_aggr'
_out=sys.argv[2] if len(sys.argv)>2 else 'aliasing_timeseries.png'
D=np.load(f'{_path}/zu_log.npz',allow_pickle=True); data=D['data']; dt=float(D['dt'])
# 공격+급기동이 다 있는 에피 고르기
def epi(e): return data[data[:,0]==e]
best=None; bs=-1
for e in np.unique(data[:,0]):
    d=epi(e); atk=d[:,2].sum(); wmax=np.max([np.linalg.norm(r[10:13]) for r in d])
    sc=atk*(wmax>1.0)
    if sc>bs: bs=sc; best=e
d=epi(best)
calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
for i in (9,10,11): ukf.Q[i,i]=2e-3
st=[]; G=[]; V=[]; W=[]; ATK=[]; sie=0
ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1
for k,r in enumerate(d):
    z=r[4:13].astype(float); u=r[13:17].astype(float)
    w=float(np.linalg.norm(z[6:9])); fresh=(sie%5==0)
    res,Pzz=ukf.step(z,u,gps_fresh=fresh)
    _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
    _,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)
    st.append(k); G.append(gs); V.append(vs if fresh else np.nan); W.append(w); ATK.append(int(r[2])); sie+=1
st=np.array(st); G=np.array(G); V=np.array(V); W=np.array(W); ATK=np.array(ATK)

fig,ax=plt.subplots(2,1,figsize=(15,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
# 공격 구간 음영
onsets=np.where((ATK[:-1]==0)&(ATK[1:]==1))[0]; offs=np.where((ATK[:-1]==1)&(ATK[1:]==0))[0]
segs=[]; s=0 if ATK[0]==1 else None
for i in range(1,len(ATK)):
    if ATK[i]==1 and ATK[i-1]==0: s=i
    if ATK[i]==0 and ATK[i-1]==1 and s is not None: segs.append((s,i)); s=None
if s is not None: segs.append((s,len(ATK)))
for a,b in segs: ax[0].axvspan(a,b,color='red',alpha=0.12)
# 급기동 구간(|ω|>1.5) 음영
for i in range(len(W)):
    if W[i]>1.5: ax[0].axvspan(i-0.5,i+0.5,color='orange',alpha=0.15)
ax[0].plot(st,G,color='#C44E52',lw=1.4,label='gyro NIS(압축)')
ax[0].plot(st,V,color='#4C72B0',lw=1.2,label='vel NIS(압축)')
ax[0].axhline(2.35,color='gray',ls=':',lw=0.9,label='급기동 aliasing 90p (2.35)')
ax[0].set_ylabel('압축 NIS [0,3]'); ax[0].set_ylim(0,3.15); ax[0].legend(loc='upper right',fontsize=9); ax[0].grid(alpha=0.25)
ax[0].set_title(f'기동 aliasing time-step (에피{int(best)}): 빨강=공격 / 주황=급기동(|ω|>1.5) → gyro가 둘 다서 튐 = POMDP',fontsize=12)
ax[1].plot(st,W,color='#55A868',lw=1.2); ax[1].axhline(1.5,color='orange',ls=':',lw=0.9)
ax[1].fill_between(st,0,W,where=ATK==1,color='red',alpha=0.08)
ax[1].set_ylabel('기동강도 |ω|'); ax[1].set_xlabel('step (10Hz)'); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(_out,dpi=120,bbox_inches='tight'); plt.close(fig)
print(f'[plot] {_out} (에피 {int(best)}, 공격스텝 {int(ATK.sum())}, 급기동스텝 {int((W>1.5).sum())})')
# 요약
ben=ATK==0
print(f'평시 gyro: 전체중앙 {np.median(G[ben]):.2f} / 급기동(|ω|>1.5) 중앙 {np.median(G[ben&(W>1.5)]):.2f} 90p {np.percentile(G[ben&(W>1.5)],90) if (ben&(W>1.5)).sum() else 0:.2f}')
print(f'공격 gyro: 중앙 {np.median(G[ATK==1]):.2f}')
