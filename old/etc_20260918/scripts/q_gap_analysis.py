"""zu_log 재구동으로 Q_gyro별 gyro NIS — gap 채우나 + 스케일차 + 공격 d′. (raw→압축 comp())"""
import sys, os, numpy as np
sys.path.insert(0,'/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus']=False
comp=lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x,0.0))),3.0)
d=np.load(sys.argv[1]); arr=d['data'] if 'data' in d else d[d.files[0]]
RESET,ATK=arr[:,1],arr[:,2]; Z,U=arr[:,4:13],arr[:,13:17]
dz=np.abs(np.diff(Z[:,0:6],axis=0)).sum(axis=1); FRESH=np.concatenate([[True],dz>1e-9])
starts=list(np.where(RESET>0.5)[0])
if not starts or starts[0]!=0: starts=[0]+starts
EP=[(starts[i],starts[i+1] if i+1<len(starts) else arr.shape[0]) for i in range(len(starts))]
calib=load_calibration()
def replay(qg):  # raw NIS 반환
    NG=np.full(Z.shape[0],np.nan)
    for a,b in EP:
        f=DynamicsUKF(dt=0.02,calib=calib)
        for i in (6,7,8): f.R[i,i]=0.2
        for i in (9,10,11): f.Q[i,i]=qg
        for k in range(a,b):
            res,Pzz=f.step(Z[k],U[k],gps_fresh=bool(FRESH[k])); g,_=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0); NG[k]=g
    return NG
def dpr(a,b): return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9) if a.size>5 and b.size>5 else float('nan')
ben=ATK<=0.5; atk=ATK>0.5
print('zu=%s  %d스텝 %d에피 공격%d/평시%d'%(sys.argv[1],arr.shape[0],len(EP),int(atk.sum()),int(ben.sum())))
print('%7s | %16s %10s | %14s %8s | %9s | %6s'%('Q_gyro','평시raw p50/p90','공격raw p50','평시압축 p50/p90','공격압축','평시 1-2%','d_prime'))
Qs=[5e-3,2e-3,1e-3,5e-4]; store={}
for qg in Qs:
    NG=replay(qg); store[qg]=NG
    braw=NG[ben&np.isfinite(NG)]; araw=NG[atk&np.isfinite(NG)]
    bc=comp(braw); ac=comp(araw)
    gapf=((bc>1.0)&(bc<2.0)).mean()*100
    print('%7.0e | %7.1f/%8.1f %10.1f | %6.2f/%7.2f %8.2f | %8.1f%% | %6.2f'%(
        qg,np.percentile(braw,50),np.percentile(braw,90),np.percentile(araw,50),
        np.percentile(bc,50),np.percentile(bc,90),np.percentile(ac,50),gapf,dpr(ac,bc)))
print('→ Q↓: 평시압축 p90이 1-2로 오르면 gap 채움. 스케일차↓. 단 평시가 3에 닿으면 오탐↑·d′↓.')
fig,axes=plt.subplots(1,len(Qs),figsize=(4.5*len(Qs),4.5),sharey=True)
bins=np.linspace(0,3,40)
for ax,qg in zip(axes,Qs):
    NG=store[qg]
    ax.hist(comp(NG[ben&np.isfinite(NG)]),bins,alpha=0.55,density=True,label='평시',color='#4C72B0')
    ax.hist(comp(NG[atk&np.isfinite(NG)]),bins,alpha=0.55,density=True,label='공격',color='#C44E52')
    ax.axvspan(1.0,2.0,color='gray',alpha=0.12)
    ax.set_title('Q=%.0e'%qg,fontsize=11); ax.set_xlabel('압축 gyro NIS'); ax.legend(fontsize=8)
fig.suptitle('Q_gyro 낮추면 평시 gyro가 gap(1-2 회색)으로 오르나 = 스케일차 감소 (단 오탐↑)',fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig('results_q_gap.png',dpi=110,bbox_inches='tight'); print('[plot] results_q_gap.png')
