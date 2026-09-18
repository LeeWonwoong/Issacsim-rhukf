#!/usr/bin/env python3
# rotwind_plot — arm별 zu_log 리플레이 → wind 가 gyro 스파이크 만드는지 4클래스 분포.
import numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, load_calibration
calib=load_calibration('calibration.json')
ARMS=['0.05','0.10','0.15']
CL=['benign','maneuver','wind','attack']; CO={'benign':'#2a9d4a','maneuver':'#e08a1e','wind':'#3b7dd8','attack':'#d1495b'}

def replay(path):
    z=np.load(path,allow_pickle=True); A=z['data']; dt=float(z['dt']); qg=float(z['q_gate'])
    ZV=slice(4,13); U=slice(13,17); GYR=slice(10,13)
    gmag=np.linalg.norm(A[:,GYR],axis=1)
    ukf=DynamicsUKF(dt=dt,calib=calib,q_gate=qg); NISg=np.full(len(A),np.nan); prev=None; sr=0
    for i in range(len(A)):
        if A[i,1]>0.5: ukf=DynamicsUKF(dt=dt,calib=calib,q_gate=qg); prev=None; sr=0
        z9=A[i,ZV].astype(float); u=A[i,U].astype(float); gps=z9[0:3]
        fr=(prev is None) or (not np.allclose(gps,prev,atol=1e-6)); prev=gps.copy()
        try: res,Pzz=ukf.step(z9,u,gps_fresh=fr)
        except: sr+=1; continue
        sr+=1
        if sr<12: continue
        rg=res[6:9]
        try: NISg[i]=float(rg@np.linalg.solve(Pzz[6:9,6:9],rg))
        except: pass
    VAg=np.minimum(np.log1p(np.sqrt(np.maximum(NISg,0))),3.0)
    atk=A[:,2]>0.5; wind=A[:,23]; valid=~np.isnan(NISg)
    cls=np.full(len(A),'',dtype=object); cls[atk]='attack'; non=~atk
    cls[non&(wind>2)&(gmag<0.6)]='wind'; cls[non&(wind<2)&(gmag>0.7)]='maneuver'; cls[non&(wind<2)&(gmag<0.4)]='benign'
    return VAg,cls,valid

fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
def dp(a,b): a=a[~np.isnan(a)];b=b[~np.isnan(b)]; return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9) if len(a)and len(b) else float('nan')
print(f'{"arm":>5} | {"wind med":>9} {"man med":>8} {"atk med":>8} | {"d(wind-benign)":>14}')
for k,arm in enumerate(ARMS):
    try: VAg,cls,valid=replay(f'results_rotwind_a{arm}/zu_log.npz')
    except Exception as e: print(f'arm {arm}: {e}'); continue
    ax=axes[k]; parts=[];poss=[];cols=[]
    for j,c in enumerate(CL):
        d=VAg[(cls==c)&valid&~np.isnan(VAg)]
        if len(d)<10: continue
        parts.append(d);poss.append(j);cols.append(CO[c])
    vp=ax.violinplot(parts,positions=poss,showmedians=True,widths=0.8)
    for b,cc in zip(vp['bodies'],cols): b.set_facecolor(cc); b.set_alpha(0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(CL,fontsize=8,rotation=15)
    ax.set_title(f'WIND_MOMENT_ARM = {arm}',fontsize=11); ax.grid(alpha=0.25,axis='y')
    def md(c): d=VAg[(cls==c)&valid]; return np.nanmedian(d) if len(d) else np.nan
    dwb=dp(VAg[(cls=='wind')&valid],VAg[(cls=='benign')&valid])
    print(f'{arm:>5} | {md("wind"):>9.2f} {md("maneuver"):>8.2f} {md("attack"):>8.2f} | {dwb:>14.2f}')
axes[0].set_ylabel('gyro V-A  min(log1p√NIS,3)',fontsize=10,fontweight='bold')
plt.suptitle('회전 바람(WIND_MOMENT_ARM) ↑ → wind 가 gyro 스파이크 만드나 (V-A, 4클래스)',fontsize=12)
plt.tight_layout(); plt.savefig('/tmp/rotwind_dist.png',dpi=110,bbox_inches='tight'); print('saved /tmp/rotwind_dist.png')
