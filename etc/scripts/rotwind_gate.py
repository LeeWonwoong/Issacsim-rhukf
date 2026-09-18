#!/usr/bin/env python3
# rotwind_gate — rotwind arm 캡처를 리플레이해 wind 가 gyro aliasing 주는지 판정 + arm 선택.
#   출력(stdout 마지막 줄): 선택 arm(float) 또는 'SKIP'. d'(wind-benign) 최대 arm 채택.
import numpy as np, sys
sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, load_calibration
calib=load_calibration('calibration.json')
ARMS=['0.05','0.10','0.15']; THRESH=0.4   # d'(wind-benign) 이 이 이상이면 aliasing 확인

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
    atk=A[:,2]>0.5; wind=A[:,23]; valid=~np.isnan(NISg); non=~atk
    wmask=non&(wind>2)&(gmag<0.6)&valid; bmask=non&(wind<2)&(gmag<0.4)&valid
    return VAg[wmask], VAg[bmask]

def dprime(a,b):
    a=a[~np.isnan(a)]; b=b[~np.isnan(b)]
    if len(a)<10 or len(b)<10: return float('nan')
    return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9)

best_arm=None; best_d=-1
for arm in ARMS:
    try:
        w,b=replay(f'results_rotwind_a{arm}/zu_log.npz')
        d=dprime(w,b)
        sys.stderr.write(f'arm {arm}: d(wind-benign)={d:.2f}  wind_med={np.nanmedian(w):.2f} benign_med={np.nanmedian(b):.2f} (nw={len(w)})\n')
        if not np.isnan(d) and d>best_d: best_d=d; best_arm=arm
    except Exception as e:
        sys.stderr.write(f'arm {arm}: {e}\n')
sys.stderr.write(f'==> best arm={best_arm} d={best_d:.2f} (thresh {THRESH})\n')
print(best_arm if (best_arm and best_d>=THRESH) else 'SKIP')
