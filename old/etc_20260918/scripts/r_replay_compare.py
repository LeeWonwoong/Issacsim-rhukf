#!/usr/bin/env python3
# r_replay_compare — phase_shock zu 로그를 R_gyro {0.02,0.1,0.5}로 오프라인 리플레이.
#   페이즈별 gyro NIS: 공격지속(①)·wind aliasing·복귀τ(②) 를 R축으로 비교.
import numpy as np, sys
sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, load_calibration
calib=load_calibration('calibration.json')
z=np.load('results_phase_shock/zu_log.npz',allow_pickle=True); A=z['data']; dt=float(z['dt']); qg=float(z['q_gate'])
cols=z['cols'].item() if z['cols'].ndim==0 else str(z['cols'])
ci={c:i for i,c in enumerate(cols.split(','))}
ZV=slice(ci['z0_gpsN'],ci['z8_gyrz']+1); U=slice(ci['u0_thrust'],ci['u3_tz']+1)
RESET=ci['reset']; ATK=ci['attack']; STEP_of=None
# step 은 zu 에 없을 수 있음 → reset 기준 에피 내 인덱스로 페이즈 판정
RGYR=[0.02,0.1,0.5]
def replay(rg):
    ukf=DynamicsUKF(dt=dt,calib=calib,q_gate=qg)
    ukf.R[6,6]=ukf.R[7,7]=ukf.R[8,8]=rg
    NIS=np.full(len(A),np.nan); prev=None; sr=0; k=np.zeros(len(A),int)
    epid=np.zeros(len(A),int); ep=0
    idx=0
    for i in range(len(A)):
        if A[i,RESET]>0.5:
            ukf=DynamicsUKF(dt=dt,calib=calib,q_gate=qg); ukf.R[6,6]=ukf.R[7,7]=ukf.R[8,8]=rg
            prev=None; sr=0; idx=0; ep+=1
        z9=A[i,ZV].astype(float); u=A[i,U].astype(float); gps=z9[0:3]
        fr=(prev is None) or (not np.allclose(gps,prev,atol=1e-6)); prev=gps.copy()
        try: res,Pzz=ukf.step(z9,u,gps_fresh=fr)
        except: sr+=1; idx+=1; continue
        sr+=1; k[i]=idx; epid[i]=ep; idx+=1
        if sr<12: continue
        rg_=res[6:9]
        try: NIS[i]=float(rg_@np.linalg.solve(Pzz[6:9,6:9],rg_))
        except: pass
    return NIS,k,epid
# 페이즈: 에피 내 인덱스 k → 0-60 clean / 60-120 wind / 120-180 wind+atk / 180-240 atk / 240+ clean2
def phase_arr(k, ep_id, atk_col):
    ph=np.full(len(k),-1)
    for e in np.unique(ep_id):
        m=ep_id==e
        ka=k[m&(atk_col>0.5)]
        if len(ka)==0: continue
        on=ka.min()
        b=[on-600,on-300,on,on+300,on+600]
        km=np.where(m,k,-10**9)
        sel=np.select([km<b[0],km<b[1],km<b[2],km<b[3],km<b[4]],[-1,0,1,2,3],4)
        ph[m]=sel[m]
        ph[m&(k<max(b[0],220))]=-1
    return ph
print(f'{"R_gyro":>7} | {"clean":>7} {"wind":>7} {"wind+atk":>9} {"atk":>7} {"clean2":>7} | {"공격지속":>8} {"복귀비":>7} {"wind오염":>8}')
for rg in RGYR:
    NIS,k,epid=replay(rg)
    VA=np.minimum(np.log1p(np.sqrt(np.maximum(NIS,0))),3.0)
    ph=phase_arr(k,epid,A[:,ATK]); valid=(~np.isnan(VA))&(ph>=0)
    m=[np.median(VA[(ph==p)&valid]) for p in range(5)]
    # 공격지속=atk 페이즈 med-clean, 복귀비=clean2/atk(낮을수록 빠른복귀), wind오염=wind med-clean
    sustain=m[3]-m[0]; recov=m[4]/max(m[3],1e-6); windc=m[1]-m[0]
    print(f'{rg:>7} | {m[0]:>7.2f} {m[1]:>7.2f} {m[2]:>9.2f} {m[3]:>7.2f} {m[4]:>7.2f} | {sustain:>8.2f} {recov:>7.2f} {windc:>8.2f}')
print('\n판정: 공격지속↑(①좋음) · 복귀비↓(②좋음, clean2가 atk보다 낮음) · wind오염↓(aliasing 무익 방지)')
