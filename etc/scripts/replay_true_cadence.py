# -*- coding: utf-8 -*-
"""오프라인 재구동 케이던스 검증 (2026-08-27, 코드-온리).
발견: 전 캡처 실효 RTF=2.5 → 틱은 sim 0.05s 간격, GPS 는 2틱마다 fresh.
구 재구동(Ⅷ·Ⅸ·qr스윕)은 %5 가정 = GPS 를 실제의 2.5배 덜 반영.
여기서는 fresh 를 데이터에서 직접 검출해 '온라인이 실제로 본 필터'를 재현,
config A/E 4목표 지표가 유지되는지 판정."""
import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
CLIP=3.0
CFG_A=dict(q_eul=5e-4,q_vel=5e-3,q_gyr=5e-3,r_vel=0.1, r_gyr=0.2)
CFG_E=dict(q_eul=2e-3,q_vel=5e-3,q_gyr=2e-2,r_vel=0.01,r_gyr=0.02)
SLOT=dict(q_eul=(3,6,'Q'),q_vel=(6,9,'Q'),q_gyr=(9,12,'Q'),r_vel=(3,6,'R'),r_gyr=(6,9,'R'))
D=np.load('results_zu_v3/zu_log.npz',allow_pickle=True)['data']
z6=D[:,4:10]
FRESH=np.r_[True, np.any(np.diff(z6,axis=0)!=0,axis=1)]
delay=D[:,21]; reset=D[:,1]; n=len(D)
so=np.full(n,9999.); c=9999
for i in range(n):
    if reset[i]==1: c=9999
    if delay[i]>=0: c=0
    else: c+=1
    so[i]=c
def replay(cfg, mode, com=0.05, seed=0):
    rng=np.random.RandomState(seed)
    ukf=DynamicsUKF(dt=0.02,calib=load_calibration('calibration.json'))
    for k,(a,b,M) in SLOT.items():
        t=ukf.Q if M=='Q' else ukf.R
        for i in range(a,b): t[i,i]=cfg[k]
    g=np.empty(n); v=np.empty(n); fr=np.zeros(n,bool); cb=np.zeros(2); sie=0
    for i,r in enumerate(D):
        if r[1]==1:
            ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0; cb=rng.uniform(-com,com,2)
        zz=r[4:13].astype(float); u=r[13:17].astype(float).copy(); u[1]+=cb[0]; u[2]+=cb[1]
        fresh = FRESH[i] if mode=='true' else (sie%5==0)
        fr[i]=fresh
        res,Pzz=ukf.step(zz,u,gps_fresh=fresh)
        g[i]=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0,clip=CLIP)[1]
        v[i]=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0,clip=CLIP)[1]
        sie+=1
    return g,v,fr
def four(g,v,fr):
    atk=D[:,2]==1
    BEN=fr&(~atk)&(so>15); ST=fr&atk&(delay>=3)
    def ovl(a,c):
        e=np.linspace(0,3.05,60); h0,_=np.histogram(a,e); h1,_=np.histogram(c,e)
        return float(np.minimum(h0/max(h0.sum(),1),h1/max(h1.sum(),1)).sum())
    out={}
    for nm,x in [('g',g),('v',v)]:
        b=x[BEN]; s=x[ST]
        late=fr&atk&(delay>=8); early=fr&atk&(delay>=3)&(delay<5)
        out[nm]=dict(base=np.median(b),plat=np.median(s),
            snr=(np.median(s)-np.median(b))/max(b.std(),1e-9),
            berr=ovl(b,s)/2,
            sus=np.median(x[late])/max(np.median(x[early]),1e-9) if late.sum()>5 else np.nan)
    # τon/τoff (RL스텝 단위 delay/so 는 sim 기준이라 그대로 유효)
    for nm,x in [('g',g),('v',v)]:
        b=np.median(x[fr&(~atk)&(so>15)])
        prof_on={k:np.median(x[fr&atk&(delay==k)]) for k in range(0,20) if (fr&atk&(delay==k)).sum()>=8}
        plat=np.median([p for k,p in prof_on.items() if k>=5]) if prof_on else np.nan
        ton=np.nan
        if prof_on and np.isfinite(plat) and plat>b:
            thr=b+0.63*(plat-b)
            for k in sorted(prof_on):
                if prof_on[k]>=thr: ton=k; break
        prof_off={k:np.median(x[fr&(~atk)&(so==k)]) for k in range(1,45) if (fr&(~atk)&(so==k)).sum()>=5}
        toff=np.nan
        if prof_off:
            ks=sorted(prof_off); p0=prof_off[ks[0]]
            if p0>b:
                thr=b+0.37*(p0-b)
                toff=next((k for k in ks if prof_off[k]<=thr),ks[-1])
            else: toff=0
        out[nm].update(ton=ton,toff=toff)
    return out
rows=[]
for cfgn,cfg in [('A',CFG_A),('E',CFG_E)]:
    for mode in ['old%5','true']:
        g,v,fr=replay(cfg,'true' if mode=='true' else 'old')
        f=four(g,v,fr); rows.append((cfgn,mode,f))
        print(f"config {cfgn} · {mode:6s} | gyro 바닥 {f['g']['base']:.2f} 고원 {f['g']['plat']:.2f} SNR {f['g']['snr']:.2f} BErr {f['g']['berr']:.3f} 고집 {f['g']['sus']:.2f} τon {f['g']['ton']} τoff {f['g']['toff']}"
              f" | vel 바닥 {f['v']['base']:.2f} SNR {f['v']['snr']:.2f} BErr {f['v']['berr']:.3f} τoff {f['v']['toff']}")
