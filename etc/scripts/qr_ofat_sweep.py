# -*- coding: utf-8 -*-
"""Q·R 전 노브 OFAT 스윕 — 원웅님 4목표 동시 평가.
  ① 고집(공격 흡수 안 함)  ② 온셋 빠름  ③ 오프셋 빠름  ④ 자명하지 않음(d′ 과도하지 않음)
상태 12D: pos(0:3) euler(3:6) vel(6:9) gyro(9:12)   측정 9D: pos(0:3) vel(3:6) gyro(6:9)
사용: python3 etc/scripts/qr_ofat_sweep.py [out.json]
"""
import os,sys,json,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

PATH='results_zu_v2'; CLIP=3.0
D=np.load(f'{PATH}/zu_log.npz',allow_pickle=True)
DATA=D['data']; DT=float(D['dt'])
DELAY=DATA[:,21]; RESET=DATA[:,1]; N=len(DELAY)
SINCE=np.full(N,9999.); _c=9999
for i in range(N):
    if RESET[i]==1: _c=9999
    if DELAY[i]>=0: _c=0
    else: _c+=1
    SINCE[i]=_c
EPI=DATA[:,0]

BASE=dict(q_pos=1e-3,q_eul=5e-4,q_vel=5e-3,q_gyr=5e-3,r_pos=0.5,r_vel=0.1,r_gyr=0.2)
SLOT=dict(q_pos=(0,3,'Q'),q_eul=(3,6,'Q'),q_vel=(6,9,'Q'),q_gyr=(9,12,'Q'),
          r_pos=(0,3,'R'),r_vel=(3,6,'R'),r_gyr=(6,9,'R'))

def replay(cfg, com=0.05, seed=0):
    rng=np.random.RandomState(seed)
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=DT,calib=calib)
    for k,(a,b,M) in SLOT.items():
        tgt=ukf.Q if M=='Q' else ukf.R
        for i in range(a,b): tgt[i,i]=cfg[k]
    g=np.empty(N); v=np.empty(N); fr=np.zeros(N,bool)
    sie=0; cb=np.zeros(2)
    for i,r in enumerate(DATA):
        if r[1]==1:
            ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
            cb=rng.uniform(-com,com,2) if com>0 else np.zeros(2)
        z=r[4:13].astype(float); u=r[13:17].astype(float).copy()
        u[1]+=cb[0]; u[2]+=cb[1]
        fresh=(sie%5==0); fr[i]=fresh
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        g[i]=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0,clip=CLIP)[1]
        v[i]=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0,clip=CLIP)[1]
        sie+=1
    return g,v,fr

def profile(x,fr,mask_fn,ks):
    """이벤트 정렬 앙상블 중앙값 프로파일"""
    out=[]
    for k in ks:
        m=fr&mask_fn(k)
        out.append(np.median(x[m]) if m.sum()>=8 else np.nan)
    return np.array(out)

def tau_rise(prof,base,ks):
    """base 에서 plateau 까지 63% 도달 시점 [RL스텝]"""
    fin=prof[np.isfinite(prof)]
    if len(fin)<3: return np.nan
    plat=np.nanmedian(prof[(ks>=5)]) if np.isfinite(prof[(ks>=5)]).any() else np.nanmax(prof)
    if not np.isfinite(plat) or plat<=base: return np.nan
    thr=base+0.63*(plat-base)
    for k,p in zip(ks,prof):
        if np.isfinite(p) and p>=thr: return float(k)
    return np.nan

def tau_fall(prof,base,ks):
    """공격 종료 직후 값에서 base 로 63% 감쇠 시점 [RL스텝]"""
    fin=[(k,p) for k,p in zip(ks,prof) if np.isfinite(p)]
    if len(fin)<3: return np.nan
    p0=fin[0][1]
    if p0<=base: return 0.0
    thr=base+0.37*(p0-base)
    for k,p in fin:
        if p<=thr: return float(k)
    return float(ks[-1])

def dprime(a,b):
    s=np.sqrt((a.var()+b.var())/2.0)
    return (b.mean()-a.mean())/s if s>0 else np.nan
def overlap(a,b,lo=0,hi=3.05,bins=60):
    e=np.linspace(lo,hi,bins+1)
    h0,_=np.histogram(a,e); h1,_=np.histogram(b,e)
    return float(np.minimum(h0/max(h0.sum(),1),h1/max(h1.sum(),1)).sum())

KON=np.arange(0,20); KOFF=np.arange(1,45)
def score(cfg):
    g,v,fr=replay(cfg)
    atk=DATA[:,2]==1
    BEN=fr&(~atk)&(SINCE>15); ST=fr&atk&(DELAY>=3)
    out={}
    for nm,x in [('gyro',g),('vel',v)]:
        base=float(np.median(x[BEN]))
        pon =profile(x,fr,lambda k: (DATA[:,2]==1)&(DELAY==k),KON)
        poff=profile(x,fr,lambda k: (DATA[:,2]==0)&(SINCE==k),KOFF)
        plat=float(np.nanmedian(pon[KON>=5]))
        sus =float(np.nanmedian(pon[(KON>=10)])/plat) if plat>0 else np.nan
        out[nm]=dict(base=base, plateau=plat, margin=plat-base,
                     tau_on=tau_rise(pon,base,KON), tau_off=tau_fall(poff,base,KOFF),
                     sustain=sus, dprime=float(dprime(x[BEN],x[ST])),
                     ovl=overlap(x[BEN],x[ST]),
                     p99=float(np.percentile(x[BEN],99)),
                     floor_cv=float(np.std([np.median(x[BEN&(EPI==e)]) for e in np.unique(EPI)
                                            if (BEN&(EPI==e)).sum()>20])/max(base,1e-9)))
    return out

GRID={
 # Q: 프로세스 노이즈 — 1e-4 ~ 5e-2 광역 (원웅님 지정)
 'q_pos':[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
 'q_eul':[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
 'q_vel':[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
 'q_gyr':[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2],
 # R: 측정 노이즈 — 현행(pos .5 / vel .1 / gyro .2) 기준 ±2오더 광역
 'r_pos':[1e-3,1e-2,5e-2,0.5,2.0,5.0],
 'r_vel':[1e-3,1e-2,5e-2,0.1,0.5,2.0],
 'r_gyr':[1e-3,1e-2,5e-2,0.2,0.5,2.0],
}

if __name__=='__main__':
    res=[]
    print('base config...',flush=True)
    res.append(dict(knob='BASE',val=None,**{'m':score(dict(BASE))}))
    print(json.dumps(res[-1]['m'],default=str)[:200],flush=True)
    for knob,vals in GRID.items():
        for val in vals:
            if abs(val-BASE[knob])/BASE[knob]<1e-9: continue
            cfg=dict(BASE); cfg[knob]=val
            m=score(cfg); res.append(dict(knob=knob,val=val,m=m))
            gm=m['gyro']
            print(f"{knob:6s}={val:<8g} gyro base {gm['base']:.2f} plat {gm['plateau']:.2f} "
                  f"τon {gm['tau_on']:.0f} τoff {gm['tau_off']:.0f} sus {gm['sustain']:.2f} "
                  f"d' {gm['dprime']:.2f} ovl {gm['ovl']:.3f}",flush=True)
    json.dump(res,open(sys.argv[1] if len(sys.argv)>1 else 'scratchpad/qr_ofat.json','w'),default=str)
    print('done')
