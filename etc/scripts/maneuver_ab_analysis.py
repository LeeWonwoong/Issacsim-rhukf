# -*- coding: utf-8 -*-
"""기동 A(등속)/B(기하연동 가감속) POMDP·aliasing 비교 + 기존 문서(Ⅷ·Ⅸ) 정합 점검.
   전부 오프라인: config E 재구동, GPS fresh 는 데이터에서 검출."""
import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
PATS=['circle','figure8','waypoint','aggressive']
def prep(path):
    D=np.load(f'{path}/zu_log.npz',allow_pickle=True)['data']
    z6=D[:,4:10]; FRESH=np.r_[True,np.any(np.diff(z6,axis=0)!=0,axis=1)]
    delay=D[:,21]; reset=D[:,1]; n=len(D)
    so=np.full(n,9999.); c=9999
    for i in range(n):
        if reset[i]==1: c=9999
        if delay[i]>=0: c=0
        else: c+=1
        so[i]=c
    return D,FRESH,delay,so
def replay(D,FRESH,seed=0):
    rng=np.random.RandomState(seed)
    ukf=DynamicsUKF(dt=0.02,calib=load_calibration('calibration.json'))  # config E 기본값
    n=len(D); g=np.empty(n); v=np.empty(n); cb=np.zeros(2)
    for i,r in enumerate(D):
        if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; cb=rng.uniform(-0.05,0.05,2)
        zz=r[4:13].astype(float); u=r[13:17].astype(float).copy(); u[1]+=cb[0]; u[2]+=cb[1]
        res,Pzz=ukf.step(zz,u,gps_fresh=bool(FRESH[i]))
        g[i]=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)[1]
        v[i]=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)[1]
    return g,v
def metrics(D,FRESH,delay,so,g,v):
    atk=D[:,2]==1; fr=FRESH
    BEN=fr&(~atk)&(so>15); ST=fr&atk&(delay>=3)
    def ovl(a,c):
        e=np.linspace(0,3.05,60); h0,_=np.histogram(a,e); h1,_=np.histogram(c,e)
        return float(np.minimum(h0/max(h0.sum(),1),h1/max(h1.sum(),1)).sum())
    out={}
    for nm,x in [('g',g),('v',v)]:
        b=x[BEN]; s=x[ST]
        out[nm]=dict(base=np.median(b),p90=np.percentile(b,90),p99=np.percentile(b,99),
            plat=np.median(s),snr=(np.median(s)-np.median(b))/max(b.std(),1e-9),
            berr=ovl(b,s)/2, alias=float((b>=np.percentile(s,10)).mean())*100)
        # τon/τoff
        base=np.median(b)
        pon={k:np.median(x[fr&atk&(delay==k)]) for k in range(0,15) if (fr&atk&(delay==k)).sum()>=8}
        plat=np.median([p for k,p in pon.items() if k>=5]) if pon else np.nan
        ton=np.nan
        if pon and np.isfinite(plat) and plat>base:
            thr=base+0.63*(plat-base); ton=next((k for k in sorted(pon) if pon[k]>=thr),np.nan)
        poff={k:np.median(x[fr&(~atk)&(so==k)]) for k in range(1,45) if (fr&(~atk)&(so==k)).sum()>=5}
        toff=np.nan
        if poff:
            ks=sorted(poff); p0=poff[ks[0]]
            if p0>base:
                thr2=base+0.37*(p0-base); toff=next((k for k in ks if poff[k]<=thr2),ks[-1])
            else: toff=0
        out[nm].update(ton=ton,toff=toff)
    # 기동 강도 (평시 TRACK)
    zz=D[:,4:13]; u=D[:,13:17]; act=D[:,3]
    w=np.linalg.norm(zz[:,6:9],axis=1); utq=np.linalg.norm(u[:,1:4],axis=1)
    m=fr&(~atk)&(act==0)&(so>15)
    out['dyn']=dict(w_med=np.median(w[m]),w90=np.percentile(w[m],90),w_duty1=float((w[m]>1.0).mean())*100,
                    u_med=np.median(utq[m]),u90=np.percentile(utq[m],90))
    # 패턴별 평시 gyro
    pat=D[:,22]
    thr=np.percentile(g[ST],10)
    out['per_pat']={}
    for pi,nm in enumerate(PATS):
        mm=m&(pat==pi)
        if mm.sum()>200:
            out['per_pat'][nm]=(np.median(g[mm]),np.percentile(g[mm],99),float((g[mm]>=thr).mean())*100)
    return out
if __name__=='__main__':
    R={}
    for tag,path in [('A_등속','results_zu_constA'),('B_가감속','results_zu_decelB'),('참고: s1(작은박스·등속)','results_zu_s1')]:
        D,F,dl,so=prep(path); g,v=replay(D,F)
        R[tag]=metrics(D,F,dl,so,g,v)
    hdr=['지표']+list(R.keys())+['문서 Ⅷ/Ⅸ (RTF2.5 구비행)']
    DOC={'g.base':'0.41-0.49','g.p90':'0.64-1.61','g.alias':'11-17%','g.snr':'2.8-3.9','g.berr':'0.10-0.12',
         'g.ton':'2-3','g.toff':'7-12','v.berr':'0.42-0.47','v.snr':'0.12-0.39','dyn.w_med':'0.31-0.53'}
    rows=[('gyro 평시 중앙','g','base','{:.2f}'),('gyro 평시 90p','g','p90','{:.2f}'),('gyro 평시 99p','g','p99','{:.2f}'),
          ('gyro 공격 고원','g','plat','{:.2f}'),('gyro SNR ↓비자명','g','snr','{:.2f}'),('gyro Bayes오류 ↑','g','berr','{:.3f}'),
          ('gyro aliasing율 %','g','alias','{:.1f}'),('gyro τon','g','ton','{:.0f}'),('gyro τoff','g','toff','{:.0f}'),
          ('vel SNR','v','snr','{:.2f}'),('vel Bayes오류','v','berr','{:.3f}'),('vel aliasing율 %','v','alias','{:.1f}')]
    print(f"{'지표':22s} | "+" | ".join(f"{k:>18s}" for k in R))
    print('-'*100)
    for lbl,ch,k,fmt in rows:
        print(f"{lbl:22s} | "+" | ".join(f"{fmt.format(R[t][ch][k]):>18s}" for t in R))
    print('-'*100)
    print(f"{'평시 |ω| 중앙/90p':22s} | "+" | ".join(f"{R[t]['dyn']['w_med']:.2f}/{R[t]['dyn']['w90']:.2f}".rjust(18) for t in R))
    print(f"{'|ω|>1.0 듀티 %':22s} | "+" | ".join(f"{R[t]['dyn']['w_duty1']:.1f}".rjust(18) for t in R))
    print(f"{'평시 |u_τ| 중앙/90p':22s} | "+" | ".join(f"{R[t]['dyn']['u_med']:.3f}/{R[t]['dyn']['u90']:.3f}".rjust(18) for t in R))
    print()
    print("패턴별 평시 gyro (중앙 / 99p / aliasing율%):")
    for nm in PATS:
        line=f"  {nm:11s}"
        for t in R:
            pp=R[t]['per_pat'].get(nm)
            line += f" | {pp[0]:.2f}/{pp[1]:.2f}/{pp[2]:.1f}%" if pp else " |        -"
        print(line)
