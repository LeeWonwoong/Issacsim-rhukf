"""현 프레임워크 perceptual aliasing 정밀 측정 — 코어 재구동/라벨링 모듈."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

PATS=['circle','figure8','waypoint','aggressive']
CLIP=3.0; QG=5e-3           # 확정 env (tune batch: UKF_Q_GYRO=5e-3, OBS_NORM off → clip3.0)

def replay(path='results_zu_v2', com_bias_std=0.05, seed=0, qg=QG, clip=CLIP):
    """오프라인 UKF 재구동. COM bias 는 online 과 동일하게 UKF u 에만 가산(플랜트 불변)."""
    rng=np.random.RandomState(seed)
    D=np.load(f'{path}/zu_log.npz',allow_pickle=True); data=D['data']; dt=float(D['dt'])
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
    for i in (9,10,11): ukf.Q[i,i]=qg
    rows=[]; sie=0; com=np.zeros(2); prev_g=None; prev_v=None
    for r in data:
        if r[1]==1:   # 에피소드 리셋
            ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0; prev_g=None; prev_v=None
            com = rng.uniform(-com_bias_std,com_bias_std,2) if com_bias_std>0 else np.zeros(2)
        z=r[4:13].astype(float); u=r[13:17].astype(float).copy()
        u[1]+=com[0]; u[2]+=com[1]
        fresh=(sie%5==0)
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0,clip=clip)
        _,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0,clip=clip)
        g=z[6:9]; v=z[3:6]
        wn=float(np.linalg.norm(g))                                  # |ω| 회전 강도
        wd=float(np.linalg.norm(g-prev_g)/dt) if prev_g is not None else 0.0   # |ω̇| 급기동 sharpness
        ax=float(np.linalg.norm(v[:2]-prev_v[:2])/dt) if prev_v is not None else 0.0  # |a_xy| 가감속
        rows.append((r[0], r[2], wn, wd, ax, float(np.linalg.norm(v[:2])),
                     gs, vs, 1.0 if fresh else 0.0, r[20], r[22], r[23]))
        prev_g=g.copy(); prev_v=v.copy(); sie+=1
    A=np.array(rows)
    return dict(ep=A[:,0], atk=A[:,1].astype(int), w=A[:,2], wdot=A[:,3], acc=A[:,4],
                vxy=A[:,5], gyro=A[:,6], vel=A[:,7], fresh=A[:,8].astype(bool),
                delta=A[:,9], pat=A[:,10].astype(int), ws=A[:,11], dt=dt)

# ───────── aliasing 지표 ─────────
def dprime(a,b):
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<5 or len(b)<5: return np.nan
    s=np.sqrt((a.var()+b.var())/2.0)
    return (b.mean()-a.mean())/s if s>0 else np.nan

def auc(neg,pos):
    neg=neg[np.isfinite(neg)]; pos=pos[np.isfinite(pos)]
    if len(neg)<5 or len(pos)<5: return np.nan
    x=np.concatenate([neg,pos]); r=x.argsort().argsort()+1.0
    return (r[len(neg):].sum()-len(pos)*(len(pos)+1)/2.0)/(len(neg)*len(pos))

def overlap(neg,pos,lo=None,hi=None,bins=60):
    """OVL = ∫min(p0,p1) — 두 분포가 겹치는 확률질량. 1=완전 aliasing, 0=완전 분리."""
    neg=neg[np.isfinite(neg)]; pos=pos[np.isfinite(pos)]
    if len(neg)<5 or len(pos)<5: return np.nan
    lo=min(neg.min(),pos.min()) if lo is None else lo
    hi=max(neg.max(),pos.max()) if hi is None else hi
    e=np.linspace(lo,hi,bins+1)
    h0,_=np.histogram(neg,e,density=False); h1,_=np.histogram(pos,e,density=False)
    return float(np.minimum(h0/max(h0.sum(),1),h1/max(h1.sum(),1)).sum())

def bayes_err(neg,pos,bins=60):
    """균형 사전확률(0.5/0.5) 1스텝 최적판정 오류율 = OVL/2 의 히스토그램 추정."""
    return overlap(neg,pos,bins=bins)/2.0

def ambiguous_frac(neg,pos,bins=40,lo_p=0.3,hi_p=0.7):
    """사후확률 P(atk|x) ∈ [0.3,0.7] 인 구간에 떨어지는 표본 비율 = '진짜 애매' 질량."""
    neg=neg[np.isfinite(neg)]; pos=pos[np.isfinite(pos)]
    if len(neg)<5 or len(pos)<5: return np.nan
    lo=min(neg.min(),pos.min()); hi=max(neg.max(),pos.max())
    e=np.linspace(lo,hi,bins+1)
    h0,_=np.histogram(neg,e); h1,_=np.histogram(pos,e)
    p0=h0/max(h0.sum(),1); p1=h1/max(h1.sum(),1)
    with np.errstate(divide='ignore',invalid='ignore'):
        post=np.where((p0+p1)>0, p1/(p0+p1), np.nan)
    amb=(post>=lo_p)&(post<=hi_p)
    return float((p0[amb].sum()+p1[amb].sum())/2.0)

def runs(mask):
    out=[];c=0
    for x in mask:
        if x: c+=1
        elif c>0: out.append(c); c=0
    if c>0: out.append(c)
    return np.array(out) if out else np.array([0])
