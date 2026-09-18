"""오프라인 POMDP/aliasing 다방법 탐색 (빠른판: 2에피 subset).
목표: gyro/vel 바닥을 올려 약공격이 바닥에 묻히는(POMDP) 메커니즘 찾기. 강공격은 탐지유지.
메커니즘: mass오차·COM토크바이어스·I오차·gyro센서노이즈·vel노이즈·drag오차·저Q_gyro.
공격을 δ별(약<0.25/중0.25-0.5/강>0.5) 분리해 각 bin의 바닥대비 gap 측정.
사용: python offline_pomdp_explore.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(0)
import sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

D=np.load('results_zu_capture/zu_log.npz', allow_pickle=True)
data_full=D['data']; dt=float(D['dt'])
# 2에피만 (속도). 에피 경계는 reset==1
eps=np.unique(data_full[:,0])[:2]
data=data_full[np.isin(data_full[:,0],eps)]
print(f'subset: {len(data)}행 ({len(eps)}에피), 공격스텝 {int(data[:,2].sum())}, dt={dt}')
# 공격 δ 분포
sc=data[data[:,2]==1,20]
print(f'공격 δ(atk_scale): 약<0.25 {int((sc<0.25).sum())} / 중0.25-0.5 {int(((sc>=0.25)&(sc<0.5)).sum())} / 강≥0.5 {int((sc>=0.5).sum())}')

def replay(mass_err=0,tq_bias=0,I_err=0,gyro_noise=0,vel_noise=0,drag_err=0,Qg=2e-3,Qv=5e-3,Rg=0.2,Rv=0.1):
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
    ukf.m*=(1+mass_err); ukf.I=[x*(1+I_err) for x in ukf.I]; ukf.drag=ukf.drag*(1+drag_err)
    for i in (6,7,8): ukf.Q[i,i]=Qv
    for i in (9,10,11): ukf.Q[i,i]=Qg
    for i in (3,4,5): ukf.R[i,i]=Rv
    for i in (6,7,8): ukf.R[i,i]=Rg
    out=[]; sie=0
    for r in data:
        if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
        z=r[4:13].astype(float).copy(); u=r[13:17].astype(float).copy()
        if gyro_noise>0: z[6:9]+=np.random.normal(0,gyro_noise,3)
        if vel_noise>0:  z[3:6]+=np.random.normal(0,vel_noise,3)
        u[1]+=tq_bias; u[2]+=tq_bias
        fresh=(sie%5==0)
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
        _,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)
        out.append((int(r[2]),float(r[20]),float(gs),float(vs) if fresh else np.nan)); sie+=1
    return np.array(out)

def analyze(a):
    atk=a[:,0]; sc=a[:,1]; g=a[:,2]; v=a[:,3]
    fg=np.median(g[atk==0]); fv=np.nanmedian(v[atk==0])
    res={'fg':fg,'fv':fv}
    for nm,lo,hi in [('약',0.0,0.25),('중',0.25,0.5),('강',0.5,9)]:
        msk=(atk==1)&(sc>=lo)&(sc<hi)
        res[nm+'g']=np.median(g[msk]) if msk.sum() else np.nan
        res[nm+'v']=np.nanmedian(v[msk]) if msk.sum() else np.nan
    return res

L=[]; P=lambda s='':(L.append(str(s)),print(s))
P('='*100); P(' 오프라인 POMDP 탐색 — 바닥 g/v | 공격 gyro(약/중/강) | 공격 vel(약/중/강). 약이 바닥에 묻히면 POMDP'); P('='*100)
P(' 메커니즘              | 바닥g 바닥v | gyro 약/중/강      | vel 약/중/강')
CFG=[('baseline',dict()),
     ('mass 5%',dict(mass_err=0.05)),('mass 10%',dict(mass_err=0.10)),
     ('COM tq0.02',dict(tq_bias=0.02)),('COM tq0.05',dict(tq_bias=0.05)),
     ('I 20%',dict(I_err=0.20)),
     ('gyro노이즈 x2',dict(gyro_noise=0.05)),('gyro노이즈 x3',dict(gyro_noise=0.10)),
     ('vel노이즈 0.2',dict(vel_noise=0.2)),('vel노이즈 0.4',dict(vel_noise=0.4)),
     ('drag 40%',dict(drag_err=0.40)),
     ('저Q_gyro 5e-4',dict(Qg=5e-4)),
     ('복합(m5+COM.02+gN.05)',dict(mass_err=0.05,tq_bias=0.02,gyro_noise=0.05))]
for lbl,kw in CFG:
    r=analyze(replay(**kw))
    f=lambda x: f'{x:.2f}' if not np.isnan(x) else ' - '
    P(f'  {lbl:20s}| {f(r["fg"])} {f(r["fv"])} | {f(r["약g"])}/{f(r["중g"])}/{f(r["강g"])} | {f(r["약v"])}/{f(r["중v"])}/{f(r["강v"])}')
P('\n → 목표: 바닥이 올라 "약공격 gyro ≈ 바닥"(묻힘=POMDP) 이면서 "강공격 gyro"는 여전히 높음(탐지유지)')
P('   vel도 바닥 대비 약공격이 묻히면 좋음. 여러 메커니즘 복합 가능.')
open('results_pomdp_explore.txt','w').write("\n".join(L))
print('\n[저장] results_pomdp_explore.txt')
