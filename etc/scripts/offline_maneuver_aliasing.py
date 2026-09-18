"""오프라인 기동 aliasing 분석 (급기동+가감속+바람 캡처).
평시(공격X) gyro/vel NIS를 기동강도 |ω|별로 → 급기동이 aliasing 유발하나?
+ I오차(관성 미스매치) 증폭 테스트. + 지속성(공격=지속 vs 기동 aliasing=순간) 대비.
목표: 급기동/바람 aliasing이 공격 수준까지 튀되(순간), 공격은 지속으로 구분 = POMDP.
사용: python offline_maneuver_aliasing.py"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np; np.random.seed(0)
import sys; sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration

D=np.load('results_zu_aggr/zu_log.npz', allow_pickle=True)
data_full=D['data']; dt=float(D['dt'])
eps=np.unique(data_full[:,0])[:4]  # 4에피 (속도)
data=data_full[np.isin(data_full[:,0],eps)]
print(f'subset {len(data)}행 ({len(eps)}에피), 공격 {int(data[:,2].sum())}, dt={dt}')

def replay(I_err=0.0, mtau_extra=0.0, Qg=2e-3):
    calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
    ukf.I=[x*(1+I_err) for x in ukf.I]
    for i in (9,10,11): ukf.Q[i,i]=Qg
    out=[]; sie=0
    for r in data:
        if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
        z=r[4:13].astype(float); u=r[13:17].astype(float).copy()
        w=float(np.linalg.norm(z[6:9]))   # 기동강도 |ω|
        fresh=(sie%5==0)
        res,Pzz=ukf.step(z,u,gps_fresh=fresh)
        _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
        _,vs=compute_nis_scaled(res[3:6],Pzz[3:6,3:6],3.0)
        out.append((int(r[2]),w,float(gs),float(vs) if fresh else np.nan)); sie+=1
    return np.array(out)

def analyze(a, lbl):
    atk=a[:,0]; w=a[:,1]; g=a[:,2]
    ben=(atk==0)
    print(f'\n【{lbl}】 |ω| 분포: 중앙 {np.median(w[ben]):.2f} 90p {np.percentile(w[ben],90):.2f} max {w[ben].max():.2f}')
    print('  평시 기동강도별 gyro NIS (aliasing):')
    for nm,lo,hi in [('gentle',0,0.5),('중간',0.5,1.5),('급기동',1.5,99)]:
        m=ben&(w>=lo)&(w<hi)
        if m.sum(): print(f'    |ω|{nm:6s}({lo}-{hi}): gyro중앙 {np.median(g[m]):.2f} 90p {np.percentile(g[m],90):.2f} max {g[m].max():.2f} (n={m.sum()})')
    am=(atk==1)
    if am.sum(): print(f'    [공격]              : gyro중앙 {np.median(g[am]):.2f} 90p {np.percentile(g[am],90):.2f}')
    # 지속성: 평시 高NIS(>1.5) run-length vs 공격 run-length
    def runs(mask):
        r=[];c=0
        for x in mask:
            if x: c+=1
            elif c>0: r.append(c); c=0
        if c>0: r.append(c)
        return r
    br=runs(ben&(g>1.5)); ar=runs(am)
    print(f'  지속성: 평시 高NIS(>1.5) run 중앙 {np.median(br) if br else 0:.0f}스텝(n={len(br)}) vs 공격 run 중앙 {np.median(ar) if ar else 0:.0f}스텝')

print('='*80)
for I,mt in [(0.0,0),(0.15,0),(0.30,0)]:
    analyze(replay(I_err=I), f'I오차 {I*100:.0f}%')
print('\n → 급기동 gyro가 공격 수준(2-3)까지 튀면 aliasing 성공. I오차가 급기동서 증폭하나 확인.')
print('   지속성: 공격 run이 기동aliasing run보다 길면 → 시간패턴으로 구분(POMDP 학습가능)')
