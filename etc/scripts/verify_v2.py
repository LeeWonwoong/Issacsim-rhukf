import numpy as np, warnings, sys; warnings.filterwarnings("ignore")
sys.path.insert(0,'.')
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration
path=sys.argv[1] if len(sys.argv)>1 else 'results_zu_v2'; Qg=float(sys.argv[2]) if len(sys.argv)>2 else 5e-3
D=np.load(f'{path}/zu_log.npz',allow_pickle=True); data=D['data']; dt=float(D['dt'])
calib=load_calibration('calibration.json'); ukf=DynamicsUKF(dt=dt,calib=calib)
for i in (9,10,11): ukf.Q[i,i]=Qg
out=[];sie=0
for r in data:
    if r[1]==1: ukf.x=np.zeros(12); ukf.P=np.eye(12)*0.1; sie=0
    z=r[4:13].astype(float); u=r[13:17].astype(float)
    res,Pzz=ukf.step(z,u,gps_fresh=(sie%5==0))
    _,gs=compute_nis_scaled(res[6:9],Pzz[6:9,6:9],3.0)
    out.append((int(r[2]),np.linalg.norm(z[6:9]),gs,float(r[20]))); sie+=1
a=np.array(out); atk=a[:,0]; W=a[:,1]; G=a[:,2]; sc=a[:,3]
far=np.ones(len(a),bool)
for i in np.where(atk==1)[0]: far[max(0,i-10):min(len(a),i+11)]=False
pb=(atk==0)&far
print(f'[{path} Qg={Qg:.0e}] 공격{int(atk.sum())} 급기동(순수){int((pb&(W>1.5)).sum())}')
print(f'  gentle바닥: {np.median(G[pb&(W<0.5)]):.2f}')
for nm,lo,hi in [('1.0-1.5',1.0,1.5),('급기동>1.5',1.5,99)]:
    m=pb&(W>=lo)&(W<hi)
    if m.sum(): print(f'  순수기동 |ω|{nm}: 중앙 {np.median(G[m]):.2f} 90p {np.percentile(G[m],90):.2f}')
print(f'  공격 gyro: 중앙 {np.median(G[atk==1]):.2f} 90p {np.percentile(G[atk==1],90):.2f}')
mv90=np.percentile(G[pb&(W>1.5)],90) if (pb&(W>1.5)).sum() else 0
print(f'  ★겹침: 기동90p {mv90:.2f} vs 공격중앙 {np.median(G[atk==1]):.2f} → {"겹침(POMDP)" if mv90>np.median(G[atk==1])-0.5 else "gap 큼"}')
# offset
fg=np.median(G[pb&(W<0.5)]); ag=np.median(G[atk==1]); tr=np.where((atk[:-1]==1)&(atk[1:]==0))[0]; hs=[]
for t in tr:
    half=fg+0.5*(ag-fg)
    for k in range(1,15):
        if t+k>=len(G):break
        if G[t+k]<=half: hs.append(k);break
print(f'  offset 반감: {np.median(hs) if hs else 0:.0f}스텝')
