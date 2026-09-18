import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'etc/scripts'); sys.path.insert(0,'.')
import numpy as np
from alias_core import replay, dprime, auc, overlap, bayes_err, ambiguous_frac, runs, PATS
OUT=[]
def P(s=''): OUT.append(str(s)); print(s)
R=np.load('scratchpad/A_cache.npy',allow_pickle=True).item()
D=np.load('results_zu_v2/zu_log.npz',allow_pickle=True)['data']
delay=D[:,21]; reset=D[:,1]; n=len(delay)
since_off=np.full(n,9999.0); c=9999
for i in range(n):
    if reset[i]==1: c=9999
    if delay[i]>=0: c=0
    else: c+=1
    since_off[i]=c
f=R['fresh']; atk=R['atk']==1
BEN=f&(~atk)&(since_off>15); STEADY=f&atk&(delay>=3)
gA=R['gyro'][STEADY]; vA=R['vel'][STEADY]
THR_G=np.percentile(gA,10)   # 공격 정상상태의 하위10% = "공격이라면 최소 이 정도" 선
P('='*106); P(' Ⅲ. 기동(급기동·가감속)이 만드는 aliasing — 평시인데 공격처럼 보이는 순간'); P('='*106)
P(f' 기준선: 공격 정상상태 gyro 하위10% = {THR_G:.2f}  (이 위로 올라온 평시 스텝 = gyro aliasing)')
P()
def cond(mask,nm,src):
    m=BEN&mask
    if m.sum()<20: return
    g=R['gyro'][m]
    al=float((g>=THR_G).mean())
    P(f' {nm:22s} n={m.sum():5d} | gyro 중앙 {np.median(g):4.2f} 90p {np.percentile(g,90):4.2f} 99p {np.percentile(g,99):4.2f} max {g.max():4.2f} '
      f'| **aliasing율 {al*100:5.1f}%** | OVL={overlap(g,gA,0,3):.3f}')
P(' ── 회전 강도 |ω| [rad/s] ──')
for lo,hi in [(0,0.3),(0.3,0.6),(0.6,1.0),(1.0,1.5),(1.5,99)]:
    cond((R['w']>=lo)&(R['w']<hi), f'|ω| {lo:.1f}-{hi if hi<90 else 9.9:.1f}', 'w')
P(' ── 각가속도 |ω̇| [rad/s²]  (급기동 sharpness) ──')
for lo,hi in [(0,5),(5,15),(15,30),(30,60),(60,9999)]:
    cond((R['wdot']>=lo)&(R['wdot']<hi), f'|ω̇| {lo}-{hi if hi<9000 else "∞"}', 'wd')
P(' ── 병진 가감속 |a_xy| [m/s²]  (SPEED_MOD 노브의 직접 지표) ──')
for lo,hi in [(0,2),(2,8),(8,20),(20,40),(40,9999)]:
    cond((R['acc']>=lo)&(R['acc']<hi), f'|a| {lo}-{hi if hi<9000 else "∞"}', 'a')
P(' ── 회전바람 풍속 ws [m/s] (arm=0.02 → 바람이 모멘트를 만든다) ──')
for lo,hi in [(0,1),(1,2),(2,5),(5,8),(8,11)]:
    cond((R['ws']>=lo)&(R['ws']<hi), f'ws {lo}-{hi}', 'ws')
P(' ── 궤적 ──')
for pi,nm in enumerate(PATS): cond(R['pat']==pi, nm, 'p')

P(); P('='*106); P(' Ⅳ. 요인 분해 — 평시 gyro NIS 를 무엇이 끌어올리나 (선형회귀, 표준화계수)'); P('='*106)
X=np.c_[R['w'][BEN],R['wdot'][BEN],R['acc'][BEN],R['ws'][BEN]]
y=R['gyro'][BEN]
Xs=(X-X.mean(0))/X.std(0); ys=(y-y.mean())/y.std()
beta,*_=np.linalg.lstsq(np.c_[Xs,np.ones(len(Xs))],ys,rcond=None)
r2=1-((ys-np.c_[Xs,np.ones(len(Xs))]@beta)**2).sum()/ (ys**2).sum()
for nm,b in zip(['|ω| 회전강도','|ω̇| 급기동','|a| 가감속','ws 회전바람'],beta[:4]):
    P(f'   {nm:14s} 표준화β = {b:+.3f}')
P(f'   전체 R² = {r2:.3f}   (나머지는 COM 바이어스·모터지연·센서노이즈 등 설명 안 되는 성분)')
# 단독 R²
for i,nm in enumerate(['|ω|','|ω̇|','|a|','ws']):
    xi=Xs[:,i]; b1=(xi@ys)/(xi@xi); P(f'   {nm:5s} 단독 R² = {(b1*xi@ys)/(ys@ys):.3f}')

P(); P('='*106); P(' Ⅴ. COM 바이어스(±0.05 N·m)의 기여 — 재구동 A/B'); P('='*106)
from alias_core import replay as _rp
R0=_rp('results_zu_v2',com_bias_std=0.0)
b0=R0['fresh']&(R0['atk']==0)&(since_off>15); s0=R0['fresh']&(R0['atk']==1)&(delay>=3)
P(f" {'설정':18s} {'평시gyro중앙':>11s} {'90p':>5s} {'공격중앙':>8s} | {'d′':>5s} {'OVL':>6s} {'BayesErr':>8s} {'aliasing율':>10s}")
for nm,X_ in [('COM 0 (미주입)',R0),('COM ±0.05 (확정)',R)]:
    bb=BEN if nm.startswith('COM ±') else b0; ss=STEADY if nm.startswith('COM ±') else s0
    g=X_['gyro'][bb]; ga=X_['gyro'][ss]; t=np.percentile(ga,10)
    P(f" {nm:18s} {np.median(g):11.2f} {np.percentile(g,90):5.2f} {np.median(ga):8.2f} | "
      f"{dprime(g,ga):5.2f} {overlap(g,ga,0,3):6.3f} {bayes_err(g,ga):8.3f} {float((g>=t).mean())*100:9.1f}%")
open('scratchpad/alias_p2.txt','w').write('\n'.join(OUT))
