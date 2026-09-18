import os,sys,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'etc/scripts'); sys.path.insert(0,'.')
import numpy as np
from alias_core import dprime, auc, overlap, bayes_err, ambiguous_frac, runs, PATS
OUT=[]
def P(s=''): OUT.append(str(s)); print(s)
def bayes2d(n0,n1,bins=26):
    e=np.linspace(0,3.0,bins+1)
    h0,_,_=np.histogram2d(n0[:,0],n0[:,1],[e,e]); h1,_,_=np.histogram2d(n1[:,0],n1[:,1],[e,e])
    return float(np.minimum(h0/h0.sum(),h1/h1.sum()).sum())/2.0

B=np.load('scratchpad/B_cache.npy',allow_pickle=True).item()
ep,st,a,tr=B['ep'],B['step'],B['atk'],B['track']
cont=np.r_[False,(ep[1:]==ep[:-1])&(st[1:]>st[:-1])]
# 평시 = 직전 4개 로그샘플(=20 RL스텝=2s) 동안 공격이 없었던 표본  ← vel tail 오염 제거
clean=np.ones(len(a),bool)
for k in range(1,5):
    prev=np.r_[np.zeros(k,int),a[:-k]]
    same=np.r_[np.zeros(k,bool),(ep[k:]==ep[:-k])]
    clean &= ~((prev==1)&same)
BEN=(a==0)&cont&clean
SUS=(a==1)&cont&(np.r_[False,a[:-1]==1])
P('='*108); P(' Ⅵ. 공격세기 δ 별 aliasing — 학습로그 4런 통합 (온셋 아티팩트 + 복귀 tail 오염 모두 제거)'); P('='*108)
P(f" 평시(TRACK, 공격 후 2s 배제) n={int((BEN&tr).sum()):,}   /   공격 정상상태(TRACK) n={int((SUS&tr).sum()):,}")
P(f" {'δ 구간':13s} {'n':>5s} | {'gyro d′':>7s} {'AUC':>6s} {'OVL':>6s} {'BErr':>6s} {'중앙':>5s} | {'vel d′':>6s} {'AUC':>6s} {'OVL':>6s} {'BErr':>6s} {'중앙':>5s} | {'2D BErr':>7s}")
g0=B['gyro'][BEN&tr]; v0=B['vel'][BEN&tr]
for lo,hi in [(0.10,0.20),(0.20,0.30),(0.30,0.45),(0.45,0.60),(0.60,0.80)]:
    m=SUS&tr&(B['delta']>=lo)&(B['delta']<hi)
    if m.sum()<60: continue
    g1=B['gyro'][m]; v1=B['vel'][m]
    P(f" δ {lo:.2f}-{hi:.2f} {m.sum():5d} | {dprime(g0,g1):7.2f} {auc(g0,g1):6.3f} {overlap(g0,g1,0,3):6.3f} {bayes_err(g0,g1):6.3f} {np.median(g1):5.2f} "
      f"| {dprime(v0,v1):6.2f} {auc(v0,v1):6.3f} {overlap(v0,v1,0,3):6.3f} {bayes_err(v0,v1):6.3f} {np.median(v1):5.2f} | {bayes2d(np.c_[v0,g0],np.c_[v1,g1]):7.3f}")
P(f" {'[평시]':13s} {len(g0):5d} | gyro 중앙 {np.median(g0):.2f} 90p {np.percentile(g0,90):.2f} 99p {np.percentile(g0,99):.2f} | vel 중앙 {np.median(v0):.2f} 90p {np.percentile(v0,90):.2f} 99p {np.percentile(v0,99):.2f}")
open('scratchpad/alias_p4.txt','w').write('\n'.join(OUT))

P(); P('='*108); P(' Ⅶ. 회전바람(arm=0.02)·궤적이 평시 gyro 바닥을 얼마나 올리나 (B, TRACK, clean 평시)'); P('='*108)
gA=B['gyro'][SUS&tr]; THR=np.percentile(gA,10)
P(f' aliasing 판정선 = 공격 정상상태 gyro 하위10% = {THR:.2f}')
P(f" {'조건':16s} {'n':>6s} | {'중앙':>5s} {'90p':>5s} {'99p':>5s} {'max':>5s} | {'aliasing율':>10s}")
for lo,hi,nm in [(0,0.5,'무풍 0-0.5'),(0.5,2,'약풍 0.5-2'),(2,4,'중풍 2-4'),(4,7,'강풍 4-7'),(7,11,'최강 7-10')]:
    m=BEN&tr&(B['ws']>=lo)&(B['ws']<hi)
    if m.sum()<100: continue
    g=B['gyro'][m]
    P(f" {nm:16s} {m.sum():6d} | {np.median(g):5.2f} {np.percentile(g,90):5.2f} {np.percentile(g,99):5.2f} {g.max():5.2f} | {float((g>=THR).mean())*100:9.2f}%")
P()
for pi,nm in enumerate(PATS):
    m=BEN&tr&(B['pat']==pi)
    if m.sum()<100: continue
    g=B['gyro'][m]
    P(f" {nm:16s} {m.sum():6d} | {np.median(g):5.2f} {np.percentile(g,90):5.2f} {np.percentile(g,99):5.2f} {g.max():5.2f} | {float((g>=THR).mean())*100:9.2f}%")
P()
m=BEN&tr&(B['ws']>=7)&(B['pat']==3); g=B['gyro'][m]
P(f" {'최강풍×aggressive':16s} {m.sum():6d} | {np.median(g):5.2f} {np.percentile(g,90):5.2f} {np.percentile(g,99):5.2f} {g.max():5.2f} | {float((g>=THR).mean())*100:9.2f}%   ← 최악조합")
open('scratchpad/alias_p4.txt','w').write('\n'.join(OUT))
