#!/usr/bin/env python3
# v6 정합 검증 v2 — 순수 정책 분리 (track-only / hover-only) 로 선택편향 제거
import sys, os, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0,'etc/scripts')
os.environ['SURR_V6']='1'
from surrogate_env6 import SurrogateEnv
def auc(a,b):
    x=np.r_[a,b]; y=np.r_[np.ones(len(a)),np.zeros(len(b))]
    o=np.argsort(x); ys=y[o]; n1=ys.sum(); n0=len(ys)-n1
    r=np.arange(1,len(ys)+1); return (r[ys==1].sum()-n1*(n1+1)/2)/(n1*n0)
# ── ① track-only: 온셋 AUC by δ + 형태학
tc=[]; onset={0:[],1:[],2:[]}; runs=[]; cur=0
for ep in range(300):
    e=SurrogateEnv(seed=ep); e.reset()
    for t in range(e.ep_steps):
        v,g,a=e.nis(0)
        d=float(e.dl[t]) if a else 0
        if a and d>=0.1: onset[0 if d<0.3 else (1 if d<0.5 else 2)].append(g)
        elif not a:
            tc.append(g)
            if g>1.2: cur+=1
            else:
                if cur: runs.append(cur); cur=0
        if e.step(): break
tc=np.array(tc); r=np.array(runs)
print("── ① track-only (온셋 판별·형태학)")
print(f"  track 평시 중앙 {np.median(tc):.3f}                      [타깃 ~0.42]")
for i,lab,tgt in [(0,'약 δ<0.3',0.87),(1,'중 0.3~0.5',0.91),(2,'강 ≥0.5',0.94)]:
    print(f"  온셋 AUC {lab:10s} {auc(onset[i],tc):.3f}            [Isaac {tgt}]")
print(f"  평시 초과 run(>1.2): p90 {np.percentile(r,90) if len(r) else 0:.0f} 최대 {r.max() if len(r) else 0}  [Isaac 최대 2]")
# ── ② 공격 온셋에서 호버 래치 (dhover2 유사): 호버 중 판별 + 바닥
hov_on={0:[],1:[],2:[]}; hov_off23=[]; hov_entry=[]; hov_set=[]
for ep in range(300):
    e=SurrogateEnv(seed=1000+ep); e.reset()
    act=0; seen=0; se=99
    for t in range(e.ep_steps):
        v,g,a=e.nis(act)
        d=float(e.dl[t]) if a else 0
        se=0 if a else min(se+1,99)
        if act==1:
            if a and d>=0.1: hov_on[0 if d<0.3 else (1 if d<0.5 else 2)].append(g)
            elif 2<=se<=3: hov_off23.append(g)
        seen = seen+1 if a else 0
        act = 1 if seen>=3 or (act==1 and se<15) else 0   # 온셋+3 래치, 종료+15 복귀
        if e.step(): break
# ③ 평시 강제 호버 (진입/정착 바닥)
for ep in range(80):
    e=SurrogateEnv(seed=2000+ep); e.reset()
    dw=-1
    for t in range(e.ep_steps):
        act=1 if (t//20)%2==1 else 0     # 20스텝 주기 강제 호버
        v,g,a=e.nis(act)
        dw=(dw+1 if dw>=0 else 0) if act==1 else -1
        if not a and act==1:
            (hov_entry if dw<=2 else hov_set).append(g)
        if e.step(): break
print("\n── ② 호버 문맥")
print(f"  호버 진입(평시) 중앙 {np.median(hov_entry):.3f} / 정착 {np.median(hov_set):.3f}   [진입 ~1.04, 정착 ~1.0+]")
print(f"  종료+2~3(호버) 중앙 {np.median(hov_off23):.3f}")
for i,lab,tgt in [(0,'약',0.59),(1,'중',0.87),(2,'강',0.97)]:
    if hov_on[i] and hov_off23:
        print(f"  호버복귀 AUC {lab} {auc(hov_on[i],hov_off23):.3f}   [Isaac {tgt}]")
# ④ 절벽
cr_t=0; cr_h=0
for ep in range(200):
    e=SurrogateEnv(seed=3000+ep); e.reset()
    for t in range(e.ep_steps):
        e.nis(0)
        if e.step(): break
    cr_t+=int(e.crashed)
for ep in range(200):
    e=SurrogateEnv(seed=3000+ep); e.reset(); act=0; seen=0; se=99
    for t in range(e.ep_steps):
        v,g,a=e.nis(act); se=0 if a else min(se+1,99)
        seen=seen+1 if (a and g>1.0) else 0
        act=1 if seen>=2 or (act==1 and se<3) else 0
        if e.step(): break
    cr_h+=int(e.crashed)
print(f"\n── ③ 절벽: 무대응 추락 {cr_t}/200 ({cr_t/2:.0f}%)  vs  대응정책 {cr_h}/200   [track死∧대응生]")
