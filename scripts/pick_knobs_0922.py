#!/usr/bin/env python3
"""③(v2 노브·희소보상) vs ②(현 노브·희소보상) Isaac s42 ep150+ 공격에피 F1 비교 → 'lowK' 또는 'cur' 출력."""
import csv, numpy as np, sys
def late(p):
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f1=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
    m=(ep>=150)&(b>0); return f1[m].mean(), rw[ep>=150].mean()
R='results/claudecodefortest/isaac_v5/'
f3,r3=late(R+'swirl_v2knobs_s42/metrics_rhukf.csv'); f2,r2=late(R+'swirl_s42/metrics_rhukf.csv')
print(f'v2노브 F1 {f3:.3f} 리턴 {r3:.1f} | 현노브 F1 {f2:.3f} 리턴 {r2:.1f}', file=sys.stderr)
print('lowK' if f3>=f2 else 'cur')
