#!/usr/bin/env python3
"""burstduty_analysis.py — burst duty별 결과성(표류) vs 회피성(gyro d′ 단일/윈도우) (2026-08-18)
  사용: burstduty_analysis.py <dir1> <label1> <dir2> <label2> ...
  공격 step30~끝(긴 캠페인). hover track 표류=결과성. gyro d′(캠페인 vs 평시) 단일/윈4/윈8=탐지.
"""
import csv,os,sys
from collections import defaultdict
import numpy as np
def dprime(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.size<5 or b.size<5: return float('nan')
    return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9)
def winmax(v,w): return [max(v[max(0,i-w+1):i+1]) for i in range(len(v))]
def load(d):
    # hover track: 표류(bias>0) / gyro NIS 캠페인(bias>0,step>=30) vs 평시(bias0)
    drift=[]; camp=defaultdict(list); ben=[]
    try: rows=list(csv.DictReader(open(os.path.join(d,'sweep_detail.csv'))))
    except: return None
    for r in rows:
        if r['policy']!='track' or r['pattern']!='hover': continue
        st=int(r['step']); b=float(r['bias']); ng=float(r['nis_g_raw']); ge=float(r['gt_err'])
        if b>0 and st>=30:
            camp[r['episode']].append((st,ng)); drift.append(ge)
        elif b<=0:
            ben.append(ng)
    # 윈도우: 에피소드별 시퀀스에서 winmax
    cs,cw4,cw8=[],[],[]
    for ep,seq in camp.items():
        seq.sort(); vals=[x[1] for x in seq]
        cs+=vals; cw4+=winmax(vals,4); cw8+=winmax(vals,8)
    return dict(drift=np.percentile(drift,95) if drift else float('nan'),
               d_single=dprime(cs,ben), d_w4=dprime(cw4,ben), d_w8=dprime(cw8,ben))
args=sys.argv[1:]
pairs=[(args[i],args[i+1]) for i in range(0,len(args)-1,2)]
print(f"\n{'='*80}\n burst duty: 결과성(표류) vs 회피성(gyro d′)  [hover, δ0.7]\n{'='*80}")
print(f"  {'설정(ON,OFF)':>14s} | {'표류p95':>7s} | {'gyro d′ 단일/윈4/윈8':>20s} | 판정")
for d,lab in pairs:
    r=load(d)
    if not r: print(f"  {lab:>14s} | 데이터없음"); continue
    conseq = r['drift']>2.0
    evasive = r['d_single']<3.0
    verd = ('★결과성∧회피성' if (conseq and evasive) else
            ('결과성O·탐지자명' if conseq else ('회피성O·무결과' if evasive else '무결과·탐지쉬움')))
    print(f"  {lab:>14s} | {r['drift']:6.2f}m | {r['d_single']:5.2f}/{r['d_w4']:5.2f}/{r['d_w8']:5.2f}      | {verd}")
print("\n─ ★결과성∧회피성 = 표류>2m ∧ 단일 d′<3 (윈도우 필요) → 프레임 정합 확정.")
print("  · 표류↑에 단일 d′도↑(자명) 이면 = 결과성과 회피성 양립 불가 → QCD-혼합 서사로.")
