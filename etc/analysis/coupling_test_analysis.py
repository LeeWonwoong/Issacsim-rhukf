#!/usr/bin/env python3
"""coupling_test_analysis.py — 커플링 유/무 × hover/aggressive 구간별 res·NIS (2026-08-18)
  시간창: clean<110 / 바람110~160 / 겹침160~200 / 공격만200~250 / 복귀250+
"""
import csv,os,sys
from collections import defaultdict
import numpy as np
A=4.36
def region(s):
    if s<110: return 'clean'
    if s<160: return 'wind'
    if s<200: return 'overlap'
    if s<250: return 'atk'
    return 'recov'
def load(d):
    agg=defaultdict(list)
    for r in csv.DictReader(open(os.path.join(d,'sweep_detail.csv'))):
        if r['policy']!='track': continue
        try:
            k=(r['pattern'],r['disturbance_type'],float(r['bias']),region(int(r['step'])))
            agg[k].append((float(r['res_g']),float(r['res_v']),float(r['nis_g_raw']),float(r['nis_v_raw'])))
        except: pass
    return agg
def med(a,k,c):
    v=a.get(k,[]); return np.median([x[c] for x in v]) if v else float('nan')
d1=sys.argv[1] if len(sys.argv)>1 else 'results_coupling_on'
d2=sys.argv[2] if len(sys.argv)>2 else 'results_coupling_off'
a1,a2=load(d1),load(d2)
pats=sorted({k[0] for k in a1})
print(f"\n{'='*96}\n 커플링ON({d1}) vs OFF({d2}) — res_g / NIS_g (p50)\n{'='*96}")
print("  [평시(bias0) gyro 바닥 — OFF서 오르나 / aggressive가 hover보다 오르나]")
print(f"  {'패턴':>10s} {'외란':>15s} {'구간':>8s} | {'res_g ON→OFF':>14s} | {'NIS_g ON→OFF':>16s}")
for pat in pats:
  for dt in ['none','wind_turbulence']:
    for rg in ['clean','wind']:
      k=(pat,dt,0.0,rg)
      if k in a1 or k in a2:
        print(f"  {pat:>10s} {dt:>15s} {rg:>8s} | {med(a1,k,0):6.3f}→{med(a2,k,0):6.3f} | {med(a1,k,2):7.2f}→{med(a2,k,2):7.2f}")
print("\n  [공격 겹침구간(160~200) — 약공격 δ0.4 / 강공격 δ0.7, gyro NIS]")
print(f"  {'패턴':>10s} {'δ':>4s} | {'NIS_g ON→OFF':>16s} | {'평시대비 배수(OFF)':>18s}")
for pat in pats:
  base=med(a2,(pat,'wind_turbulence',0.0,'wind'),2)  # OFF 평시 바닥
  for b in [1.74,3.05]:
    k=(pat,'wind_turbulence',b,'overlap')
    n1,n2=med(a1,k,2),med(a2,k,2)
    ratio=n2/base if base==base and base>0 else float('nan')
    print(f"  {pat:>10s} {b/A:4.2f} | {n1:7.1f}→{n2:7.1f} | {ratio:8.0f}배")
print("\n─ 읽는 법 ─")
print("  · OFF서 평시 gyro NIS 바닥↑ = 커플링 제거로 표준모델 한계 드러남(CLAUDE 예측 ~2.87).")
print("  · aggressive 평시가 hover보다 크게↑ = 급기동이 gyro 잔차 만듦(=gyro aliasing 원천).")
print("  · 약공격 δ0.4 배수가 작아지면(평시대비) = threshold 애매해짐(RL 필요근거).")
