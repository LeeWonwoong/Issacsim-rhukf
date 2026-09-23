#!/usr/bin/env python3
"""n41(B128·N10/N12) vs n40 U(0,9)(B256·N6) → SWIRL greedy F1 2시드 평균 최대(동률 ±0.003 이면 FA 에피 낮은 쪽). 출력: 'BATCH N' 한 줄. NIGHT 에 판정표 기록."""
import json, glob, os, numpy as np, datetime, sys
DRY='--dry' in sys.argv
R='results/claudecodefortest'
def ev(d): e=json.load(open(d+'/eval.json')); return e['f1'], e['fa_episode_rate'], e['prec'], e['fpr']
cands={'B256·N6': sorted(glob.glob(f'{R}/n40_wind_range/a_s_range[[]0.0, 9.0[]]_s4[23]')),
       'B128·N10': sorted(glob.glob(f'{R}/n41_batch128/a_s_N10_s4[23]')), 'B128·N12': sorted(glob.glob(f'{R}/n41_batch128/a_s_N12_s4[23]'))}
adam={'B256': sorted(glob.glob(f'{R}/n40_wind_range/a_a_range[[]0.0, 9.0[]]_s4[23]')), 'B128': sorted(glob.glob(f'{R}/n41_batch128/a_a_batch128_s4[23]'))}
rows={}
for k,ds in {**cands, **{'Adam '+k:v for k,v in adam.items()}}.items():
    ds=[d for d in ds if os.path.exists(d+'/eval.json')]
    if len(ds)<2: print('불완전', k, ds, file=sys.stderr); continue
    m=np.array([ev(d) for d in ds]); rows[k]=(m.mean(0), m[:,0])
best=None
for k in cands:
    if k not in rows: continue
    f1,fa=rows[k][0][0],rows[k][0][1]
    if best is None or f1>rows[best][0][0]+0.003 or (abs(f1-rows[best][0][0])<=0.003 and fa<rows[best][0][1]): best=k
B,N=best.replace('B','').split('·N'); N=int(N); B=int(B)
lines=[f'  {k:10s} F1 {v[0][0]:.3f} [{v[1][0]:.3f}/{v[1][1]:.3f}] · 정밀 {v[0][2]:.3f} · FPR {v[0][3]:.4f} · FA에피 {v[0][1]:.3f}' for k,v in rows.items()]
if not DRY: open(f'{R}/NIGHT_0919.md','a').write(f"- {datetime.datetime.now():%H:%M (%m-%d)} n41 판정(U(0,9)·pΔ0.1, 2시드 greedy):\n"+'\n'.join(lines)+f"\n  → SWIRL **{best}** 선택(규칙: F1 최대, ±0.003 동률이면 FA 낮은 쪽). Adam·칼만-TD 는 같은 batch {B}. Isaac 3시드 기동.\n")
print(B, N)
