#!/usr/bin/env python3
"""Adam 최선 셀 선택(U(0,9)): 후보 n40 B256·3e-4·β5 / n41 B128·3e-4·β5 / n42 5셀(B256). greedy F1 2시드 평균 최대(±0.003 동률이면 FA 낮은 쪽).
출력: --set 토큰 한 줄(예: --set agent.batch=256 --set agent.adam.lr=0.0003 --set agent.adam.huber_beta=5.0). NIGHT 기록(--dry 면 기록 안 함)."""
import json, glob, os, sys, numpy as np, datetime, yaml
DRY='--dry' in sys.argv; R='results/claudecodefortest'
def ev(d): e=json.load(open(d+'/eval.json')); return e['f1'], e['fa_episode_rate'], e['prec'], e['fpr']
cands={'B256·lr3e-4·β5': glob.glob(f'{R}/n40_wind_range/a_a_range[[]0.0, 9.0[]]_s4[23]'), 'B128·lr3e-4·β5': glob.glob(f'{R}/n41_batch128/a_a_batch128_s4[23]')}
for d in glob.glob(f'{R}/n42_adam_lr_huber/a_a_*_s42'):
    tag=os.path.basename(d)[4:-4]; cands['B256·'+tag]=glob.glob(f'{R}/n42_adam_lr_huber/a_a_{tag}_s4[23]')
rows={}; cfg={}
for k,ds in cands.items():
    ds=[d for d in sorted(ds) if os.path.exists(d+'/eval.json')]
    if len(ds)<2: print('불완전',k,file=sys.stderr); continue
    m=np.array([ev(d) for d in ds]); rows[k]=(m.mean(0), m[:,0]); c=yaml.safe_load(open(ds[0]+'/config.yaml')); cfg[k]=(c['agent']['batch'], c['agent']['adam']['lr'], c['agent']['adam']['huber_beta'])
best=None
for k in rows:
    f1,fa=rows[k][0][0],rows[k][0][1]
    if best is None or f1>rows[best][0][0]+0.003 or (abs(f1-rows[best][0][0])<=0.003 and fa<rows[best][0][1]): best=k
B,lr,hb=cfg[best]
lines=[f'  {k:16s} F1 {v[0][0]:.3f} [{v[1][0]:.3f}/{v[1][1]:.3f}] · 정밀 {v[0][2]:.3f} · FPR {v[0][3]:.4f} · FA에피 {v[0][1]:.3f}' for k,v in rows.items()]
if not DRY: open(f'{R}/NIGHT_0919.md','a').write(f"- {datetime.datetime.now():%H:%M (%m-%d)} Adam 판정(n40/n41/n42, U(0,9), 2시드 greedy):\n"+'\n'.join(lines)+f"\n  → Isaac Adam = **{best}** (규칙: F1 최대, ±0.003 동률이면 FA 낮은 쪽).\n")
print(f'--set agent.batch={B} --set agent.adam.lr={lr} --set agent.adam.huber_beta={hb}')
