# -*- coding: utf-8 -*-
"""RHUKF 하이퍼 그리드 (surrogate, 2026-08-28 사용자 지정 432 config).
   축: (tau,ui)∈{(0.005,1),(0.02,4)} × alpha{0.1,0.5,0.9} × R{1,1.5,2} × Q{1e-2,1e-3,1e-4}
       × N{5,6} × pΔ{0.01,0.03,0.05,0.1}
   실행: python3 etc/scripts/surrogate_grid.py --shard 0 --nshards 2
   출력: /tmp/surr_grid_shard{i}.jsonl (config별 즉시 flush)"""
import os, sys, json, time, itertools, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=2)
ap.add_argument('--nep', type=int, default=100)
args = ap.parse_args()

TAUUI = [(0.02, 4), (0.005, 1)]   # ui4 선행(빠른 절반 먼저)
ALPHA = [0.1, 0.5, 0.9]
R = [1.0, 2.0]
Q = ['1e-2', '1e-3']
N = [5, 6]
PD = [0.01, 0.03, 0.05, 0.1]

grid = list(itertools.product(TAUUI, ALPHA, R, Q, N, PD))
mine = grid[args.shard::args.nshards]
out = f'/tmp/surr_grid_shard{args.shard}.jsonl'
done_keys = set()
if os.path.exists(out):                      # 재시작 시 완료분 skip
    for line in open(out):
        try: done_keys.add(json.loads(line)['key'])
        except Exception: pass

from surrogate_run import run_config, summarize
t_start = time.time()
for i, ((tau, ui), al, r, q, n, pd) in enumerate(mine):
    key = f"t{tau}_ui{ui}_a{al}_r{r}_q{q}_n{n}_pd{pd}"
    if key in done_keys:
        continue
    env = dict(NET_HIDDEN=16, RHUKF_TAU=tau, RHUKF_UI=ui, RHUKF_ALPHA=al,
               RHUKF_R=r, RHUKF_Q=q, RHUKF_N=n, RHUKF_PD=pd)
    t0 = time.time()
    try:
        h = run_config(env, obs_mode='raw4', n_ep=args.nep, seed=42)
        s = summarize(h)
        rec = dict(key=key, tau=tau, ui=ui, alpha=al, R_axis=r, Q_axis=q, N_axis=n, PD_axis=pd,
                   sec=round(time.time()-t0))
        rec.update({('m_'+k): (None if v != v else v) for k, v in s.items()})
    except Exception as e:
        rec = dict(key=key, error=str(e)[:200], sec=round(time.time()-t0))
    with open(out, 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')
    el = time.time() - t_start
    print(f"[shard{args.shard}] {i+1}/{len(mine)} {key} F1={rec.get('m_F1')} ({rec['sec']}s, 경과 {el/60:.0f}m)", flush=True)
print(f"[shard{args.shard}] 완료", flush=True)
