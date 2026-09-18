#!/usr/bin/env python3
# swirl_grid64 — surrogate v4 전체 그리드 (2026-09-02 사용자 지정 축):
#   {tau0.02/ui4, tau0.005/ui1} × R{1,2} × Q{1e-2,1e-3} × P{0.2,0.1,0.05,0.01} × {absolute,error}
#   = 64 config × 160ep × seed42.  worker 2개 병렬(인터리브) — 인자: worker_id(0|1)
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
from surrogate_run import run_config, summarize

WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CONFIGS = []
for tau, ui in [('0.02', 4), ('0.005', 1)]:
    for R in ['1', '2']:
        for Q in ['1e-2', '1e-3']:
            for P in ['0.2', '0.1', '0.05', '0.01']:
                for mode in ['error', 'absolute']:
                    name = f't{tau[2:]}u{ui}_R{R}_Q{Q[-1]}_P{P}_{mode[:3]}'
                    env = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q=Q, RHUKF_TAU=tau, RHUKF_UI=ui, RHUKF_R=R)
                    if mode == 'error': env['RHUKF_PD'] = P
                    else: env['RHUKF_FORM'] = 'absolute'; env['RHUKF_PINIT'] = P
                    CONFIGS.append((name, env))
mine = CONFIGS[WID::2]
print(f'[worker {WID}] {len(mine)} configs', flush=True)
out = {}
for name, env in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD'):
        os.environ.pop(k, None)
    t0 = time.time()
    try:
        h = run_config(env, obs_mode='raw4', n_ep=160, seed=42, agent_type='rhukf')
        s = summarize(h)
        out[name] = s
        print(f'  {name:26} F1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:26} FAIL {e}', flush=True)
json.dump(out, open(f'/tmp/grid64_w{WID}.json', 'w'), default=float)
open(f'GRID64_W{WID}_DONE', 'w').close()
