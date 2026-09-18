#!/usr/bin/env python3
# adam_surr (2026-09-03) — surrogate 에서 Adam 베이스라인 측정. SWIRL 승자(0.974)와 같은 조건.
#   순수 Adam(MSE · no-AMSGrad) lr {3e-4, 1e-3} — o3 Isaac 큐가 쓴 것과 동일 설정.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
from surrogate_run import run_config, summarize
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
out = {}
for tag, lr in [('adam_mse_3e4', '3e-4'), ('adam_mse_1e3', '1e-3')]:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD', 'RHUKF_ALPHA', 'OPT'):
        os.environ.pop(k, None)
    env = dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR=lr)
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, agent_type='adam', probe=True)
        s = summarize(hist); s.update(p); s['lr'] = lr
        out[tag] = s
        print(f'  {tag:14} F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} sR={s["probe_strong_rec"]:.2f} '
              f'rwd={s["reward"]:.1f} loss={s["loss"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {tag:14} FAIL {type(e).__name__}: {e}', flush=True)
        out[tag] = {'FAIL': str(e)}
    json.dump(out, open(f'{OUT}/adam_surr.json', 'w'), default=float)
open(f'{OUT}/ADAM_SURR_DONE', 'w').close()
print('done', flush=True)
