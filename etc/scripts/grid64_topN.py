#!/usr/bin/env python3
# grid64_topN v2 — grid64 완료 → F1 상위 4 config × N{5,6,7} 을 **평가 배터리**로 재실행.
#   배터리: F1/P/R/delay/fpr + 수렴에피(conv_ep) + 안정성(stab) + 약/강 recall + greedy 프로브(FP·약/강 rec·delay)
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
print('[wait] grid64 완료 대기...', flush=True)
while not (os.path.exists('GRID64_W0_DONE') and os.path.exists('GRID64_W1_DONE')):
    time.sleep(60)
time.sleep(5)
res = {}
for w in (0, 1):
    res.update(json.load(open(f'/tmp/grid64_w{w}.json')))
rank = sorted(res.items(), key=lambda kv: -kv[1]['F1'])
print(f'[grid64] {len(res)} configs. 상위 8:', flush=True)
for n, s in rank[:8]:
    print(f'  {n:26} F1={s["F1"]:.3f} rwd={s["reward"]:.1f}', flush=True)
from surrogate_run import run_config, summarize
def env_of(name):
    t, R, Q, P, m = name.split('_')
    tau = '0.' + t[1:t.index('u')]; ui = int(t[t.index('u') + 1:])
    env = dict(NET_HIDDEN=16, RHUKF_Q=('1e-' + Q[1:])[:4], RHUKF_TAU=tau, RHUKF_UI=ui, RHUKF_R=R[1:])
    if m == 'err': env['RHUKF_PD'] = P[1:]
    else: env['RHUKF_FORM'] = 'absolute'; env['RHUKF_PINIT'] = P[1:]
    return env
out = {}
for name, _ in rank[:4]:
    for N in (5, 6, 7):
        for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD'):
            os.environ.pop(k, None)
        env = env_of(name); env['RHUKF_N'] = N
        t0 = time.time()
        h, p = run_config(env, n_ep=160, seed=42, agent_type='rhukf', probe=True)
        s = summarize(h); s.update(p); out[f'{name}_N{N}'] = s
        print(f'  {name}_N{N}: F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f}/{s["probe_weak_delay"]:.0f} '
              f'sR={s["probe_strong_rec"]:.2f}/{s["probe_strong_delay"]:.0f} ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open('/tmp/grid64_topN.json', 'w'), default=float)
open('GRID64_TOPN_DONE', 'w').close()
