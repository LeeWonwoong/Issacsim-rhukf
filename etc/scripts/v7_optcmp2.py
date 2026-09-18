#!/usr/bin/env python3
# v7_optcmp — v7(실측절벽) 무대에서 옵티마이저 비교: SWIRL(top) vs Adam+Huber vs SGD × seed{42,43}
#   판정: 옵티마이저가 갈리는가 (F1·곡선·추락수·시드분산). 인자: worker_id(0|1)
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V7'] = '1'
from surrogate_run import run_config, summarize
WID = int(sys.argv[1])
def go(name, env, agent, seed):
    for k in ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_HUBER_C','RHUKF_ALPHA','RHUKF_R','RHUKF_N','ADAM_LR','OPT'):
        os.environ.pop(k, None)
    t0 = time.time()
    h = run_config(env, n_ep=160, seed=seed, ep_steps=400, agent_type=agent)
    s = summarize(h); s['crash'] = int(sum(r['crashed'] for r in h))
    s['crash_late'] = int(sum(r['crashed'] for r in h[80:]))
    s['f1_early'] = float(np.mean([r['f1'] for r in h[20:60] if r['has_atk']]))   # 샘플효율 창
    print(f"  {name:22s} F1={s['F1']:.3f} early={s['f1_early']:.3f} rec={s['R']:.3f} fpr={s['fpr']:.3f} "
          f"crash={s['crash']}/{s['crash_late']}후반 rwd={s['reward']:.0f} ({time.time()-t0:.0f}s)", flush=True)
    return s
SW = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
          RHUKF_ALPHA='0.1', RHUKF_R='1', RHUKF_FORM='absolute', RHUKF_PINIT='0.01', RHUKF_SPAS=1)
jobs = []
for sd in (44, 45):
    jobs.append((f'SWIRL_s{sd}', dict(SW), 'rhukf', sd))
    jobs.append((f'AdamHuber_s{sd}', dict(NET_HIDDEN=16, ADAM_LR='3e-4'), 'adam', sd))
    jobs.append((f'SGD_s{sd}', dict(NET_HIDDEN=16, ADAM_LR='3e-4', OPT='sgd'), 'adam', sd))
mine = jobs[WID::2]; out = {}
print(f'[w{WID}] {len(mine)} runs', flush=True)
for name, env, ag, sd in mine: out[name] = go(name, env, ag, sd)
json.dump(out, open(f'/tmp/v7optcmp2_w{WID}.json','w'), default=float)
open(f'V7OPTCMP2_W{WID}_DONE','w').close()
