#!/usr/bin/env python3
# swirl_ft_scan — surrogate v4 에서 SWIRL 파인튜닝 스캔 (Isaac o3 큐와 병행, GPU 공유).
#   축: error-state pΔ {0.01,0.03,0.1} · absolute P {0.01,0.05,0.1} · (+참조 adam_mse)
#   150ep × seed{42,43}. Isaac 대응쌍(o3_swirl_*)과 랭킹 대조해 surrogate 신뢰도도 판정.
import sys, os, time, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
from surrogate_run import run_config, summarize

CB = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_R='1.5')
CONFIGS = [
    ('err_pd001', 'rhukf', dict(CB, RHUKF_PD='0.01')),
    ('err_pd003', 'rhukf', dict(CB, RHUKF_PD='0.03')),
    ('err_pd01',  'rhukf', dict(CB, RHUKF_PD='0.1')),
    ('abs_p001',  'rhukf', dict(CB, RHUKF_FORM='absolute', RHUKF_PINIT='0.01')),
    ('abs_p005',  'rhukf', dict(CB, RHUKF_FORM='absolute', RHUKF_PINIT='0.05')),
    ('abs_p01',   'rhukf', dict(CB, RHUKF_FORM='absolute', RHUKF_PINIT='0.1')),
    ('adam_mse',  'adam',  dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD='0', ADAM_LR='1e-3')),
]
SEEDS = [42, 43]; NEP = 150
res = {}
for name, ag, env in CONFIGS:
    res[name] = []
    # RHUKF_FORM 미설정 시 이전 config 잔류 방지
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD', 'ADAM_LOSS', 'ADAM_AMSGRAD', 'ADAM_LR', 'OPT'):
        os.environ.pop(k, None)
    for sd in SEEDS:
        t0 = time.time()
        h = run_config(env, obs_mode='raw4', n_ep=NEP, seed=sd, agent_type=ag)
        s = summarize(h); res[name].append(s)
        print(f'  {name:10} seed{sd}: F1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} '
              f'delay={s["delay"]:.2f} rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
print('\n══ 집계 (mean±std, 2 seeds) ══')
rank = []
for name, _, _ in CONFIGS:
    f1 = np.mean([r['F1'] for r in res[name]]); f1s = np.std([r['F1'] for r in res[name]])
    rw = np.mean([r['reward'] for r in res[name]])
    rank.append((f1, name, f1s, rw))
    print(f'{name:10}: F1={f1:.3f}±{f1s:.3f}  rwd={rw:.1f}')
rank.sort(reverse=True)
print(f'\n랭킹: ' + ' > '.join(f'{n}({f:.3f})' for f, n, _, _ in rank))
np.save('/tmp/swirl_ft_res.npy', res, allow_pickle=True)
open('SWIRL_FT_DONE', 'w').close()
