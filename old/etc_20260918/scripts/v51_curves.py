#!/usr/bin/env python3
# v51_curves (2026-09-04) — env5.1 에서 **에피소드별 이력 전체**를 저장해 학습곡선을 뽑는다.
#   기존 스윕들은 summarize() 요약 스칼라만 남겨서 곡선을 못 그렸다.
#   대상: P 사다리 핵심 4점(R1.5) + P0.03·R1 + Adam 2점(참조)
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V51'] = '1'; os.environ.pop('SURR_V5', None); os.environ.pop('SURR_V4', None)
os.environ['SURR_ATK_PROB'] = '0.70'
OUT = 'results_v51curve'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
B = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
         RHUKF_ALPHA='0.10', RHUKF_FORM='absolute')
ALL = [
  ('SWIRL_P0.002_R1.5','rhukf', dict(B, RHUKF_R='1.5', RHUKF_PINIT='0.002')),
  ('SWIRL_P0.01_R1.5', 'rhukf', dict(B, RHUKF_R='1.5', RHUKF_PINIT='0.01')),
  ('SWIRL_P0.03_R1.5', 'rhukf', dict(B, RHUKF_R='1.5', RHUKF_PINIT='0.03')),
  ('SWIRL_P0.03_R1',   'rhukf', dict(B, RHUKF_R='1',   RHUKF_PINIT='0.03')),
  ('SWIRL_P0.1_R1.5',  'rhukf', dict(B, RHUKF_R='1.5', RHUKF_PINIT='0.1')),
  ('Adam_mse_3e-4',    'adam',  dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='3e-4')),
  ('Adam_mse_1e-3',    'adam',  dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='1e-3')),
]
mine = ALL[WID::2]
print(f'[worker {WID}] {[c[0] for c in mine]}', flush=True)
for name, ag, env in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','RHUKF_R',
              'OPT','ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_V51']='1'; os.environ['SURR_ATK_PROB']='0.70'
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, ep_steps=300,
                             agent_type=ag, probe=True, obs_noise=0.0)
        s = summarize(hist); s.update(p)
        json.dump({'name': name, 'agent': ag, 'env': {k: str(v) for k, v in env.items()},
                   'summary': s, 'hist': hist},
                  open(f'{OUT}/{name}.json', 'w'), default=float)
        print(f'  {name:20} F1={s["F1"]:.3f} rwd={s["reward"]:.1f} conv={s["conv_ep"]:.0f} '
              f'ep={len(hist)} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:20} FAIL {type(e).__name__}: {e}', flush=True)
open(f'{OUT}/CURVE_W{WID}_DONE', 'w').close(); print(f'[worker {WID}] done', flush=True)
