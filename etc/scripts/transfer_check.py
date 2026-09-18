#!/usr/bin/env python3
# transfer_check (2026-09-03) — Isaac 정합(CAL) surrogate 가 Isaac 랭킹을 재현하는가?
#   CAL:  σ_v=1.04  σ_g=0.86  SURR_ATK_PROB=0.54
#         → d'(v)=0.237 [Isaac 0.19~0.24] · d'(g)=1.565 [1.47~1.67] · 공격비율 41.1% [44.0%]
#   대조: Isaac o3 마지막 100ep pooled F1 (동일 시드42·200ep·FROZEN-v2)
#   판정: Spearman 순위상관. 재현되면 surrogate 를 튜닝 플랫폼으로 승격.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
os.environ['SURR_ATK_PROB'] = '0.54'          # ← CAL
SIGMA = (1.04, 0.86)                           # ← CAL (σ_v, σ_g)
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
B15 = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_R='1.5')
# (이름, agent, env, Isaac F1)  — Isaac 실측 랭킹 내림차순
ALL = [
 ('T_P0.01abs', 'rhukf', dict(B15, RHUKF_FORM='absolute', RHUKF_PINIT='0.01'), 0.925),
 ('T_P0.1err',  'rhukf', dict(B15, RHUKF_PD='0.1'),                            0.921),
 ('T_P0.1abs',  'rhukf', dict(B15, RHUKF_FORM='absolute', RHUKF_PINIT='0.1'),  0.919),
 ('T_P0.01err', 'rhukf', dict(B15, RHUKF_PD='0.01'),                           0.916),
 ('T_P0.2err',  'rhukf', dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005',
                               RHUKF_UI=1, RHUKF_R='1', RHUKF_PD='0.2'),       0.907),
 ('T_adam3e4',  'adam',  dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='3e-4'), 0.927),
 ('T_adam1e3',  'adam',  dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='1e-3'), 0.924),
]
W0 = [ALL[0], ALL[2], ALL[4]]              # SWIRL 3
W1 = [ALL[1], ALL[3], ALL[5], ALL[6]]      # SWIRL 2 + Adam 2
mine = W0 if WID == 0 else W1
print(f'[worker {WID}] {[c[0] for c in mine]}  (CAL σ={SIGMA} atk_p=0.54)', flush=True)
out = {}
for name, ag, env, isaac_f1 in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_ATK_PROB'] = '0.54'
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type=ag, probe=True, obs_noise=SIGMA)
        s = summarize(hist); s.update(p); s['isaac_F1'] = isaac_f1
        out[name] = s
        print(f'  {name:12} surrF1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} fpr={s["fpr"]:.3f} '
              f'conv={s["conv_ep"]:.0f} pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} '
              f'| IsaacF1={isaac_f1:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:12} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'isaac_F1': isaac_f1}
    json.dump(out, open(f'{OUT}/transfer_w{WID}.json', 'w'), default=float)
open(f'{OUT}/TRANSFER_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
