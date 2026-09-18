#!/usr/bin/env python3
# noise_matched v2 (2026-09-03) — Isaac 난이도 정합 대역에서 4-arm 비교.
#   난이도 근거 (실측 d'):  Isaac  v 0.19~0.24 | g 1.47~1.67
#                            surr σ=0  v 1.03   | g 4.28      ← 4~5배 쉬움
#   σ_req = sqrt((Δμ/d'_tgt)² − σ_pool²) → v:1.15 g:0.90 ⇒ σ≈1.0 에서 동시 정합.
#   ∴ σ ∈ {0.6, 0.9, 1.2} 로 Isaac 대역을 관통.
#   4 arm:  SWIRL(Huber ON) / SWIRL(Huber OFF, 기제 대조)
#           Adam+MSE(무계) / ★Adam+Huber(유계) ← huber_off 결과로 필수가 된 베이스라인
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print('[wait] huber_off 완료 대기...', flush=True)
while not (os.path.exists(f'{OUT}/HUBEROFF_W0_DONE') and os.path.exists(f'{OUT}/HUBEROFF_W1_DONE')):
    time.sleep(60)
time.sleep(5)
from surrogate_run import run_config, summarize
SW  = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005', RHUKF_UI=1,
           RHUKF_R='1', RHUKF_ALPHA='0.10', RHUKF_PD='0.2')
ADM = dict(NET_HIDDEN=16, ADAM_LOSS='mse',   ADAM_AMSGRAD=0, ADAM_LR='1e-3')
ADH = dict(NET_HIDDEN=16, ADAM_LOSS='huber', ADAM_AMSGRAD=0, ADAM_LR='1e-3')
W0, W1 = [], []
for sg in (0.6, 0.9, 1.2):
    W0.append((f'M_swirl_s{sg}',      'rhukf', dict(SW), sg))
    W1.append((f'M_swirlNoHub_s{sg}', 'rhukf', dict(SW, RHUKF_HUBER_C='1e9'), sg))
for sg in (0.6, 0.9, 1.2):
    W0.append((f'M_adamMSE_s{sg}',   'adam', dict(ADM), sg))
    W1.append((f'M_adamHuber_s{sg}', 'adam', dict(ADH), sg))
mine = W0 if WID == 0 else W1
print(f'[worker {WID}] {[c[0] for c in mine]}', flush=True)
out = {}
for name, ag, env, sg in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type=ag, probe=True, obs_noise=sg)
        s = summarize(hist); s.update(p); s['sigma'] = sg; s['agent'] = ag
        out[name] = s
        print(f'  {name:20} F1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} conv={s["conv_ep"]:.0f} '
              f'stab={s["stab"]:.3f} pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} '
              f'sR={s["probe_strong_rec"]:.2f} rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:20} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'sigma': sg, 'agent': ag}
    json.dump(out, open(f'{OUT}/matched_w{WID}.json', 'w'), default=float)
open(f'{OUT}/MATCHED_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
