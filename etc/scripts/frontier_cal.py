#!/usr/bin/env python3
# frontier_cal (2026-09-04) — CAL surrogate 에서 (학습셋 d′ ↔ held-out probe FP) 전선 그리기.
#   Isaac 실측 구조: P₀↓ → d′↑ · held-out↓  /  P₀↑ → d′↓ · held-out↑ (정규화 다이얼)
#   Adam 은 Isaac 에 2점뿐이라 전선 비교 불가 → CAL(런당 ~200초)에서 조밀하게 채운다.
#   w0: SWIRL P 사다리 6점 (Isaac 과 동일 config: abs·R1.5·Q1e-3·τ0.005/ui1) — 전선 재현 검증
#   w1: Adam(MSE/Huber) × lr 4점 + SGD × lr 4점 = 12점
#   CAL: σ_v=1.04 σ_g=0.86 SURR_ATK_PROB=0.54
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
SIGMA = (1.04, 0.86)
OUT = 'results_frontier'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
SWB = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
           RHUKF_R='1.5', RHUKF_ALPHA='0.10', RHUKF_FORM='absolute')
W0 = [(f'SW_P{p}', 'rhukf', dict(SWB, RHUKF_PINIT=p)) for p in
      ('0.002', '0.005', '0.01', '0.02', '0.05', '0.1')]
W1 = []
for lr in ('1e-4', '3e-4', '1e-3', '3e-3'):
    W1.append((f'AdamMSE_{lr}',   'adam', dict(NET_HIDDEN=16, ADAM_LOSS='mse',   ADAM_AMSGRAD=0, ADAM_LR=lr)))
    W1.append((f'AdamHuber_{lr}', 'adam', dict(NET_HIDDEN=16, ADAM_LOSS='huber', ADAM_AMSGRAD=0, ADAM_LR=lr)))
for lr in ('1e-3', '1e-2', '3e-2', '1e-1'):
    W1.append((f'SGD_{lr}', 'adam', dict(NET_HIDDEN=16, OPT='sgd', ADAM_LOSS='mse', ADAM_LR=lr)))
mine = W0 if WID == 0 else W1
print(f'[worker {WID}] {len(mine)} runs: {[c[0] for c in mine]}', flush=True)
out = {}
for name, ag, env in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_ATK_PROB'] = '0.54'
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type=ag, probe=True, obs_noise=SIGMA)
        s = summarize(hist); s.update(p); out[name] = s
        print(f'  {name:16} R={s["R"]:.3f} fpr={s["fpr"]:.3f} F1={s["F1"]:.3f} | '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} sR={s["probe_strong_rec"]:.2f} '
              f'conv={s["conv_ep"]:.0f} rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:16} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e)}
    json.dump(out, open(f'{OUT}/frontier_w{WID}.json', 'w'), default=float)
open(f'{OUT}/FRONTIER_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
