#!/usr/bin/env python3
# noise_ab (2026-09-03) — 관측 노이즈 하에서 SWIRL vs Adam. LL-6(아티팩트) 가설 검증:
#   "RHUKF 의 우위는 관측 노이즈가 있을 때 나타난다" — clean surrogate 에서 FA 우위가
#   재현되지 않은 이유가 노이즈 부재라면, σ 를 올릴수록 SWIRL 이 Adam 을 앞서야 한다.
#   σ 근거: 정상 클래스 NIS sd = v 0.219 / g 0.242  →  0.15(중간) · 0.30(강)
#   σ=0 기준선: SWIRL P0.2 F1 .974/pFP .019 · P0.1 .974/.013 · Adam1e3 .969/.007 · Adam3e4 .969/.011
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
from surrogate_run import run_config, summarize
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SW = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005', RHUKF_UI=1,
          RHUKF_R='1', RHUKF_ALPHA='0.10')
AD = dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0)
CONFIGS = []
for sg in (0.15, 0.30):
    CONFIGS.append((f'swirl_P0.2_s{sg}', 'rhukf', dict(SW, RHUKF_PD='0.2'), sg))
    CONFIGS.append((f'swirl_P0.1_s{sg}', 'rhukf', dict(SW, RHUKF_PD='0.1'), sg))
    CONFIGS.append((f'adam_1e3_s{sg}',   'adam',  dict(AD, ADAM_LR='1e-3'), sg))
    CONFIGS.append((f'adam_3e4_s{sg}',   'adam',  dict(AD, ADAM_LR='3e-4'), sg))
order = [0, 2, 4, 6, 1, 3, 5, 7]                      # 워커별 SWIRL2+Adam2 균형
mine = [CONFIGS[i] for i in order[WID::2]]
print(f'[worker {WID}] {[c[0] for c in mine]}', flush=True)
out = {}
for name, ag, env, sg in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD', 'RHUKF_ALPHA', 'OPT',
              'ADAM_LOSS', 'ADAM_AMSGRAD', 'ADAM_LR'):
        os.environ.pop(k, None)
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type=ag, probe=True, obs_noise=sg)
        s = summarize(hist); s.update(p); s['sigma'] = sg; s['agent'] = ag
        out[name] = s
        print(f'  {name:18} F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} sR={s["probe_strong_rec"]:.2f} '
              f'rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:18} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'sigma': sg, 'agent': ag}
    json.dump(out, open(f'{OUT}/noise_w{WID}.json', 'w'), default=float)
open(f'{OUT}/NOISE_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
