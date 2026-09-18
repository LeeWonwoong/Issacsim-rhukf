#!/usr/bin/env python3
# grid64_cal (2026-09-04) — Isaac 정합(CAL) surrogate 에서 64-config 그리드 재실행.
#   09-02 그리드는 Isaac 대비 d' 4~5배 쉬운 대역에서 뽑은 것이라 랭킹 전이 실패(1등 P0.2 → Isaac 7등).
#   CAL:  σ_v=1.04 σ_g=0.86 SURR_ATK_PROB=0.54
#         → d'(v) .237 [Isaac .19~.24] · d'(g) 1.565 [1.47~1.67] · 공격비율 41.1% [44.0%]
#   전이검증: 7 config Spearman ρ=0.786, 절대 F1 6/7 이 Isaac 과 ±0.003.
#   ★ 1차 지표를 F1 이 아니라 (약공격 recall, probe FP) 로 본다 —
#     CAL 에서 F1 폭은 0.008 뿐이지만 약공격 recall 폭은 0.25 (30배 넓음).
#   출력: results_grid_cal/  (repo 내부)
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
os.environ['SURR_ATK_PROB'] = '0.54'
SIGMA = (1.04, 0.86)
OUT = 'results_grid_cal'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
CONFIGS = []
for tau, ui in [('0.02', 4), ('0.005', 1)]:
    for R in ['1', '2']:
        for Q in ['1e-2', '1e-3']:
            for P in ['0.2', '0.1', '0.05', '0.01']:
                for mode in ['error', 'absolute']:
                    name = f't{tau[2:]}u{ui}_R{R}_Q{Q[-1]}_P{P}_{mode[:3]}'
                    env = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q=Q, RHUKF_TAU=tau,
                               RHUKF_UI=ui, RHUKF_R=R)
                    if mode == 'error': env['RHUKF_PD'] = P
                    else: env['RHUKF_FORM'] = 'absolute'; env['RHUKF_PINIT'] = P
                    CONFIGS.append((name, env))
mine = CONFIGS[WID::2]
print(f'[worker {WID}] {len(mine)} configs · CAL σ={SIGMA} atk_p=0.54', flush=True)
out = {}
done = 0
for name, env in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD', 'RHUKF_ALPHA', 'RHUKF_HUBER_C',
              'OPT', 'ADAM_LOSS', 'ADAM_AMSGRAD', 'ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_ATK_PROB'] = '0.54'
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type='rhukf', probe=True, obs_noise=SIGMA)
        s = summarize(hist); s.update(p)
        out[name] = s; done += 1
        print(f'  [{done}/{len(mine)}] {name:26} F1={s["F1"]:.3f} wR={s["probe_weak_rec"]:.2f} '
              f'pFP={s["probe_fp"]:.3f} sR={s["probe_strong_rec"]:.2f} conv={s["conv_ep"]:.0f} '
              f'fpr={s["fpr"]:.3f} rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:26} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e)}
    json.dump(out, open(f'{OUT}/grid_cal_w{WID}.json', 'w'), default=float)
open(f'{OUT}/GRIDCAL_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
