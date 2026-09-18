#!/usr/bin/env python3
# huber_off (2026-09-03) — 유계영향(Huber-adaptive R) on/off A/B.
#   질문: SWIRL 의 강건성이 정말 유계영향에서 오는가? 배치평균 adapt_ratio 는 1.074(7%)뿐인데
#         꼬리 샘플은 3~6배 잘린다. 성능에 영향이 있는가?
#   설계: 상위 2 config(α0.1 × P₀{0.2, 0.1}) × σ{0, 0.15, 0.3} × huber OFF(c=1e9) = 6런.
#         ON 쪽은 이미 있음 → 2×2×3 완전요인.
#   ★ noise_ab 완료 후 자동 시작.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print('[wait] noise_ab 완료 대기...', flush=True)
while not (os.path.exists(f'{OUT}/NOISE_W0_DONE') and os.path.exists(f'{OUT}/NOISE_W1_DONE')):
    time.sleep(60)
time.sleep(5)
from surrogate_run import run_config, summarize
SW = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005', RHUKF_UI=1,
          RHUKF_R='1', RHUKF_ALPHA='0.10', RHUKF_HUBER_C='1e9')     # ← 유계영향 OFF
CONFIGS = []
for sg in (0.0, 0.15, 0.30):
    CONFIGS.append((f'nohub_P0.2_s{sg}', dict(SW, RHUKF_PD='0.2'), sg))
    CONFIGS.append((f'nohub_P0.1_s{sg}', dict(SW, RHUKF_PD='0.1'), sg))
mine = CONFIGS[WID::2]
print(f'[worker {WID}] {[c[0] for c in mine]}', flush=True)
out = {}
for name, env, sg in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD', 'OPT', 'ADAM_LOSS', 'ADAM_AMSGRAD', 'ADAM_LR'):
        os.environ.pop(k, None)
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42,
                             agent_type='rhukf', probe=True, obs_noise=sg)
        s = summarize(hist); s.update(p); s['sigma'] = sg; s['huber'] = 'off'
        out[name] = s
        print(f'  {name:18} F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} sR={s["probe_strong_rec"]:.2f} '
              f'rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:18} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'sigma': sg}
    json.dump(out, open(f'{OUT}/huberoff_w{WID}.json', 'w'), default=float)
open(f'{OUT}/HUBEROFF_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
