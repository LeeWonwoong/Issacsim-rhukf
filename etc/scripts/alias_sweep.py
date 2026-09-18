#!/usr/bin/env python3
# alias_sweep (2026-09-05) — 기동 aliasing 을 키워가며 SWIRL vs Adam/SGD 강건성 비교.
#   축: SURR_ALIAS_RATE ∈ {0, 10, 20, 35}  (100스텝당 alias 발생수, 길이 1~2스텝)
#       평시 프레임을 공격 풀에서 뽑아 삽입 → 한 프레임으론 구별 불가, 지속시간으로만 구별.
#       실측 캘리브: d′(W1) 2.02→1.41→1.15→0.83,  POMDP지수 1.01→1.11→1.19→1.21
#   arm: SWIRL P{0.02,0.05,0.1} × R{1,1.5} = 6  +  Adam+Huber 3e-4 · Adam MSE 1e-3 · SGD 1e-2
#   seed: 42, 43   (n=1 은 Isaac 에서 오판을 낳았음 — 최소 2시드)
#   총 9×4×2 = 72런.  이력 전체 저장 → 학습곡선.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V51'] = ''; os.environ['SURR_V5'] = ''; os.environ['SURR_V4'] = ''
os.environ['SURR_V52'] = '1'
os.environ['SURR_ATK_PROB'] = '0.70'; os.environ['SURR_SLOW_W'] = '1.0'; os.environ['SURR_ALIAS_LEN'] = '1,3'
OUT = 'results_alias'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
SWB = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
           RHUKF_ALPHA='0.10', RHUKF_FORM='absolute')
ARMS = []
for P in ('0.02', '0.05', '0.1'):
    for R in ('1', '1.5'):
        ARMS.append((f'SW_P{P}_R{R}', 'rhukf', dict(SWB, RHUKF_PINIT=P, RHUKF_R=R)))
ARMS.append(('AdamHuber_3e4', 'adam', dict(NET_HIDDEN=16, ADAM_LOSS='huber', ADAM_AMSGRAD=0, ADAM_LR='3e-4')))
ARMS.append(('AdamMSE_1e3',   'adam', dict(NET_HIDDEN=16, ADAM_LOSS='mse',   ADAM_AMSGRAD=0, ADAM_LR='1e-3')))
ARMS.append(('SGD_1e2',       'adam', dict(NET_HIDDEN=16, OPT='sgd', ADAM_LOSS='mse', ADAM_LR='1e-2')))
JOBS = [(rate, sd, a) for sd in (42, 43) for rate in (0, 10, 20, 35) for a in ARMS]
mine = JOBS[WID::2]
print(f'[worker {WID}] {len(mine)} jobs', flush=True)
done = 0
for rate, sd, (name, ag, env) in mine:
    tag = f'{name}_a{rate}_s{sd}'
    if os.path.exists(f'{OUT}/{tag}.json'):
        done += 1; print(f'  SKIP {tag}', flush=True); continue
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','RHUKF_R',
              'OPT','ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ.update({'SURR_V52':'1','SURR_ATK_PROB':'0.70','SURR_SLOW_W':'1.0',
                       'SURR_ALIAS_LEN':'1,3','SURR_ALIAS_RATE':str(rate)})
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=sd, ep_steps=300,
                             agent_type=ag, probe=True, obs_noise=0.0)
        s = summarize(hist); s.update(p); s.update(alias=rate, seed=sd, arm=name)
        json.dump({'tag': tag, 'summary': s, 'hist': hist}, open(f'{OUT}/{tag}.json', 'w'), default=float)
        done += 1
        print(f'  [{done}/{len(mine)}] {tag:26} F1={s["F1"]:.3f} rec={s["R"]:.3f} fpr={s["fpr"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} conv={s["conv_ep"]:.0f} '
              f'rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {tag:26} FAIL {type(e).__name__}: {e}', flush=True)
open(f'{OUT}/ALIAS_W{WID}_DONE', 'w').close(); print(f'[worker {WID}] done', flush=True)
