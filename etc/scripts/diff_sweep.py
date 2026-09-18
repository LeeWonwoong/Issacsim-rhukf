#!/usr/bin/env python3
# diff_sweep (2026-09-05) — **과제 난이도** 축에서 강건성 비교. 유계영향은 양쪽 모두 OFF.
#   의도: TD 이상치(오염) 노이즈가 아니라 **학습 난이도 자체**가 올라갈 때 누가 덜 무너지나.
#   축 : SURR_DELTA_HI ∈ {0.7, 0.5, 0.3, 0.2}  (공격 세기 상한 ↓ = 신호 약화)
#        실측 d′(W1) 2.02 → 1.79 → 1.58 → 1.39.  alias=0 이므로 **클래스 혼동 없음**.
#   arm: SWIRL(Huber OFF, RHUKF_HUBER_C=1e9) × 3   ·  Adam MSE lr{1e-3,3e-4}  ·  SGD 1e-2
#        Adam 은 ADAM_LOSS=mse 가 곧 유계영향 OFF → 양쪽 동일 조건.
#   seed 42, 43.  이력 전체 저장 → 학습곡선.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0,'etc/scripts')
for k in ('SURR_V4','SURR_V5','SURR_V51'): os.environ[k]=''
os.environ['SURR_V52']='1'; os.environ['SURR_ATK_PROB']='0.70'
os.environ['SURR_SLOW_W']='1.0'; os.environ['SURR_ALIAS_RATE']='0'
OUT='results_diff'; os.makedirs(OUT, exist_ok=True)
WID=int(sys.argv[1]) if len(sys.argv)>1 else 0
from surrogate_run import run_config, summarize
SWB=dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
         RHUKF_ALPHA='0.10', RHUKF_FORM='absolute', RHUKF_HUBER_C='1e9')   # ← 유계영향 OFF
ARMS=[('SWnoH_P0.01_R1.5','rhukf',dict(SWB, RHUKF_PINIT='0.01', RHUKF_R='1.5')),
      ('SWnoH_P0.02_R1',  'rhukf',dict(SWB, RHUKF_PINIT='0.02', RHUKF_R='1')),
      ('SWnoH_P0.05_R1',  'rhukf',dict(SWB, RHUKF_PINIT='0.05', RHUKF_R='1')),
      ('AdamMSE_1e3','adam',dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='1e-3')),
      ('AdamMSE_3e4','adam',dict(NET_HIDDEN=16, ADAM_LOSS='mse', ADAM_AMSGRAD=0, ADAM_LR='3e-4')),
      ('SGD_1e2',    'adam',dict(NET_HIDDEN=16, OPT='sgd', ADAM_LOSS='mse', ADAM_LR='1e-2'))]
JOBS=[(hi,sd,a) for sd in (42,43) for hi in ('0.7','0.5','0.3','0.2') for a in ARMS]
mine=JOBS[WID::2]
print(f'[worker {WID}] {len(mine)} jobs — 유계영향 양쪽 OFF, alias 0', flush=True)
done=0
for hi, sd, (name, ag, env) in mine:
    tag=f'{name}_d{hi}_s{sd}'
    if os.path.exists(f'{OUT}/{tag}.json'): done+=1; continue
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','RHUKF_R',
              'OPT','ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k,None)
    os.environ.update({'SURR_V52':'1','SURR_ATK_PROB':'0.70','SURR_SLOW_W':'1.0',
                       'SURR_ALIAS_RATE':'0','SURR_DELTA_LO':'0.1','SURR_DELTA_HI':hi})
    t0=time.time()
    try:
        hist,p=run_config(env, obs_mode='raw4', n_ep=160, seed=sd, ep_steps=300,
                          agent_type=ag, probe=True, obs_noise=0.0)
        s=summarize(hist); s.update(p); s.update(dhi=float(hi), seed=sd, arm=name)
        json.dump({'tag':tag,'summary':s,'hist':hist}, open(f'{OUT}/{tag}.json','w'), default=float)
        done+=1
        print(f'  [{done}/{len(mine)}] {tag:26} F1={s["F1"]:.3f} rec={s["R"]:.3f} fpr={s["fpr"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} conv={s["conv_ep"]:.0f} '
              f'rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {tag:26} FAIL {type(e).__name__}: {e}', flush=True)
open(f'{OUT}/DIFF_W{WID}_DONE','w').close(); print(f'[worker {WID}] done', flush=True)
