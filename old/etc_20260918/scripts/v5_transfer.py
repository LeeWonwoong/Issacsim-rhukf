#!/usr/bin/env python3
# v5_transfer (2026-09-04) — env5(Isaac 학습분포 풀, 인공노이즈 0)가 Isaac 랭킹을 재현하는가.
#   ★ 핵심 검증: Isaac P 곡선(d′ 2.62~2.93, P0.002 봉우리 F1 .938)을 env5 가 보는가.
#      CAL(v4+인공노이즈)은 P 사다리가 d′ 2.20~2.25 로 완전히 평평해서 P 축에 장님이었다.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V5'] = '1'; os.environ.pop('SURR_V4', None)
os.environ['SURR_ATK_PROB'] = '0.70'
OUT = 'results_v5'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
B = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
         RHUKF_R='1.5', RHUKF_ALPHA='0.10', RHUKF_FORM='absolute')
ISAAC = {'0.002':0.938,'0.005':0.921,'0.01':0.925,'0.02':0.922,'0.05':0.929,'0.1':0.919}
W0 = [(f'v5_P{p}', 'rhukf', dict(B, RHUKF_PINIT=p), ISAAC[p]) for p in ('0.002','0.01','0.05')]
W1 = [(f'v5_P{p}', 'rhukf', dict(B, RHUKF_PINIT=p), ISAAC[p]) for p in ('0.005','0.02','0.1')]
W1 += [('v5_adam3e4','adam',dict(NET_HIDDEN=16,ADAM_LOSS='mse',ADAM_AMSGRAD=0,ADAM_LR='3e-4'),0.927),
       ('v5_adam1e3','adam',dict(NET_HIDDEN=16,ADAM_LOSS='mse',ADAM_AMSGRAD=0,ADAM_LR='1e-3'),0.924)]
mine = W0 if WID==0 else W1
print(f'[worker {WID}] {[c[0] for c in mine]}  env5, obs_noise=0, ep_steps=300', flush=True)
out={}
for name, ag, env, isaac_f1 in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_V5']='1'; os.environ['SURR_ATK_PROB']='0.70'
    t0=time.time()
    try:
        hist,p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, ep_steps=300,
                            agent_type=ag, probe=True, obs_noise=0.0)
        s=summarize(hist); s.update(p); s['isaac_F1']=isaac_f1; out[name]=s
        print(f'  {name:12} F1={s["F1"]:.3f} R={s["R"]:.3f} fpr={s["fpr"]:.3f} pFP={s["probe_fp"]:.3f} '
              f'wR={s["probe_weak_rec"]:.2f} conv={s["conv_ep"]:.0f} | IsaacF1={isaac_f1:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:12} FAIL {type(e).__name__}: {e}', flush=True); out[name]={'FAIL':str(e)}
    json.dump(out, open(f'{OUT}/v5_w{WID}.json','w'), default=float)
open(f'{OUT}/V5_W{WID}_DONE','w').close(); print(f'[worker {WID}] done', flush=True)
