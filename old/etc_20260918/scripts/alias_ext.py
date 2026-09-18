#!/usr/bin/env python3
# alias_ext (2026-09-05) — alias 스윕 확장 + Huber 기여 격리.
#  A. 확장격자 : P {0.01,0.05,0.1,0.2} × R {1,1.5,2,3} = 16 config,  alias {0,35}, seed {42,43}
#     (기존 P{0.05,0.1}×R{1,1.5} 16런은 재사용 → 신규 48런)
#     ※ 중간레벨(10,20)은 F1 이 비단조라 정보가 적어 **양끝점**만 본다. ΔFP 가 판정 지표.
#  B. Huber OFF : 직전 스윕 강건성 상위 5 config × RHUKF_HUBER_C=1e9 × alias{0,35} × 2시드 = 20런
#     ※ 직전 스윕은 huber_c=3(ON)으로 돌았음. Adam MSE(+0.245) vs Adam Huber(+0.014) 17배 대조의
#       SWIRL 내부판. 유계영향이 SWIRL 강건성에 얼마를 기여하는지 직접 측정.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
for k in ('SURR_V4','SURR_V5','SURR_V51'): os.environ[k]=''
os.environ['SURR_V52']='1'; os.environ['SURR_ATK_PROB']='0.70'
os.environ['SURR_SLOW_W']='1.0'; os.environ['SURR_ALIAS_LEN']='1,3'
OUT='results_alias'; os.makedirs(OUT, exist_ok=True)
WID=int(sys.argv[1]) if len(sys.argv)>1 else 0
from surrogate_run import run_config, summarize
SWB=dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
         RHUKF_ALPHA='0.10', RHUKF_FORM='absolute')
JOBS=[]
# A. 확장격자
for P in ('0.01','0.05','0.1','0.2'):
    for R in ('1','1.5','2','3'):
        JOBS.append((f'SW_P{P}_R{R}', dict(SWB, RHUKF_PINIT=P, RHUKF_R=R), None))
# B. Huber OFF (직전 ΔFP 상위 5)
for P,R in [('0.02','1.5'),('0.05','1'),('0.02','1'),('0.05','1.5'),('0.1','1')]:
    JOBS.append((f'SWnoHub_P{P}_R{R}', dict(SWB, RHUKF_PINIT=P, RHUKF_R=R, RHUKF_HUBER_C='1e9'), 'off'))
ALL=[(a,sd,j) for sd in (42,43) for a in (0,35) for j in JOBS]
mine=ALL[WID::2]
print(f'[worker {WID}] {len(mine)} jobs', flush=True)
done=0
for rate, sd, (name, env, hub) in mine:
    tag=f'{name}_a{rate}_s{sd}'
    if os.path.exists(f'{OUT}/{tag}.json'):
        done+=1; continue
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','RHUKF_R',
              'OPT','ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k,None)
    os.environ.update({'SURR_V52':'1','SURR_ATK_PROB':'0.70','SURR_SLOW_W':'1.0',
                       'SURR_ALIAS_LEN':'1,3','SURR_ALIAS_RATE':str(rate)})
    t0=time.time()
    try:
        hist,p=run_config(env, obs_mode='raw4', n_ep=160, seed=sd, ep_steps=300,
                          agent_type='rhukf', probe=True, obs_noise=0.0)
        s=summarize(hist); s.update(p); s.update(alias=rate, seed=sd, arm=name, huber=(hub or 'on'))
        json.dump({'tag':tag,'summary':s,'hist':hist}, open(f'{OUT}/{tag}.json','w'), default=float)
        done+=1
        print(f'  [{done}/{len(mine)}] {tag:28} F1={s["F1"]:.3f} fpr={s["fpr"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} rwd={s["reward"]:.1f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {tag:28} FAIL {type(e).__name__}: {e}', flush=True)
open(f'{OUT}/EXT_W{WID}_DONE','w').close(); print(f'[worker {WID}] done', flush=True)
