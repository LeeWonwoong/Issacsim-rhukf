#!/usr/bin/env python3
# axis2_sweep (2026-09-05) — surrogate 상위8 × 2축 비교 + 상위3 R0.5.
#   축① state_form : absolute vs error-state
#       ★2026-09-05 부터 abs 도 RHUKF_SPAS=1 → h=0 argmax 를 시그마 앙상블로 통일.
#         (h>0 은 원래 양쪽 다 θ_current = moving). 이제 두 모드는 파라미터화만 다르다.
#   축② 호라이즌 N  : 5 vs 6                       (N·B ≥ n_x 여유가 늘면 무엇이 바뀌나)
#   → 상위8 × {abs,err} × {N5,N6} × seed{42,43} = 64,  기존 (abs,N5,s42) 8 은 SKIP → 56 신규
#   추가: 상위3 × R=0.5 × {abs,N5} × 2시드 = 6      (R 을 게인 증가 방향으로 처음 밀어봄)
#   ★ diff_sweep 완료 후 자동 시작.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0,'etc/scripts')
for k in ('SURR_V4','SURR_V5','SURR_V51'): os.environ[k]=''
os.environ['SURR_V52']='1'
OUT='results_axis2'; os.makedirs(OUT, exist_ok=True)
WID=int(sys.argv[1]) if len(sys.argv)>1 else 0
print('[wait] diff_sweep 완료 대기...', flush=True)
while not (os.path.exists('results_diff/DIFF_W0_DONE') and os.path.exists('results_diff/DIFF_W1_DONE')):
    time.sleep(120)
time.sleep(5)
from surrogate_run import run_config, summarize
TOP8=[('0.03','1.5'),('0.05','1.5'),('0.03','1'),('0.02','1.5'),
      ('0.02','1'),('0.07','1'),('0.02','2'),('0.1','1')]
# ── h=0 초기화(init) 행동을 3단계로 분해한다 ─────────────────────────────
#   absNS : absolute · SPAS off → h=0 argmax 를 θ_target 에서 (기존 abs 동작)
#   absSP : absolute · SPAS on  → h=0 argmax 를 시그마 앙상블에서 (err 과 동일)
#   err   : error-state         → 항상 시그마 앙상블 (h0_online_moving_init='spas')
#   absNS ↔ absSP  = **init argmax 출처만** 다름 (파라미터화 동일)
#   absSP ↔ err    = **파라미터화만** 다름 (argmax 동일)
#   두 대비를 이어붙이면 "init 차이"와 "좌표계 차이"가 분리된다.
JOBS=[]
for P,R in TOP8:
    for N in (5,6):
        JOBS.append((f'A2_P{P}_R{R}_absSP_N{N}', P, R, 'abs', N, '1'))
        JOBS.append((f'A2_P{P}_R{R}_err_N{N}',   P, R, 'err', N, '1'))
    JOBS.append((f'A2_P{P}_R{R}_absNS_N5', P, R, 'abs', 5, '0'))   # init 대조 (N5 만)
for P,R in TOP8[:3]:
    JOBS.append((f'A2_P{P}_R0.5_absSP_N5', P, '0.5', 'abs', 5, '1'))
ALL=[(sd,j) for sd in (42,43) for j in JOBS]
mine=ALL[WID::2]
print(f'[worker {WID}] {len(mine)} jobs', flush=True)
done=0
for sd,(tag_,P,R,mode,N,spas) in mine:
    tag=f'{tag_}_s{sd}'
    if os.path.exists(f'{OUT}/{tag}.json'): done+=1; continue
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','RHUKF_R','RHUKF_SPAS',
              'OPT','ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k,None)
    os.environ.update({'SURR_V52':'1','SURR_ATK_PROB':'0.70','SURR_SLOW_W':'1.0',
                       'SURR_ALIAS_RATE':'0','SURR_DELTA_LO':'0.1','SURR_DELTA_HI':'0.7',
                       'RHUKF_SPAS':spas})   # ★abs 의 h=0 argmax 출처 (1=시그마앙상블, 0=θ_target)
    env=dict(NET_HIDDEN=16, RHUKF_N=N, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
             RHUKF_R=R, RHUKF_ALPHA='0.10')
    if mode=='abs': env['RHUKF_FORM']='absolute'; env['RHUKF_PINIT']=P
    else:           env['RHUKF_PD']=P                       # error-state = p_delta_init
    t0=time.time()
    try:
        hist,p=run_config(env, obs_mode='raw4', n_ep=160, seed=sd, ep_steps=300,
                          agent_type='rhukf', probe=True, obs_noise=0.0)
        s=summarize(hist); s.update(p); s.update(P0=float(P), R=R, mode=mode, N=N, spas=spas, seed=sd)
        json.dump({'tag':tag,'summary':s,'hist':hist}, open(f'{OUT}/{tag}.json','w'), default=float)
        done+=1
        print(f'  [{done}/{len(mine)}] {tag:30} F1={s["F1"]:.3f} fpr={s["fpr"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} conv={s["conv_ep"]:.0f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {tag:30} FAIL {type(e).__name__}: {e}', flush=True)
open(f'{OUT}/AXIS2_W{WID}_DONE','w').close(); print(f'[worker {WID}] done', flush=True)
