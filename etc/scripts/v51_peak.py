#!/usr/bin/env python3
# v51_peak (2026-09-04) — env5.1 에서 P 봉우리 위치 + R 상호작용 확정.
#   기존: P사다리(R1.5)에서 0.05 최고(F1 .936) · 0.1 급락(.923) → 봉우리가 0.02~0.05 사이 어딘가.
#         P0.05·R1 은 미측정. R 상호작용은 P0.02(R영향 없음) vs P0.1(R↑ 나빠짐) 로 갈림.
#   본 스윕: P {0.03, 0.05, 0.07, 0.1} × R {1, 1.5}  @ Q1e-3  = 8
#   + Q 축 가설검정: P0.1 × R{1,1.5} @ Q1e-2 = 2
#     (근거: Q/P₀ 가 클수록 호라이즌 안에서 P 가 부풀어 설정값이 무의미해진다.
#      P0.02+Q1e-2 → 2.25배 · P0.1+Q1e-2 → 1.25배. 그래서 Q1e-2 는 P≥0.1 에서만 시험한다.)
#   고정: absolute · N5 · tau0.005/ui1 · alpha0.10 · [16,16] · seed42 · 160ep · 300스텝
import sys, os, time, json, math
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V51'] = '1'; os.environ.pop('SURR_V5', None); os.environ.pop('SURR_V4', None)
os.environ['SURR_ATK_PROB'] = '0.70'
OUT = 'results_v51peak'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
CFG = []
for P in ('0.03', '0.05', '0.07', '0.1'):
    for R in ('1', '1.5'):
        CFG.append((f'P{P}_R{R}_Q3', P, R, '1e-3'))
for R in ('1', '1.5'):
    CFG.append((f'P0.1_R{R}_Q2', '0.1', R, '1e-2'))
mine = CFG[WID::2]
print(f'[worker {WID}] {[c[0] for c in mine]}', flush=True)
out = {}
for name, P, R, Q in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_V51']='1'; os.environ['SURR_ATK_PROB']='0.70'
    env = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q=Q, RHUKF_TAU='0.005', RHUKF_UI=1,
               RHUKF_R=R, RHUKF_FORM='absolute', RHUKF_PINIT=P, RHUKF_ALPHA='0.10')
    h = 0.10 * math.sqrt(514 * float(P))
    Pmid = float(P) + 2.5 * float(Q)          # 호라이즌 평균 실효 P (상한)
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, ep_steps=300,
                             agent_type='rhukf', probe=True, obs_noise=0.0)
        s = summarize(hist); s.update(p); s['h_fd']=h; s['P0']=float(P); s['R']=s['R']; s['Q']=Q; s['P_mid']=Pmid
        out[name] = s
        print(f'  {name:14} h={h:5.2f} Peff={Pmid:.3f} F1={s["F1"]:.3f} prec={s["P"]:.3f} rec={s["R"]:.3f} '
              f'fpr={s["fpr"]:.3f} pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} '
              f'conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:14} FAIL {type(e).__name__}: {e}', flush=True); out[name] = {'FAIL': str(e)}
    json.dump(out, open(f'{OUT}/peak_w{WID}.json', 'w'), default=float)
open(f'{OUT}/PEAK_W{WID}_DONE', 'w').close(); print(f'[worker {WID}] done', flush=True)
