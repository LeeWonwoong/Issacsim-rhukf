#!/usr/bin/env python3
# v6_abserr v2 (1시드) — ①상위2 × {abs,err} × Huber{ON3,OFF} ②상위3 × R0.5
#   ③P상향 {0.15,0.2,0.3} (h 붕괴 확인) ④★matched-h 사다리: h=α√(nP) 고정한 채 P/R↑
#     = "P 를 키우면서(FIR 성) 성능 유지가 가능한가" 의 직접 실험. α∝1/√P 보상.
import sys, os, time, json
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V6'] = '1'
from surrogate_run import run_config, summarize
WID = int(sys.argv[1])
ON, OFF = (10, 21), (25, 41)
def go(name, env, seed=42):
    for k in ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_HUBER_C','RHUKF_ALPHA','RHUKF_R'):
        os.environ.pop(k, None)
    t0 = time.time()
    h = run_config(env, n_ep=160, seed=seed, ep_steps=400, on_range=ON, off_range=OFF)
    s = summarize(h); s['crash'] = int(sum(r['crashed'] for r in h)); s['crash_late'] = int(sum(r['crashed'] for r in h[80:]))
    print(f"  {name:34s} F1={s['F1']:.3f} rec={s['R']:.3f} fpr={s['fpr']:.3f} rwd={s['reward']:.0f} crash={s['crash']}/{s['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return s
B = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
         RHUKF_ALPHA='0.1', RHUKF_R='1', RHUKF_SPAS=1)
jobs = []
for P in ('0.01', '0.03'):                                   # ① abs vs err × huber
    for form in ('abs', 'err'):
        for hub, hc in (('hON', '3.0'), ('hOFF', '1e9')):
            e = dict(B, RHUKF_HUBER_C=hc)
            if form == 'abs': e.update(RHUKF_FORM='absolute', RHUKF_PINIT=P)
            else:             e.update(RHUKF_PD=P)
            jobs.append((f'P{P}_{form}_{hub}', e))
for P in ('0.01', '0.03', '0.05'):                           # ② R0.5
    jobs.append((f'P{P}_R0.5', dict(B, RHUKF_R='0.5', RHUKF_FORM='absolute', RHUKF_PINIT=P, RHUKF_HUBER_C='3.0')))
for P in ('0.15', '0.2', '0.3'):                             # ③ P상향 (α0.1 고정 = h 커짐)
    jobs.append((f'Pup{P}_a0.1', dict(B, RHUKF_FORM='absolute', RHUKF_PINIT=P, RHUKF_HUBER_C='3.0')))
for P, A in (('0.04','0.05'), ('0.16','0.025'), ('0.36','0.0167')):   # ④ matched-h (h≈0.227 고정, P/R↑)
    jobs.append((f'Pfir{P}_a{A}', dict(B, RHUKF_FORM='absolute', RHUKF_PINIT=P, RHUKF_ALPHA=A, RHUKF_HUBER_C='3.0')))
mine = jobs[WID::2]; out = {}
print(f'[w{WID}] {len(mine)} runs', flush=True)
for name, env in mine: out[name] = go(name, env)
json.dump(out, open(f'/tmp/v6abserr_w{WID}.json','w'), default=float)
open(f'V6ABSERR_W{WID}_DONE','w').close()
