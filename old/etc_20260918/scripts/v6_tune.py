#!/usr/bin/env python3
# v6_tune — v3.1 surrogate 에서 SWIRL 재튜닝 (2026-09-07 사용자 지정 그리드)
#   stage1: α{0.1,0.5,0.9} × R{1,2} × P{0.01,0.03,0.05,0.07,0.1} = 30 (Q1e-3·abs·SPAS·N5·b128)
#   stage2: 상위 10 × {N6 seed42, N5 seed43}
#   인자: worker_id(0|1) stage(1|2)
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V6'] = '1'
from surrogate_run import run_config, summarize
WID = int(sys.argv[1]); STAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ON, OFF = (10, 21), (25, 41)
def go(name, env, n_ep=160, seed=42):
    for k in ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_SPAS','RHUKF_N'):
        os.environ.pop(k, None)
    t0 = time.time()
    h = run_config(env, n_ep=n_ep, seed=seed, ep_steps=400, on_range=ON, off_range=OFF)
    s = summarize(h); s['crash'] = int(sum(r['crashed'] for r in h)); s['crash_late'] = int(sum(r['crashed'] for r in h[80:]))
    print(f"  {name:34s} F1={s['F1']:.3f} R={s['R']:.3f} fpr={s['fpr']:.3f} rwd={s['reward']:.0f} "
          f"crash={s['crash']}/{s['crash_late']}후반 ({time.time()-t0:.0f}s)", flush=True)
    return s
BASE = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
            RHUKF_FORM='absolute', RHUKF_SPAS=1)
if STAGE == 1:
    CFG = []
    for a in ('0.1','0.5','0.9'):
        for R in ('1','2'):
            for P in ('0.01','0.03','0.05','0.07','0.1'):
                CFG.append((f'a{a}_R{R}_P{P}', dict(BASE, RHUKF_ALPHA=a, RHUKF_R=R, RHUKF_PINIT=P)))
    mine = CFG[WID::2]; out = {}
    print(f'[w{WID}] stage1 {len(mine)} configs', flush=True)
    for name, env in mine: out[name] = go(name, env)
    json.dump(out, open(f'/tmp/v6tune_s1_w{WID}.json','w'), default=float)
    open(f'V6TUNE_S1_W{WID}_DONE','w').close()
else:
    res = {}
    for w in (0,1): res.update(json.load(open(f'/tmp/v6tune_s1_w{w}.json')))
    rank = sorted(res.items(), key=lambda kv: -kv[1]['F1'])[:10]
    if WID == 0:
        print('[stage1 상위10]', flush=True)
        for n,s in rank: print(f"  {n:22s} F1={s['F1']:.3f} crash={s['crash']}", flush=True)
    jobs = []
    for n,_ in rank:
        a = n.split('_')[0][1:]; R = n.split('_')[1][1:]; P = n.split('_')[2][1:]
        e = dict(BASE, RHUKF_ALPHA=a, RHUKF_R=R, RHUKF_PINIT=P)
        jobs.append((f'{n}_N6', dict(e, RHUKF_N=6), 42))
        jobs.append((f'{n}_s43', e, 43))
    mine = jobs[WID::2]; out = {}
    print(f'[w{WID}] stage2 {len(mine)} runs', flush=True)
    for name, env, sd in mine: out[name] = go(name, env, seed=sd)
    json.dump(out, open(f'/tmp/v6tune_s2_w{WID}.json','w'), default=float)
    open(f'V6TUNE_S2_W{WID}_DONE','w').close()
