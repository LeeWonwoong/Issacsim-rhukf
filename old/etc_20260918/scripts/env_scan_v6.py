#!/usr/bin/env python3
"""env_scan_v6 (2026-09-12 사용자 스펙) — 무대 난이도 스캔 (surrogate v5 풀): 공격 plateau 공통 {U(10,30), U(15,30)} × 상단 가중 {0.2, 0.3} × P(공격) 0.5
   × {SWIRL top-2 (P0.05·R3·N7, P0.1·R3·N7 · Q1e-3 · α0.1), Adam} × seed {42, 43} = 24 런. 절벽 = 실측 δ·plateau·데드라인 곡선. 벌점 0. 인자: WID NW"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
ENV0 = {'SURR_V7': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v5.npz'), 'SURR_ATK_PROB': '0.5',
        'ATK_DELTA_LO': '0.15', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.85'}
CLR = ('RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3')
AG = {'SW1': (dict(BASE, RHUKF_PINIT='0.05', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'SW2': (dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'Adam': (dict(NET_HIDDEN=16, ADAM_LR='3e-4'), 'adam')}
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env7 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=200, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[150:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])), hist=[dict(ep=r['ep'], reward=r['reward'], f1=r['f1'], crashed=r['crashed'], has_atk=r['has_atk']) for r in h])
    print(f"  {name:40s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return m
jobs = []
for on in (('10', '30'), ('15', '30')):
    for pu in ('0.2', '0.3'):
        for ag in ('SW1', 'SW2', 'Adam'):
            for sd in (42, 43):
                jobs.append((f'on{on[0]}-{on[1]}_pu{pu}_{ag}_s{sd}', AG[ag][0], AG[ag][1], sd, {'ATK_ON_LO': on[0], 'ATK_ON_HI': on[1], 'ATK_P_UPPER': pu}))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v6_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v6_w{WID}.json', 'w'), default=float)
open(f'{N}/V6_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
