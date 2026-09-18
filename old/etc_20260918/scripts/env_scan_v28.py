#!/usr/bin/env python3
"""env_scan_v28 (09-14 낮, 사용자 스펙) — 자연 라벨잡음 무대(v27 gU 돌풍·uniform 50k) + 무돌풍 대조(nU)
   × {SWa: SWIRL P0.03·R1·N6 (v18 최적), SWb: SWIRL P0.1·R3·N7, Adam 3e-4(He·β3·noAMS) 만, UKF-TD(P 지속·N1), EKF-TD(P 지속·야코비안)}
   × seed 42–44 (시드 3), 150 ep, 에피소드별 greedy 프로브 4. 돌풍 배율 G2.5/V2.0 은 cert_WIND 실측 전 잠정값(v27 과 동일).
   UKF/EKF 구현은 scratchpad/verify_ktd.py 로 검증(야코비안=FD·P 지속·PD·복구0) 후 착수. 인자: WID NW, env JOBSET=gpu|cpu"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
ENV0 = {'SURR_V7': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v5b.npz'), 'SURR_ATK_PROB': '0.5',
        'ATK_DELTA_LO': '0.15', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.85'}
CLR = ('REPLAY_MODE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'PROBE_EVERY', 'PROBE_N', 'RHUKF_MODE', 'SURR_V5', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env7 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=150, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[100:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])), prep=int(h[-1].get('prep', 0)), hist=[dict(r) for r in h])
    print(f"  {name:40s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} prep={m['prep']} ({time.time()-t0:.0f}s)", flush=True)
    return m
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3')
STAGE = {'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5'}
FRAME = dict(STAGE, EPS_HOVER_P='0.1', GAMMA='0.9', EPS_DECAY='4000', HOVER_DWELL='5', SURR_V5='1', PROBE_EVERY='1', PROBE_N='4')
GUST = dict(FRAME, SURR_GUST_P='0.5', SURR_GUST_LO='20', SURR_GUST_HI='50', SURR_GUST_G='2.5', SURR_GUST_V='2.0', SURR_GUST_ATK_AFTER='1')
CELLS = {'gU': dict(GUST, REPLAY_MODE='uniform', BUFFER_SIZE='50000'), 'nU': dict(FRAME, REPLAY_MODE='uniform', BUFFER_SIZE='50000')}
SWa = dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='6')
SWb = dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
JOBSET = os.environ.get('JOBSET', 'gpu')
jobs = []
for cn, cv in CELLS.items():
    ags = [('SWa', SWa, 'rhukf'), ('SWb', SWb, 'rhukf'), ('UKF', dict(KTD, RHUKF_MODE='ukf'), 'rhukf'), ('EKF', dict(KTD, RHUKF_MODE='ekf'), 'rhukf')] if JOBSET == 'gpu' else [('Adam', ADAMP, 'adam')]
    for ag, envv, agt in ags:
        for sd in range(42, 45):
            jobs.append((f'{cn}_{ag}_s{sd}', envv, agt, sd, cv))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v28{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs: {[j[0] for j in mine]}', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v28{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V28{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
