#!/usr/bin/env python3
"""env_scan_v29 (09-14 오후, 사용자 스펙: 바람 무대가 바뀌었으니 SWIRL 재튜닝) — 새 무대 = v5c 풀(COM 편향 재생·티어 분리·스텔스 실측 빈) + env8
   (에피소드별 바람 티어 추첨 SURR_TIER_P, 돌풍 배율 off) + 가족 하한 ATK_DELTA_LO(기본 0.05: cert_S 애매 대역).
   GPU: SWIRL 격자 P₀{0.01,0.03,0.05,0.1} × R{1,3} × N{5,6,7} (Q1e-3·α0.1·SPAS·Huber3 고정) + UKF-TD/EKF-TD × P₀{0.03,0.1} → (24+4) × 시드 3 = 84
   CPU: Adam 3e-4(He·β3·noAMS) × 시드 3.  150 ep, greedy 프로브 4/에피. 인자: WID NW, env JOBSET=gpu|cpu, TIER_P, DELTA_LO, POOL"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
TIER_P = os.environ.get('TIER_P', '0:0.4,7:0.3,10:0.3'); DELTA_LO = os.environ.get('DELTA_LO', '0.05'); POOL = os.environ.get('POOL', f'{N}/train_pool_v5c.npz')
ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(POOL), 'SURR_ATK_PROB': '0.5', 'SURR_TIER_P': TIER_P, 'SURR_PLOC_TIERS': os.environ.get('SURR_PLOC_TIERS', ''),
        'ATK_DELTA_LO': DELTA_LO, 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.9',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '5', 'PROBE_EVERY': '1', 'PROBE_N': '4', 'REPLAY_MODE': 'uniform'}
CLR = ('REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'RHUKF_MODE', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
def run(name, env, ag, sd):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    import env.reward as RW; importlib.reload(RW); import surrogate_env8 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=150, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[100:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])), prep=int(h[-1].get('prep', 0)), hist=[dict(r) for r in h])
    print(f"  {name:32s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} prep={m['prep']} ({time.time()-t0:.0f}s)", flush=True)
    return m
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
JOBSET = os.environ.get('JOBSET', 'gpu'); SEEDS = range(42, 45)
jobs = []
if JOBSET == 'gpu':
    for P in ('0.01', '0.03', '0.05', '0.1'):
        for R in ('1', '3'):
            for Nh in ('5', '6', '7'):
                for sd in SEEDS: jobs.append((f'SW_P{P}R{R}N{Nh}_s{sd}', dict(BASE, RHUKF_PINIT=P, RHUKF_R=R, RHUKF_N=Nh), 'rhukf', sd))
    for md in ('ukf', 'ekf'):
        for P in ('0.03', '0.1'):
            for sd in SEEDS: jobs.append((f'{md.upper()}_P{P}_s{sd}', dict(KTD, RHUKF_MODE=md, RHUKF_PINIT=P), 'rhukf', sd))
else:
    for sd in SEEDS: jobs.append((f'Adam_s{sd}', ADAMP, 'adam', sd))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v29{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs (TIER_P={TIER_P} DELTA_LO={DELTA_LO} POOL={POOL}): {[j[0] for j in mine]}', flush=True)
for name, env, ag, sd in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v29{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V29{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
