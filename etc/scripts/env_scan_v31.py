#!/usr/bin/env python3
"""env_scan_v31 (09-14 밤) — aliasing 다이얼 + ablation (v30 뒤): mix 무대에서 (a) g4 gyro-only 8D (b) ws10 비중 0.5 (c) 스텔스 off(하한 0.15)
   앞뒤는 잔잔 혼합(ws0 0.6 / ws7 0.4). 티어는 관측에 없으므로 겹침 영역(gyro 1.2–1.9)에서 P(공격|관측)이 체제마다 뒤집힘
   = 부분관측이 만드는 자연 reward-타깃 반전(라벨은 clean). 대조 = 정상 혼합(v29 무대).
   셀: {block, mix} × 버퍼 {50k, 10k(희석 완화)} × {SWIRL 최적(v29 자동 선택), Adam 3e-4, UKF-TD, EKF-TD} × 시드 42–46, 150 ep, 프로브 4/에피.
   지표(집계): 블록 진입(ep60)·이탈(ep110) 후 greedy F1 재수렴 에피 수, 블록 내/외 F1·FP, reward. 인자: WID NW, env JOBSET, SW_CFG(P,R,N) 덮어쓰기 가능."""
import sys, os, json, time, glob, re, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
POOL = os.environ.get('POOL', f'{N}/train_pool_v5d.npz' if os.path.exists(f'{N}/train_pool_v5d.npz') else f'{N}/train_pool_v5c.npz'); DELTA_LO = os.environ.get('DELTA_LO', '0.05')   # v5d = v5c + cert_SW(바람 속 약공격·스텔스) + cert_DEAD
PLOC = os.environ.get('SURR_PLOC_TIERS', '7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1')
DEADT = os.environ.get('SURR_DEAD_TIERS', '10:0.78,26,0.80,16,0.84,4')   # cert_DEAD 09-14: ws10 δ0.84 d5 생존 60% → 데드라인 ~4 스텝 (ws7 은 무풍 곡선과 동일)
ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(POOL), 'SURR_ATK_PROB': '0.5', 'SURR_PLOC_TIERS': PLOC, 'SURR_DEAD_TIERS': DEADT,
        'ATK_DELTA_LO': DELTA_LO, 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'TERMINAL_PEN': '0', 'GAMMA': '0.9',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '5', 'PROBE_EVERY': '1', 'PROBE_N': '4', 'REPLAY_MODE': 'uniform'}
CLR = ('OBS_MODE', 'SURR_TIER_P', 'SURR_TIER_SCHED', 'SURR_FAM_SCHED', 'BUFFER_SIZE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'RHUKF_MODE', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env8 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, obs_mode=os.environ.get('OBS_MODE', 'raw4'), n_ep=150, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[110:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[110:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[110:])), prep=int(h[-1].get('prep', 0)), hist=[dict(r) for r in h])
    print(f"  {name:28s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} prep={m['prep']} ({time.time()-t0:.0f}s)", flush=True)
    return m
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
# ── SWIRL 최적 config: v29 격자에서 시드 평균 F1 최대(동률 시 auc100) 자동 선택. SW_CFG="P,R,N" 로 덮어쓰기 가능.
def pick_best():
    t = {}
    for f in glob.glob(f'{N}/v29gpu_w*.json'): t.update(json.load(open(f)))
    agg = {}
    for k, m in t.items():
        mm = re.match(r'SW_P([0-9.]+)R([0-9]+)N([0-9]+)_s(\d+)', k)
        if mm: agg.setdefault(mm.group(1, 2, 3), []).append((m['F1'], m['auc100']))
    if not agg: return ('0.03', '1', '6'), 'fallback(v29 없음)'
    best = max(agg, key=lambda c: (np.mean([x[0] for x in agg[c]]), np.mean([x[1] for x in agg[c]])))
    return best, f'v29 시드{len(agg[best])} F1 {np.mean([x[0] for x in agg[best]]):.3f}'
if os.environ.get('SW_CFG'): P, R, Nh = os.environ['SW_CFG'].split(','); src = 'env SW_CFG'
else: (P, R, Nh), src = pick_best()
SWB = dict(BASE, RHUKF_PINIT=P, RHUKF_R=R, RHUKF_N=Nh)
if WID == 0: open(f'{N}/V31_SWIRL_CFG', 'w').write(f'P={P} R={R} N={Nh} ({src})\n')
print(f'[v31] SWIRL cfg P={P} R={R} N={Nh} ({src})', flush=True)
MIX = '0:0.4,7:0.3,10:0.3'
CELLS = {'g4mix': dict(SURR_TIER_P=MIX, BUFFER_SIZE='50000', OBS_MODE='g4'),                       # (a) gyro-only 8D
         'ws10h': dict(SURR_TIER_P='0:0.2,7:0.3,10:0.5', BUFFER_SIZE='50000'),                     # (b) ws10 비중 0.5
         'nost': dict(SURR_TIER_P=MIX, BUFFER_SIZE='50000', ATK_DELTA_LO='0.15'),                  # (c) 스텔스 off
         'pa0.2': dict(SURR_TIER_P=MIX, BUFFER_SIZE='50000', SURR_ATK_PROB='0.2'),                 # (d) 희소성 P(공격)=0.2 (연구목표 ②; 구 v13 은 쉬운 무대라 무격차)
         'fam6': dict(SURR_TIER_P=MIX, BUFFER_SIZE='50000', ATK_WEAK_HI='0.5', ATK_SPLIT='0.80', ATK_P_UPPER='0.25')}   # (e) 가족 v6 = .75·U(.05,.5)+.25·U(.80,.84) (cert_MID 함의)

JOBSET = os.environ.get('JOBSET', 'gpu'); SEEDS = range(42, 47)
jobs = []
for cn, cv in CELLS.items():
    ags = [('SW', SWB, 'rhukf'), ('UKF', dict(KTD, RHUKF_MODE='ukf'), 'rhukf'), ('EKF', dict(KTD, RHUKF_MODE='ekf'), 'rhukf')] if JOBSET == 'gpu' else [('Adam', ADAMP, 'adam')]
    for ag, envv, agt in ags:
        for sd in SEEDS: jobs.append((f'{cn}_{ag}_s{sd}', envv, agt, sd, cv))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v31{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs: {[j[0] for j in mine]}', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v31{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V31{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
