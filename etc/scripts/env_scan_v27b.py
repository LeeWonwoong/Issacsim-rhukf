#!/usr/bin/env python3
"""env_scan_v21b (09-13 저녁) — 확정 세트(D5·decay4000·hover탐험0.1·4항·AMSGrad off) 에서 γ{0.85,0.9,0.95} × SWIRL × 5시드 = 15 (GPU); Adam 짝은 v21c(CPU). 원본 v19 — 5시드 확인: v18 최적 SWIRL vs Adam × {frame, g95d8k, shiftN04, st03w20p30(스텔스 70%), st03w10p30(스텔스 U(.03,.1) 70%), noise1.5} × seed42–46 = 60. 원본 v15 — 약속 hover D∈{5,10} × γ{0.85,0.9,0.95} × ε decay {4000,8000,12000} × 4항 보상 × {SW1,SW4,Adam} × seed{42,43} = 36 런 + Kurt형 ablation(D10·8000·s42) 3 런. hover 탐험 0.05, γ0.95, 200ep, v5b 풀 (사용자 09-13 03:30: D≤10, decay 3종). 원본 v14 — A3 약속 hover(최소 체류 D∈{10,20} 강제, 보상 track +p−c·1{atk}, hover 0, FA=잃은 진행) × {SW1,SW4,Adam} × seed{42,43} + D0 대조, 200ep, v5b 풀, P(atk) 0.5. 원본 v13 —: R0 보상 그대로, P(공격 에피) {0.1, 0.2} × {SW1, SW4, Adam, SGD} × seed {42,43,44}, 200ep, v5b 풀. 지표 = 학습곡선(hist) 로 F1 0.8 도달 에피·auc. 원본 v10:: R0(γ0.85·decay4000) / A2(γ0.95·decay12000·p0.5·c6·c0 0.5·f1·5분할) × {SW1,SW2,SW4,Adam} × seed{42,43}, 250 ep
   + A2_cp0.7(c 0.35) · A2_noinst(1분할) × {SW1, Adam} × seed 42.  plateau U(25,40)·상단 0.5.
   원본: env_scan_v7 — 보상구조 스캔: R0 현행 / R1 희소화(TP·FP·FN ×0.2, TN +0.5 유지 = 임무 진행 보상 우세) / R2 γ0.9 / R12 / R123(+ε decay 12000)
   무대 고정 plateau U(15,30)·상단 0.3·P(공격) 0.5. × {SW1, Adam} × seed {42,43} = 20. 원본: env_scan_v6 (2026-09-12 사용자 스펙) — 무대 난이도 스캔 (surrogate v5 풀): 공격 plateau 공통 {U(10,30), U(15,30)} × 상단 가중 {0.2, 0.3} × P(공격) 0.5
   × {SWIRL top-2 (P0.05·R3·N7, P0.1·R3·N7 · Q1e-3 · α0.1), Adam} × seed {42, 43} = 24 런. 절벽 = 실측 δ·plateau·데드라인 곡선. 벌점 0. 인자: WID NW"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
ENV0 = {'SURR_V7': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v5b.npz' if os.path.exists(f'{N}/train_pool_v5b.npz') else f'{N}/train_pool_v5.npz'), 'SURR_ATK_PROB': '0.5',
        'ATK_DELTA_LO': '0.15', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.85'}
CLR = ('REPLAY_MODE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'PROBE_EVERY', 'PROBE_N', 'RHUKF_MODE', 'SURR_V5', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3')
AG = {'SW1': (dict(BASE, RHUKF_PINIT='0.05', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'SW2': (dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'Adam': (dict(NET_HIDDEN=16, ADAM_LR='3e-4'), 'adam')}
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env7 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=150, seed=sd, ep_steps=300, agent_type=ag, obs_noise=NOISE.get(name.split('_')[0], 0.0))
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[200:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])), hist=[dict(r) for r in h])   # ★전체 hist 저장(프로브·돌풍·필터 진단 포함)
    print(f"  {name:40s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return m
jobs = []
SW = {'SW1': dict(BASE, RHUKF_PINIT='0.05', RHUKF_R='3', RHUKF_N='7'), 'SW2': dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7'), 'SW4': dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='6')}
STAGE = {'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5'}
import glob as _g, re as _re
_t = {}
for _f in _g.glob(f'{N}/v18_w*.json'): _t.update(json.load(open(_f)))
def _key(k):
    h = _t[k]['hist']; f = np.array([x['f1'] for x in h]); ha = np.array([bool(x['has_atk']) for x in h])
    return (float(_t[k]['F1']), float(f[ha][40:80].mean()) if ha.sum() > 80 else 0.0)
best = max(_t, key=_key) if _t else 'tuneP0.05R1N6_SW_s42'
mm = _re.match(r'tuneP([0-9.]+)R([0-9]+)N([0-9]+)', best); SWB = dict(BASE, RHUKF_PINIT=mm.group(1), RHUKF_R=mm.group(2), RHUKF_N=mm.group(3))
open('results/claudecodefortest/SWIRL_BEST', 'w').write(f'{best} P={mm.group(1)} R={mm.group(2)} N={mm.group(3)}\n')
print(f'[v19] SWIRL best {best}', flush=True)
# 다중 버스트 체인 가족 (SURR_V5=0): 약버스트 δU(0.1,0.6) rise U(0,R) hold U(10,40) off U(25,40) 반복 + 치명 30% (δ0.74–0.80 hold 40–60)
CH = dict(EPS_HOVER_P='0.1', GAMMA='0.9', EPS_DECAY='4000', HOVER_DWELL='5', SURR_V5='0', SURR_WEAK_LO='0.1', SURR_WEAK_HI='0.6', SURR_WEAK_ON_LO='10', SURR_WEAK_ON_HI='40', SURR_LETH_LO='0.74', SURR_LETH_HI='0.80')
# v27 (09-14 밤, 자연 감독충돌 축): 돌풍 창(관측만 상승·라벨 clean, P0.5·20–50스텝·G×2.5·V×2.0, 공격은 돌풍 직후 5–20스텝 뒤) × 리플레이 {uniform50k, cer, recency(hl3000), small5k}
#   × {SWIRL, UKF-TD(m), EKF-TD(m), Adam(He·β3·noAMS·3e-4)} × seed42–46, 150ep, 에피소드별 greedy 프로브 4. + 무돌풍 대조(uniform).
# v27b: 드문·긴 난기류 에피소드(P 0.1, 100–200 스텝 = 에피소드 대부분) = 갱신 흐름의 유계 교란(자연 임펄스). 리플레이 {cer, recency} × {SWIRL, UKF-TD, Adam} × 5시드, 150ep
FRAME = dict(STAGE, EPS_HOVER_P='0.1', GAMMA='0.9', EPS_DECAY='4000', HOVER_DWELL='5', SURR_V5='1', PROBE_EVERY='1', PROBE_N='4')
GUST = dict(FRAME, SURR_GUST_P='0.1', SURR_GUST_LO='100', SURR_GUST_HI='200', SURR_GUST_G='2.5', SURR_GUST_V='2.0', SURR_GUST_ATK_AFTER='0')
CELLS = {'rC': dict(GUST, REPLAY_MODE='cer', BUFFER_SIZE='50000'), 'rR': dict(GUST, REPLAY_MODE='recency', REPLAY_HALFLIFE='3000', BUFFER_SIZE='50000'), 'rU': dict(GUST, REPLAY_MODE='uniform', BUFFER_SIZE='50000')}
NOISE = {}
SWB = dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='6')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
JOBSET = os.environ.get('JOBSET', 'gpu')
for cn, cv in CELLS.items():
    ags = [('SW', SWB, 'rhukf'), ('UKFm', dict(KTD, RHUKF_MODE='ukf'), 'rhukf')] if JOBSET == 'gpu' else [('Adam', ADAMP, 'adam')]
    if cn == 'rU' and JOBSET == 'gpu': ags = [('SW', SWB, 'rhukf')]
    for ag, envv, agt in ags:
        for sd in range(42, 47):
            jobs.append((f'{cn}_{ag}_s{sd}', envv, agt, sd, cv))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v27b{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v27b{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V27b{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
