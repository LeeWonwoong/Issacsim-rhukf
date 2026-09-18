#!/usr/bin/env python3
"""env_scan_v34 (09-15 저녁 큐, 사용자 요청: v29 튜닝값의 급변 전이 + Huber R·c 영향) — 무대 = v30 blk50k(v5d, 시드 42–46, v30 과 짝).
   A tune*: v29 SWIRL 상위 10 중 v30 기준(P0.03·R1·N7) 외 9 config · base: SW/UKF/EKF 기준 재실행(혁신 로그 포함, v30 재현 확인)
   B hubc{1.5,6,off}·r{0.3,3}: SWIRL Huber c 와 기본 R (나머지 P0.03·R1·N7) · C 공정성: UKF-TD c{off,1.5}·EKF-TD c off·Adam Huber β{1, MSE}
   hist 에 에피소드별 필터 혁신 innov/innov_max·Huber 팽창 adapt·갱신 수 n_upd 기록(09-15 추가) → 갱신 인덱스 단위 급변 가시화(리플레이 방어).
   원 v33 설명: — ① 보상 패널티 크기 k(FP·FN 항 ×k, TP/TN 고정) 가 학습기 격차를 키우나
   ② 급변을 '특정 구간 블록' 이 아니라 '환경 기본값(마르코프 날씨 체제 전환)' 으로 두면 격차가 남나.
   무대·풀·학습기 = v30 과 동일(v5d, SWIRL P0.03·R1·N7, UKF/EKF-TD P₀0.03, Adam 3e-4) → k=1 대조는 v30 blk50k/mix50k 시드 42–46.
   셀: blk50k_k2 · blk50k_k4 · mix50k_k4 · mkv50k(k=1, 시드별 체제 스케줄: 잔잔 ws0/7=.6/.4 ↔ 폭풍 ws10, 체류 U(20,40) ep, 잔잔 시작)
   × {SW, UKF, EKF}(GPU) · Adam(CPU) × 시드 42–46, 150 ep, 프로브 4/에피. 인자: WID NW, env JOBSET=gpu|cpu"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
POOL = f'{N}/train_pool_v5d.npz'
ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(POOL), 'SURR_ATK_PROB': '0.5',
        'SURR_PLOC_TIERS': '7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1', 'SURR_DEAD_TIERS': '10:0.78,26,0.80,16,0.84,4',
        'ATK_DELTA_LO': '0.05', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'TERMINAL_PEN': '0', 'GAMMA': '0.9',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '5', 'PROBE_EVERY': '1', 'PROBE_N': '4', 'REPLAY_MODE': 'uniform'}
CLR = ('OBS_MODE', 'SURR_TIER_P', 'SURR_TIER_SCHED', 'SURR_FAM_SCHED', 'BUFFER_SIZE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'RHUKF_MODE', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_TN', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env8 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=150, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[110:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[110:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[110:])), prep=int(h[-1].get('prep', 0)), extra=dict(extra), hist=[dict(r) for r in h])
    print(f"  {name:26s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.1f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} prep={m['prep']} ({time.time()-t0:.0f}s)", flush=True)
    return m
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3')
SWB = dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='7')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
BLOCK = '0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4'
BLK = dict(SURR_TIER_SCHED=BLOCK, BUFFER_SIZE='50000')
SWc = lambda **kw: dict(SWB, **{k: str(v) for k, v in kw.items()})
KTc = lambda md, **kw: dict(KTD, RHUKF_MODE=md, **{k: str(v) for k, v in kw.items()})
TOP = [('0.01', '1', '6'), ('0.05', '3', '7'), ('0.01', '1', '5'), ('0.03', '1', '6'), ('0.1', '1', '6'), ('0.01', '1', '7'), ('0.05', '1', '7'), ('0.1', '3', '7'), ('0.01', '3', '7')]
GPU_CELLS = [('base', 'SW', SWB), ('base', 'UKF', KTc('ukf')), ('base', 'EKF', KTc('ekf'))]
GPU_CELLS += [(f'tuneP{p}R{r}N{n}', 'SW', SWc(RHUKF_PINIT=p, RHUKF_R=r, RHUKF_N=n)) for p, r, n in TOP]
GPU_CELLS += [('hubc1.5', 'SW', SWc(RHUKF_HUBER_C=1.5)), ('hubc6', 'SW', SWc(RHUKF_HUBER_C=6)), ('hubcoff', 'SW', SWc(RHUKF_HUBER_C=1e9)),
              ('r0.3', 'SW', SWc(RHUKF_R=0.3)), ('r3', 'SW', SWc(RHUKF_R=3))]
GPU_CELLS += [('kthubcoff', 'UKF', KTc('ukf', RHUKF_HUBER_C=1e9)), ('kthubc1.5', 'UKF', KTc('ukf', RHUKF_HUBER_C=1.5)), ('kthubcoff', 'EKF', KTc('ekf', RHUKF_HUBER_C=1e9))]
CPU_CELLS = [('adamb1', 'Adam', dict(ADAMP, ADAM_HUBER_BETA='1')), ('adambmse', 'Adam', dict(ADAMP, ADAM_HUBER_BETA='1e9'))]
JOBSET = os.environ.get('JOBSET', 'gpu'); SEEDS = range(42, 47)
jobs = []
for sd in SEEDS:
    for cn, ag, envv in (GPU_CELLS if JOBSET == 'gpu' else CPU_CELLS):
        jobs.append((f'{cn}_{ag}_s{sd}', envv, 'adam' if ag == 'Adam' else 'rhukf', sd, BLK))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v34{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs: {[j[0] for j in mine]}', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v34{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V34{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
