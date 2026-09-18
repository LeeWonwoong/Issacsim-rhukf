#!/usr/bin/env python3
"""env_scan_v35 (09-15 밤 큐, 무대 탐색 — STAGE_SEARCH_0915) — 하한 ATK_DELTA_LO 0.10(15:45 확정) 새 무대의 SWIRL 우위 탐색. v5d 풀, 시드 42–46, hist 전체 저장.
   ★ v34(하한 0.05, blk50k γ0.9 튜닝 전이·Huber c/R·공정성) 를 **대체**한다: v34 base/tune* 은 옛 하한 무대라 폐기, 튜닝 전이는 아래 T 셀로 흡수.
     v34 의 Huber c{1.5,6,off}·R{0.3,3}·KTD c·Adam β 공정성 셀은 v35 로 무대가 정해진 뒤 하한 0.10 으로 v36 에서 재편성(병합 안 함).
   무대(우선순위 순 — 워커가 jobs[WID::NW] 를 앞에서부터 돈다. 자동 절단은 없다 — 시간이 모자라면 T 셀이 마지막에 돌므로 사람이 판단해 멈춘다):
     P1 blk50k_g90 : v30 blk50k 재기준(하한 0.10). SURR_TIER_SCHED 잔잔 ws0/7 .6/.4 → ep60–109 ws10 100 % → 잔잔, 버퍼 50k, γ0.9, ε decay 4000, 150 ep
     P2 blk50k_g95 : 같은 블록, γ0.95(v21bc: γ↑ 이면 SW−Adam fpr·auc 격차 단조 증가, 교호 t 2.9/4.7; decay 4000 그대로 → 진입 시 ε≈0.01 유지)
     P3 cold30_w10h: 콜드스타트(FIR (d) 초기조건 오지정) — ε decay 500, ws10 비중 0.5(SURR_TIER_P 0:.25,7:.25,10:.5), 버퍼 50k, γ0.9, 30 ep
                     (v26 d500: SW 가 Adam lr 3e-4/1e-3/3e-3 전부와 UKF/EKF-TD 를 이김; v31 ws10h: 초기 프로브 격차 최대)
     T  blk50k_g95 SWIRL 무대 맞춤 튜닝: P₀0.1(신뢰영역↑ → 재수렴 가속 가설), N5(창 짧게 → 망각 가속 가설). 나머지 R1·Q1e-3·c3·τ.005
   학습기: GPU = SW(P₀.03·R1·N7) · UKF-TD · EKF-TD(P 지속 N1, P₀.03·Q1e-3 = 시리즈 기본값) [+ P3 에 SWp1(P₀.1)] ; CPU = Adam lr {3e-4(기본), 1e-3, 3e-3}(He·β3·AMSGrad off)
   런타임 가정(v33 10 워커 실측): SW 150 ep ≈ 4250 s, KTD ≈ 2650 s, Adam CPU ≈ 43 s → GPU 60 런 ≈ 워커당 4.0 h, CPU 45 런 ≈ 20 분.
   사전등록 1차 지표: P1/P2 = 블록 FP(hist.fpr ep60–109) · 진입 FP(60–69) SW−Adam / SW−UKF / SW−EKF 짝, P2−P1 교호(γ); P3 = probe_f1 ep0–14 평균·reward ep0–29 합(s43 프로브 제외).
   보류(추가 구현 필요, 이 스캔에 없음): 감독 채널 일시 고장 TRAIN_CORRUPT_MODE/L0/L1(rl/memory.py sample_batch + surrogate_run push r_alt·갱신 단위 프로브),
     KTD η·P 진화잡음(문헌 KTD 기본), P 리셋 주기 사다리, 체제열 RNG·시나리오 seed+ep 패치, 온/오프 가능한 SURR_SHIFT_END.
   인자: WID NW, env JOBSET=gpu|cpu, V35_COUNT=1 이면 잡 수만 출력하고 종료."""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
POOL = f'{N}/train_pool_v5d.npz'
ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(POOL), 'SURR_ATK_PROB': '0.5',
        'SURR_PLOC_TIERS': '7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1', 'SURR_DEAD_TIERS': '10:0.78,26,0.80,16,0.84,4',
        'ATK_DELTA_LO': '0.10', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'TERMINAL_PEN': '0', 'GAMMA': '0.9',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '5', 'PROBE_EVERY': '1', 'PROBE_N': '4', 'REPLAY_MODE': 'uniform'}
CLR = ('OBS_MODE', 'SURR_TIER_P', 'SURR_TIER_SCHED', 'SURR_FAM_SCHED', 'BUFFER_SIZE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'RHUKF_MODE', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD', 'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_TN', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
def run(name, env, ag, sd, extra, n_ep):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env8 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=n_ep, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; L0 = int(round(n_ep * 0.73))   # 150 ep → 110 (v30–v34 후기 창과 동일), 30 ep → 22
    dl = [r['delay'] for r in h[L0:] if r['has_atk'] and not np.isnan(r['delay'])]
    fp = [r['fpr'] for r in h]; pf = [r.get('probe_f1', np.nan) for r in h]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[L0:])), e40=float(np.mean(rw[20:40])) if n_ep >= 40 else float('nan'),
             auc100=float(np.sum(rw[:100])), auc_all=float(np.sum(rw)), probe0_14=float(np.nanmean(pf[:15])),
             FPblk=float(np.mean(fp[60:110])) if n_ep >= 110 else float('nan'), FPenter=float(np.mean(fp[60:70])) if n_ep >= 70 else float('nan'),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[L0:])), prep=int(h[-1].get('prep', 0)), n_ep=n_ep, extra=dict(extra), hist=[dict(r) for r in h])
    print(f"  {name:30s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.1f} auc={m['auc_all']:.0f} p0_14={m['probe0_14']:.3f} FPblk={m['FPblk']:.4f} FPent={m['FPenter']:.4f} cr={m['crash']}/{m['crash_late']} prep={m['prep']} ({time.time()-t0:.0f}s)", flush=True)
    return m
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3')
SWB = dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='7')
SWc = lambda **kw: dict(SWB, **{k: str(v) for k, v in kw.items()})
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
BLOCK = '0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4'; W10H = '0:0.25,7:0.25,10:0.5'
STAGES = {'blk50k_g90':  (150, dict(SURR_TIER_SCHED=BLOCK, BUFFER_SIZE='50000', GAMMA='0.9')),
          'blk50k_g95':  (150, dict(SURR_TIER_SCHED=BLOCK, BUFFER_SIZE='50000', GAMMA='0.95')),
          'cold30_w10h': (30,  dict(SURR_TIER_P=W10H, BUFFER_SIZE='50000', GAMMA='0.9', EPS_DECAY='500'))}
KT = [('UKF', dict(KTD, RHUKF_MODE='ukf')), ('EKF', dict(KTD, RHUKF_MODE='ekf'))]
GPU_PLAN = [('blk50k_g90',  [('SW', SWB)] + KT),                                                    # P1
            ('blk50k_g95',  [('SW', SWB)] + KT),                                                    # P2
            ('cold30_w10h', [('SW', SWB), ('SWp1', SWc(RHUKF_PINIT='0.1'))] + KT),                  # P3
            ('blk50k_g95',  [('SWtP0.1', SWc(RHUKF_PINIT='0.1')), ('SWtN5', SWc(RHUKF_N='5'))])]    # T
CPU_PLAN = [(st, [('Adam3e-4', ADAMP), ('Adam1e-3', dict(ADAMP, ADAM_LR='1e-3')), ('Adam3e-3', dict(ADAMP, ADAM_LR='3e-3'))]) for st in STAGES]
JOBSET = os.environ.get('JOBSET', 'gpu'); SEEDS = range(42, 47)
def build(js):
    jobs = []
    for st, ags in (GPU_PLAN if js == 'gpu' else CPU_PLAN):
        n_ep, sx = STAGES[st]
        for sd in SEEDS:
            for ag, envv in ags:
                jobs.append((f'{st}_{ag}_s{sd}', envv, 'adam' if ag.startswith('Adam') else 'rhukf', sd, sx, n_ep))
    return jobs
if os.environ.get('V35_COUNT') == '1':
    for js in ('gpu', 'cpu'):
        jb = build(js); names = [j[0] for j in jb]; assert len(set(names)) == len(names), 'job name 중복'
        cost = sum((4250 if j[2] == 'rhukf' and j[0].split('_')[2].startswith('SW') else 2650 if j[2] == 'rhukf' else 43) * j[5] / 150 * (0.89 if 'SWtN5' in j[0] else 1) for j in jb)
        print(f'JOBSET {js}: {len(jb)} jobs, 추정 {cost/3600:.1f} 워커·h', {st: sum(1 for j in jb if j[0].startswith(st)) for st in STAGES})
    sys.exit(0)
jobs = build(JOBSET)
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v35{JOBSET}_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs: {[j[0] for j in mine]}', flush=True)
for name, env, ag, sd, extra, n_ep in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra, n_ep)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v35{JOBSET}_w{WID}.json', 'w'), default=float)
open(f'{N}/V35{JOBSET}_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
