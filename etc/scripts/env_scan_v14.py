#!/usr/bin/env python3
"""env_scan_v14 (09-13) — A3 약속 hover(최소 체류 D∈{10,20} 강제, 보상 track +p−c·1{atk}, hover 0, FA=잃은 진행) × {SW1,SW4,Adam} × seed{42,43} + D0 대조, 200ep, v5b 풀, P(atk) 0.5. 원본 v13 —: R0 보상 그대로, P(공격 에피) {0.1, 0.2} × {SW1, SW4, Adam, SGD} × seed {42,43,44}, 200ep, v5b 풀. 지표 = 학습곡선(hist) 로 F1 0.8 도달 에피·auc. 원본 v10:: R0(γ0.85·decay4000) / A2(γ0.95·decay12000·p0.5·c6·c0 0.5·f1·5분할) × {SW1,SW2,SW4,Adam} × seed{42,43}, 250 ep
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
CLR = ('REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_FP', 'FN_BASE', 'FN_PER', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3')
AG = {'SW1': (dict(BASE, RHUKF_PINIT='0.05', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'SW2': (dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7'), 'rhukf'), 'Adam': (dict(NET_HIDDEN=16, ADAM_LR='3e-4'), 'adam')}
def run(name, env, ag, sd, extra):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env7 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=200, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]; dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[200:])), e40=float(np.mean(rw[20:40])), auc100=float(np.sum(rw[:100])),
             crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])), hist=[dict(ep=r['ep'], reward=r['reward'], f1=r['f1'], crashed=r['crashed'], has_atk=r['has_atk']) for r in h])
    print(f"  {name:40s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return m
jobs = []
SW = {'SW1': dict(BASE, RHUKF_PINIT='0.05', RHUKF_R='3', RHUKF_N='7'), 'SW2': dict(BASE, RHUKF_PINIT='0.1', RHUKF_R='3', RHUKF_N='7'), 'SW4': dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='6')}
STAGE = {'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5'}
A3 = dict(STAGE, REWARD_MODE='A3', A3_P='0.5', A3_C='2.0', GAMMA='0.95', EPS_DECAY='12000')
for D in (10, 20):
    for ag in ('SW1', 'SW4', 'Adam'):
        for sd in (42, 43):
            envv, agt = (SW[ag], 'rhukf') if ag in SW else (AG['Adam'][0], 'adam')
            jobs.append((f'A3d{D}_{ag}_s{sd}', envv, agt, sd, dict(A3, HOVER_DWELL=str(D))))
for ag in ('SW1', 'SW4', 'Adam'):
    envv, agt = (SW[ag], 'rhukf') if ag in SW else (AG['Adam'][0], 'adam')
    jobs.append((f'A3d0_{ag}_s42', envv, agt, 42, dict(A3, HOVER_DWELL='0')))
mine = jobs[WID::NW]; out = {}
try: out = json.load(open(f'{N}/v14_w{WID}.json'))
except Exception: out = {}
print(f'[w{WID}] {len(mine)}/{len(jobs)} runs', flush=True)
for name, env, ag, sd, extra in mine:
    if name in out: continue
    try: out[name] = run(name, env, ag, sd, extra)
    except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
    json.dump(out, open(f'{N}/v14_w{WID}.json', 'w'), default=float)
open(f'{N}/V14_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
