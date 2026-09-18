#!/usr/bin/env python3
"""night_sweeps_v5 (2026-09-12 사용자 스펙) — v5 surrogate(G2+·클립4.0·failsafe 풀, 단일 공격 가족, 실측 절벽) 위 SWIRL 튜닝. 인자: WID NW
고정: 300스텝·200ep·γ0.85·buffer 50k·n-step 3·decay 4000·Huber·α0.1·[16,16]·seed 42
 S1: P{0.2,0.1,0.05,0.03,0.01} × R{1,2,3} × N{5,6,7} × Q{1e-2,1e-3} (P≤0.05 는 Q1e-2 스킵) = 63 SWIRL + Adam + SGD, TERMINAL_PEN 0
 S2: S1 상위 5 SWIRL × TERMINAL_PEN {0, −5} + Adam × {0, −5} = 12 → 최종 상위 2
결과: night/v5_w{WID}_S{n}.json · 배리어 V5S{n}_w{WID}_DONE"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
ENV0 = {'SURR_V7': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v5.npz'), 'SURR_ATK_PROB': '0.5',
        'ATK_DELTA_LO': '0.15', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_P_UPPER': '0.5', 'ATK_ON_LO': '25', 'ATK_ON_HI': '50', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_LO': '10', 'SURR_DEAD_HI': '20', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.85'}
CLR = ('RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF', 'TERMINAL_PEN')
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1')
ADAM = dict(NET_HIDDEN=16, ADAM_LR='3e-4'); SGD = dict(NET_HIDDEN=16, ADAM_LR='3e-4', OPT='sgd')
def run(name, env, ag, sd, extra=None):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    os.environ['TERMINAL_PEN'] = '0'
    for k, v in (extra or {}).items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW)
    import surrogate_env7 as SE; importlib.reload(SE)
    import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=200, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]
    dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[150:])), e40=float(np.mean(rw[20:40])), e100=float(np.mean(rw[:100])),
             auc100=float(np.sum(rw[:100])), crash=int(sum(r['crashed'] for r in h)), crash_late=int(sum(r['crashed'] for r in h[100:])),
             hist=[dict(ep=r['ep'], reward=r['reward'], f1=r['f1'], crashed=r['crashed'], has_atk=r['has_atk']) for r in h])
    print(f"  {name:36s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} auc100={m['auc100']:.0f} cr={m['crash']}/{m['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return m
def stage(tag, jobs):
    mine = jobs[WID::NW]; out = {}
    try: out = json.load(open(f'{N}/v5_w{WID}_{tag}.json'))   # ★resume: 기존 결과 유지, 미완 런만 실행
    except Exception: out = {}
    mine = [j for j in mine if j[0] not in out]
    print(f'[w{WID}] {tag}: {len(mine)}/{len(jobs)} runs', flush=True)
    for name, env, ag, sd, extra in mine:
        try: out[name] = run(name, env, ag, sd, extra)
        except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {name} 실패: {e}', flush=True)
        json.dump(out, open(f'{N}/v5_w{WID}_{tag}.json', 'w'), default=float)
    json.dump(out, open(f'{N}/v5_w{WID}_{tag}.json', 'w'), default=float); open(f'{N}/V5{tag}_w{WID}_DONE', 'w').close()
    while not all(os.path.exists(f'{N}/V5{tag}_w{w}_DONE') for w in range(NW)): time.sleep(20)
    res = {}
    for w in range(NW):
        try: res.update(json.load(open(f'{N}/v5_w{w}_{tag}.json')))
        except Exception: pass
    return res
J1 = []
for P in ('0.2', '0.1', '0.05', '0.03', '0.01'):
    for R in ('1', '2', '3'):
        for Nn in ('5', '6', '7'):
            for Q in ('1e-2', '1e-3'):
                if Q == '1e-2' and float(P) <= 0.05: continue
                J1.append((f'SWIRL_P{P}_R{R}_N{Nn}_Q{Q}_s42', dict(BASE, RHUKF_PINIT=P, RHUKF_R=R, RHUKF_N=Nn, RHUKF_Q=Q), 'rhukf', 42, {}))
J1 += [('Adam_s42', ADAM, 'adam', 42, {}), ('SGD_s42', SGD, 'adam', 42, {})]
R1 = stage('S1', J1)
sw = sorted([(v['F1'], v['rwd'], k) for k, v in R1.items() if k.startswith('SWIRL')], reverse=True); top = [k for _, _, k in sw[:5]]
if WID == 0: print('[S1 SWIRL 상위 10] ' + ' | '.join(f'{k}({f:.3f},{r:.0f})' for f, r, k in sw[:10]), flush=True)
def envof(name):
    g = lambda key: name.split(f'_{key}')[1].split('_')[0]
    return dict(BASE, RHUKF_PINIT=g('P'), RHUKF_R=g('R'), RHUKF_N=g('N'), RHUKF_Q=g('Q'))
J2 = []
for pen in ('0', '-5'):
    for k in top: J2.append((k.replace('_s42', f'_pen{pen}_s42'), envof(k), 'rhukf', 42, {'TERMINAL_PEN': pen}))
    J2.append((f'Adam_pen{pen}_s42', ADAM, 'adam', 42, {'TERMINAL_PEN': pen}))
R2 = stage('S2', J2)
open(f'{N}/V5SWEEPS_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
