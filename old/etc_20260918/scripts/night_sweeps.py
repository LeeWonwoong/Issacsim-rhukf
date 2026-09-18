#!/usr/bin/env python3
"""night_sweeps (2026-09-11 사용자 스펙) — v4 surrogate(G2·arm0.05 풀 + bandcap 밴드) 위 SWIRL 튜닝 스윕. 인자: WID NW
고정: 300스텝·200ep·γ0.85·buffer 50k·n-step 3·decay 4000·terminal 0(절단만)·Huber
 S1 격자: P_init{0.2,0.1,0.05,0.03,0.01} × R{1,2} × N{5,7} × Q{1e-2,1e-3} × α{0.1,0.5} = 80 SWIRL (s42) + Adam s42 + SGD s42
 S2 비율: P(치명) 0.9→0.1 (0.1 간격, 9값) × {S1 상위 2 SWIRL, Adam} s42 = 27
결과: results/claudecodefortest/night/sweep_w{WID}_S{n}.json · 배리어 S{n}_w{WID}_DONE"""
import sys, os, json, time, importlib, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
WID, NW = int(sys.argv[1]), int(sys.argv[2]); N = 'results/claudecodefortest/night'
dec = json.load(open(f'{N}/decision.json'))
ENV0 = {'SURR_V7': '1', 'SURR_V4': '1', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v4.npz'),
        'SURR_WEAK_LO': str(dec['weak_lo']), 'SURR_WEAK_HI': str(dec.get('weak_hi', 0.6)), 'SURR_WEAK_ON_LO': str(dec['weak_on'][0]), 'SURR_WEAK_ON_HI': str(dec['weak_on'][1]),
        'SURR_WEAK_RISE_MAX': '0', 'SURR_LETH_LO': str(dec['leth'][0]), 'SURR_LETH_HI': str(dec['leth'][1]),
        'SURR_LETH_HOLD_LO': str(dec['leth_hold'][0]), 'SURR_LETH_HOLD_HI': str(dec['leth_hold'][1]),
        'SURR_P_LETHAL': str(dec['p_lethal']), 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'BUFFER_SIZE': '50000', 'TERMINAL_PEN': '0', 'GAMMA': '0.85'}
CLR = ('RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI',
       'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1)
ADAM = dict(NET_HIDDEN=16, ADAM_LR='3e-4'); SGD = dict(NET_HIDDEN=16, ADAM_LR='3e-4', OPT='sgd')
def run(name, env, ag, sd, extra=None):
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in (extra or {}).items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW)
    import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=200, seed=sd, ep_steps=300, agent_type=ag)
    s = SR.summarize(h); rw = [r['reward'] for r in h]
    dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    m = dict(F1=float(s['F1']), fpr=float(s['fpr']), delay=float(np.mean(dl)) if dl else -1.0, rwd=float(np.mean(rw[150:])),
             e40=float(np.mean(rw[20:40])), e100=float(np.mean(rw[:100])), crash=int(sum(r['crashed'] for r in h)),
             crash_late=int(sum(r['crashed'] for r in h[100:])), relapse=float(np.mean([r.get('relapse', 0) for r in h[100:] if r['has_atk']] or [0])),
             hist=[dict(ep=r['ep'], reward=r['reward'], f1=r['f1'], crashed=r['crashed'], has_atk=r['has_atk']) for r in h])
    print(f"  {name:34s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} rwd={m['rwd']:.0f} e40={m['e40']:.0f} cr={m['crash']}/{m['crash_late']} ({time.time()-t0:.0f}s)", flush=True)
    return m
def stage(tag, jobs):
    mine = jobs[WID::NW]; out = {}
    print(f'[w{WID}] {tag}: {len(mine)}/{len(jobs)} runs', flush=True)
    for name, env, ag, sd, extra in mine:
        try: out[name] = run(name, env, ag, sd, extra)
        except Exception as e: print(f'  !! {name} 실패: {e}', flush=True)
        json.dump(out, open(f'{N}/sweep_w{WID}_{tag}.json', 'w'), default=float)
    json.dump(out, open(f'{N}/sweep_w{WID}_{tag}.json', 'w'), default=float)
    open(f'{N}/{tag}_w{WID}_DONE', 'w').close()
    while not all(os.path.exists(f'{N}/{tag}_w{w}_DONE') for w in range(NW)): time.sleep(20)
    res = {}
    for w in range(NW):
        try: res.update(json.load(open(f'{N}/sweep_w{w}_{tag}.json')))
        except Exception: pass
    return res
J1 = []
for P in ('0.2', '0.1', '0.05', '0.03', '0.01'):
    for R in ('1', '2'):
        for Nn in ('5', '7'):
            for Q in ('1e-2', '1e-3'):
                for al in ('0.1', '0.5'):
                    J1.append((f'SWIRL_P{P}_R{R}_N{Nn}_Q{Q}_a{al}_s42', dict(BASE, RHUKF_PINIT=P, RHUKF_R=R, RHUKF_N=Nn, RHUKF_Q=Q, RHUKF_ALPHA=al), 'rhukf', 42, {}))
J1 += [('Adam_s42', ADAM, 'adam', 42, {}), ('SGD_s42', SGD, 'adam', 42, {})]
R1 = stage('S1', J1)
sw = sorted([(v['F1'], v['rwd'], k) for k, v in R1.items() if k.startswith('SWIRL')], reverse=True)
top = [k for _, _, k in sw[:2]]
if WID == 0: print('[S1 SWIRL 상위 10] ' + ' | '.join(f'{k}({f:.3f},{r:.0f})' for f, r, k in sw[:10]), flush=True)
def envof(name):
    g = lambda key: name.split(f'_{key}')[1].split('_')[0]
    return dict(BASE, RHUKF_PINIT=g('P'), RHUKF_R=g('R'), RHUKF_N=g('N'), RHUKF_Q=g('Q'), RHUKF_ALPHA=g('a'))
J2 = []
for pl in ('0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'):
    for k in top: J2.append((k.replace('_s42', f'_pl{pl}_s42'), envof(k), 'rhukf', 42, {'SURR_P_LETHAL': pl}))
    J2.append((f'Adam_pl{pl}_s42', ADAM, 'adam', 42, {'SURR_P_LETHAL': pl}))
R2 = stage('S2', J2)
open(f'{N}/SWEEPS_w{WID}_DONE', 'w').close(); print(f'[w{WID}] 완료', flush=True)
