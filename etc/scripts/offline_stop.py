#!/usr/bin/env python3
"""offline_stop — 오프라인 최적정지 (A안: 조건 고정, 버퍼 50k = 전체 데이터셋).

구조(옛 MATLAB ver2.6 재현 + Isaac 데이터):
  행동  0=계속(track)  1=정지(착륙)  → 정지하면 에피소드 즉시 종료
  보상  c=0.2 기준 옛 비율
        평시+계속 +2c=+0.4 │ 평시+정지 −10c=−2.0(종료)
        공격+계속 −5c=−1.0 │ 공격+정지 +15c=+3.0(종료)
  γ=0.9 · 버퍼 50,000(=전체) · 배치 128 · 워밍업 없음 · 관측 8D(action 제외)
  전 학습기가 같은 풀·같은 에피소드 제시 순서를 본다(시드 고정).

주지표 = 임계 성능 도달까지 소비한 전이 수. 벽시계는 병기.
"""
import os, sys, time, json
import numpy as np

R = '/home/acsl/projects/Issacsim-rhukf'
sys.path.insert(0, R); sys.path.insert(0, f'{R}/etc/scripts')
DATA = os.environ.get('STOP_DATA', f'{R}/results/claudecodefortest/night/stop_dataset_50k.npz')
C = float(os.environ.get('STOP_C', '0.2'))
R_CONT_CLEAN, R_STOP_CLEAN = 2*C, -10*C      # +0.4 / -2.0
R_CONT_ATK,   R_STOP_ATK   = -5*C, 15*C      # -1.0 / +3.0
N_EP = int(os.environ.get('STOP_EPISODES', '500'))
EPS_HI, EPS_LO = 0.95, 0.01     # 옛 코드(ver2.6 main_all.m:52) 그대로
EPS_FRAC = float(os.environ.get('STOP_EPS_FRAC', '0.8'))   # maxEpisodes*0.8 에 걸쳐 감쇠

def eps_at(ep, n_ep):
    """★ 에피소드 기준 지수 감쇠. 정지=종료 구조에서는 스텝 기준 감쇠가 작동하지 않는다
       (탐험으로 즉시 착륙 → 스텝이 안 쌓임 → ε 가 안 내려감 → 악순환).
       옛 MATLAB: eps_decay_rate = -log(epsLo/epsHi)/(maxEpisodes*0.8)"""
    rate = -np.log(EPS_LO / EPS_HI) / max(1.0, n_ep * EPS_FRAC)
    return EPS_LO + (EPS_HI - EPS_LO) * float(np.exp(-rate * ep))
SEED = int(os.environ.get('STOP_SEED', '42'))

P_STOP  = float(os.environ.get('STOP_PSTOP', '0.5'))   # 탐험 시 stop 을 고를 확률
L_WIN   = int(os.environ.get('STOP_L', '6'))           # ★final7 과 동일 — 2번/3번 비교 가능성의 전제
DIM_S   = L_WIN * 2
OBS_DIV = float(os.environ.get('STOP_OBS_DIV', '1.0') or 1.0)   # ★[0,1] 유계화 = 4.0

def make_agent(name):
    os.environ.update({'SURROGATE': '1', 'NET_HIDDEN': '16', 'BATCH_SIZE': '128', 'GAMMA': '0.9',
                       'BUFFER_SIZE': '50000', 'NSTEP_OFF': '1', 'EPS_DECAY': '4000'})
    base = dict(RHUKF_TAU='0.005', RHUKF_UI='1', RHUKF_FORM='absolute', RHUKF_SPAS='1',
                RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03', RHUKF_R='1')
    for k in ('RHUKF_MODE', 'ADAM_AMSGRAD', 'OPT'): os.environ.pop(k, None)
    if name.startswith('Adam'):
        lr = name[4:].replace('ams', '')
        os.environ.update(ADAM_LR=lr, ADAM_AMSGRAD='1' if name.endswith('ams') else '0',
                          ADAM_INIT='he', ADAM_HUBER_BETA='3')
        kind = 'adam'
    else:
        os.environ.update(base)
        if name == 'SWIRL':  os.environ['RHUKF_N'] = '7'
        elif name == 'UKF-TD': os.environ['RHUKF_N'] = '1'
        elif name == 'EKF-TD': os.environ.update(RHUKF_N='1', RHUKF_MODE='ekf')
        for _e, _k in (('SW_P', 'RHUKF_PINIT'), ('SW_R', 'RHUKF_R'), ('SW_N_OVR', 'RHUKF_N'), ('SW_A', 'RHUKF_ALPHA')):
            if os.environ.get(_e): os.environ[_k] = os.environ[_e]   # ★09-17 하드코딩 뒤로 이동(N 덮어쓰기 버그 수정)
        kind = 'rhukf'
    import importlib, swrl_config; importlib.reload(swrl_config)
    cfg = swrl_config.Config(); cfg.dimS = DIM_S; cfg.window_size = L_WIN; cfg.obs_scale = [1.0]*DIM_S
    cfg.eps_action_probs = [1.0 - P_STOP, P_STOP]
    if kind == 'adam':
        from rl.agent_adam import OnlineAdamAgent as AG
    else:
        from rl.agent import OnlineRHUKFAgent as AG
    import importlib as il
    return AG(cfg)

def run(name, D, order):
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    ag = make_agent(name)
    obs, atk = D['obs'] / OBS_DIV, D['atk']
    st, ln = D['ep_start'], D['ep_len']
    hist = []; n_trans = 0; t0 = time.time()
    for ei, ep in enumerate(order):
        s0, L = int(st[ep]), int(ln[ep])
        eps_ep = eps_at(ei, len(order))
        prev_s = None; prev_a = None; epr = 0.0; stopped = -1; _ls = []; _qs = []
        true_on = int(np.argmax(atk[s0:s0+L])) if atk[s0:s0+L].any() else -1
        for i in range(L):
            s = obs[s0+i]
            a_now = int(atk[s0+i])
            if prev_s is not None:
                if prev_a == 1:                                    # 정지 = 종료
                    r = R_STOP_ATK if prev_atk else R_STOP_CLEAN
                    ag.push(prev_s, prev_a, r, s, True); _ls.append(ag.learn()); n_trans += 1
                    epr += r; stopped = i-1; break
                r = R_CONT_ATK if prev_atk else R_CONT_CLEAN
                ag.push(prev_s, prev_a, r, s, False); _ls.append(ag.learn()); n_trans += 1
                epr += r
            prev_a = ag.act(s, eps_ep); prev_s = s; prev_atk = a_now
            if (i & 3) == 0:
                try: _qs.append(float(max(ag.get_q_values(s))))
                except Exception: pass
        delay = (stopped - true_on) if (stopped >= 0 and true_on >= 0 and stopped >= true_on) else None
        hist.append(dict(ep=ei, eps=eps_ep, reward=epr,
                         loss=(float(sum(x[0] for x in _ls)/len(_ls)) if _ls else None),
                         qmax=(float(max(_qs)) if _qs else None),
                         qavg=(float(sum(_qs)/len(_qs)) if _qs else None),
                         tvar=(float(sum(x[2] for x in _ls)/len(_ls)) if _ls else None), stopped=stopped, onset=true_on, delay=delay,
                         fa=int(stopped >= 0 and (true_on < 0 or stopped < true_on)),
                         miss=int(stopped < 0 and true_on >= 0), n_trans=n_trans))
    return dict(learner=name, sec=time.time()-t0, n_trans=n_trans, hist=hist)

def main():
    D = dict(np.load(DATA))
    rng = np.random.default_rng(SEED)
    order = rng.integers(0, len(D['ep_start']), size=N_EP)     # 전 학습기 동일 제시 순서
    learners = os.environ.get('STOP_LEARNERS', 'SWIRL,Adam1e-3,Adam3e-4,Adam3e-4ams,EKF-TD,UKF-TD').split(',')
    out = {}
    for nm in learners:
        m = run(nm, D, order)
        h = m['hist']; last = h[-100:]
        print(f"  {nm:12s} 후반100 보상 {np.mean([x['reward'] for x in last]):7.2f} · "
              f"오경보 {np.mean([x['fa'] for x in last]):.3f} · 미탐 {np.mean([x['miss'] for x in last]):.3f} · "
              f"지연 {np.mean([x['delay'] for x in last if x['delay'] is not None] or [np.nan]):5.2f} · "
              f"전이 {m['n_trans']:,} · {m['sec']:.0f}s", flush=True)
        out[nm] = m
    o = os.environ.get('STOP_OUT', f'{R}/results/claudecodefortest/night/offline_stop_s{SEED}.json')
    json.dump(out, open(o, 'w'), default=float); print(f'저장 {o}')

main()
