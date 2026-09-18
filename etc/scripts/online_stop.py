#!/usr/bin/env python3
"""online_stop (final7) — 온라인 최적정지. 착륙(stop)하면 에피소드 종료, 다음 판으로.

★ surrogate_run.py 를 건드리지 않는 독립 스크립트(실행 중 실험 오염 방지).

설계 근거(전부 오늘 실측):
  · 관측 L=6 × [nis_v, nis_g] = 12D, action 제외
      - stop 이 종료라 prev_action 은 항상 'track' = 상수 → 정보량 0
      - L=6 이득의 원천은 '공격 신호 누적'이 아니라 '평시 잡음 감소':
        유효표본 gyro 공격 1.05→1.08(ρ0.96, 거의 상수) vs 평시 1.94→2.56(ρ0.5)
        실측 d' 이득 4→6: δ0.30-0.40 ws10 +0.55, δ0.70-0.80 ws10 +0.74
      - 12D → net[16,16] 514 파라미터 = 기존 프레임워크와 동일(비교 가능)
  · 보상: 옛 MATLAB ver2.6 비율, c=0.2
      평시+track +0.4 │ 평시+stop −2.0(종료) │ 공격+track −1.0 │ 공격+stop +3.0(종료)
  · ε: 에피소드 기준 지수 감쇠(옛 코드 main_all.m:53). 스텝 기준은 작동하지 않는다
      — 탐험으로 즉시 착륙 → 스텝이 안 쌓임 → ε 가 안 내려감 → 악순환(오늘 오프라인에서 실증)
  · 탐험 시 stop 확률 STOP_PSTOP (옛 코드는 균등 0.5)
  · 클립 제거 + 선택적 /4 정규화(SURR_OBS_DIV) — 오늘 찾은 OBS_CLIP 버그 반영
"""
import os, sys, time, json
import numpy as np

R = '/home/acsl/projects/Issacsim-rhukf'
sys.path.insert(0, R); sys.path.insert(0, f'{R}/etc/scripts')

C = float(os.environ.get('STOP_C', '0.2'))
R_CONT_CLEAN, R_STOP_CLEAN = 2*C, -10*C       # +0.4 / -2.0
R_CONT_ATK,   R_STOP_ATK   = -5*C, 15*C       # -1.0 / +3.0
N_EP     = int(os.environ.get('STOP_EPISODES', '500'))
EP_STEPS = int(os.environ.get('STOP_EP_STEPS', '300'))
SEED     = int(os.environ.get('STOP_SEED', '42'))
P_STOP   = float(os.environ.get('STOP_PSTOP', '0.5'))
L_WIN    = int(os.environ.get('STOP_L', '6'))
OBS_DIV  = float(os.environ.get('SURR_OBS_DIV', '1.0') or 1.0)
EPS_HI, EPS_LO, EPS_FRAC = 0.95, 0.01, float(os.environ.get('STOP_EPS_FRAC', '0.8'))

def eps_at(ep, n):
    rate = -np.log(EPS_LO / EPS_HI) / max(1.0, n * EPS_FRAC)
    return EPS_LO + (EPS_HI - EPS_LO) * float(np.exp(-rate * ep))

class Win:
    """L 프레임 × [v, g] — action 없음, 클립 없음, 선택적 /OBS_DIV"""
    def __init__(self, L, div): self.L, self.div = L, div
    def reset(self): self.buf = []
    def push(self, v, g):
        self.buf.append([v / self.div, g / self.div])
        if len(self.buf) > self.L: self.buf.pop(0)
        return np.array(self.buf, np.float32).ravel() if len(self.buf) == self.L else None

def make_agent(name, dimS):
    os.environ.update({'SURROGATE': '1', 'NET_HIDDEN': '16', 'BATCH_SIZE': '128',
                       'GAMMA': os.environ.get('STOP_GAMMA', '0.9'),
                       'BUFFER_SIZE': '50000', 'NSTEP_OFF': '1'})
    for k in ('RHUKF_MODE', 'ADAM_AMSGRAD', 'OPT'): os.environ.pop(k, None)
    if name.startswith('Adam'):
        os.environ.update(ADAM_LR=name[4:].replace('ams', ''),
                          ADAM_AMSGRAD='1' if name.endswith('ams') else '0',
                          ADAM_INIT='he', ADAM_HUBER_BETA='3')
        kind = 'adam'
    else:
        os.environ.update(RHUKF_TAU='0.005', RHUKF_UI='1', RHUKF_FORM='absolute', RHUKF_SPAS='1',
                          RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3',
                          RHUKF_PINIT='0.03', RHUKF_R='1')
        os.environ['RHUKF_N'] = '7' if name == 'SWIRL' else '1'
        for _e, _k in (('SW_P','RHUKF_PINIT'), ('SW_R','RHUKF_R'), ('SW_N_OVR','RHUKF_N'), ('SW_A','RHUKF_ALPHA')):
            if os.environ.get(_e): os.environ[_k] = os.environ[_e]   # ★튜닝 배선
        if name == 'EKF-TD': os.environ['RHUKF_MODE'] = 'ekf'
        kind = 'rhukf'
    import importlib, swrl_config; importlib.reload(swrl_config)
    cfg = swrl_config.Config()
    cfg.dimS = dimS; cfg.window_size = L_WIN; cfg.obs_scale = [1.0]*dimS
    cfg.eps_action_probs = [1.0 - P_STOP, P_STOP]
    if kind == 'adam':
        from rl.agent_adam import OnlineAdamAgent as AG
    else:
        from rl.agent import OnlineRHUKFAgent as AG
    return AG(cfg)

def run(name):
    import torch, importlib
    import surrogate_env8 as SE; importlib.reload(SE)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dimS = L_WIN * 2
    ag = make_agent(name, dimS)
    env = SE.SurrogateEnv(ep_steps=EP_STEPS, crash=True, seed=SEED + 777)
    win = Win(L_WIN, OBS_DIV)
    hist = []; n_tr = 0; t0 = time.time()
    for ep in range(N_EP):
        env.reset(); win.reset()
        e = eps_at(ep, N_EP)
        prev_s = None; prev_a = 0; prev_atk = False
        epr = 0.0; stopped = -1; onset = -1; steps = 0; _ls = []; _qs = []
        for t in range(EP_STEPS):
            v, g, atk = env.nis(prev_a)
            if atk and onset < 0: onset = t
            s = win.push(v, g)
            if s is None:
                if env.step(): break
                continue
            if prev_s is not None:
                if prev_a == 1:                                   # stop = 착륙 = 종료
                    r = R_STOP_ATK if prev_atk else R_STOP_CLEAN
                    ag.push(prev_s, 1, r, s, True); _ls.append(ag.learn()); n_tr += 1
                    epr += r; stopped = t - 1; break
                r = R_CONT_ATK if prev_atk else R_CONT_CLEAN
                crashed = bool(getattr(env, 'crashed', False))
                ag.push(prev_s, 0, r, s, crashed); _ls.append(ag.learn()); n_tr += 1
                epr += r
                if crashed: break
            prev_a = ag.act(s, e); prev_s = s; prev_atk = bool(atk); steps = t
            if (t & 15) == 0:
                try: _qs.append(float(max(ag.get_q_values(s))))
                except Exception: pass
            if env.step(): break
        d = (stopped - onset) if (stopped >= 0 and onset >= 0 and stopped >= onset) else None
        hist.append(dict(ep=ep, eps=e, reward=epr,
                         loss=(float(sum(x[0] for x in _ls)/len(_ls)) if _ls else None),
                         qmax=(float(max(_qs)) if _qs else None),
                         qavg=(float(sum(_qs)/len(_qs)) if _qs else None),
                         tvar=(float(sum(x[2] for x in _ls)/len(_ls)) if _ls else None), stopped=stopped, onset=onset, delay=d,
                         steps=steps, n_trans=n_tr,
                         fa=int(stopped >= 0 and (onset < 0 or stopped < onset)),
                         miss=int(stopped < 0 and onset >= 0),
                         crashed=int(getattr(env, 'crashed', False))))
    return dict(learner=name, sec=time.time() - t0, n_trans=n_tr, hist=hist,
                cfg=dict(L=L_WIN, div=OBS_DIV, p_stop=P_STOP, n_ep=N_EP, seed=SEED))

def main():
    learners = os.environ.get('STOP_LEARNERS', 'SWIRL,Adam3e-4,Adam1e-3').split(',')
    out = {}
    for nm in learners:
        m = run(nm); h = m['hist']; last = h[-100:]
        dl = [x['delay'] for x in last if x['delay'] is not None]
        print(f"  {nm:12s} 후반100 보상 {np.mean([x['reward'] for x in last]):7.2f} · "
              f"오경보 {np.mean([x['fa'] for x in last]):.3f} · 미탐 {np.mean([x['miss'] for x in last]):.3f} · "
              f"지연 {np.mean(dl) if dl else float('nan'):5.2f} · 길이 {np.mean([x['steps'] for x in last]):5.1f} · "
              f"전이 {m['n_trans']:,} · {m['sec']:.0f}s", flush=True)
        out[nm] = m
    o = os.environ.get('STOP_OUT', f'{R}/results/claudecodefortest/night/online_stop_s{SEED}.json')
    json.dump(out, open(o, 'w'), default=float); print(f'저장 {o}')

main()
