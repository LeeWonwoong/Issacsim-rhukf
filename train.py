#!/usr/bin/env python3
"""train.py — 단일 진입점. 환경(surrogate | isaac)·관측·보상·학습기를 YAML 로 정한다.

    python3 train.py --config configs/cp_regime.yaml
    python3 train.py --config configs/cp_regime.yaml --set agent.type=adam --set run.outdir=results/claudecodefortest/x

산출물(run.outdir): config.yaml(최종 병합 설정) · hist.json(에피소드별 지표) · train.log
에피소드 루프(surrogate):
    환경 → 원시 NIS → ObsBuilder(관측) → 학습기 행동 → RewardTracker(보상) → push/learn
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from cfgload import add_cli, load_experiment, save_resolved          # noqa: E402
from env.observation import ObsBuilder                                # noqa: E402
from env.reward import RewardTracker                                  # noqa: E402


def make_agent(exp):
    import torch
    torch.manual_seed(exp.cfg.seed); np.random.seed(exp.cfg.seed)
    if exp.cfg.agent_type == 'adam':
        from rl.agent_adam import OnlineAdamAgent as AG
    else:
        from rl.agent import OnlineRHUKFAgent as AG
    return AG(exp.cfg)


def probe(agent, exp, ob_template, n_ep, episode=0):
    """greedy(ε=0) 프로브: 학습·push 없음, 별도 시드의 환경, steps_done 복원."""
    from sim.surrogate import SurrogateEnv
    sd0 = agent.steps_done
    ep_steps = exp.cfg.episode_max_steps
    genv = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps, exp.cfg.seed + 9000,
                        reseed_per_episode=False)
    ob = copy.deepcopy(ob_template)
    tp = fp = fn = tn = 0; dels = []; crashes = 0
    for _ in range(n_ep):
        genv.ep_idx = int(episode) - 1          # ★09-18 검토: 바람 schedule 이 있으면 현재 학습 에피소드의 체제로 평가 (없으면 무영향)
        genv.reset(); ob.reset(); prev_a = 0; det = None
        for _t in range(ep_steps):
            v, g, atk = genv.nis(prev_a)
            s = ob.push(v, g, prev_a, key=(1, int(episode), _, _t))
            if s is not None:
                a = agent.act(s, 0.0, greedy=True)
                if atk:
                    if prev_a == 1:
                        tp += 1
                        if det is None: det = genv.attack_delay()
                    else:
                        fn += 1
                else:
                    if prev_a == 1: fp += 1
                    else: tn += 1
                prev_a = a
            if genv.step(): break
        if det is not None: dels.append(det)
        elif genv.plan.has_attack: dels.append(ep_steps)
        crashes += int(genv.crashed)
    agent.steps_done = sd0
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(probe_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, probe_fpr=fp / (fp + tn) if fp + tn else 0.0,
                probe_rec=rec, probe_delay=float(np.mean(dels)) if dels else float('nan'), probe_crash=crashes)


def evaluate(agent, exp, ob_template, n_ep, seed_off=7000):
    """★09-19 최종 평가(논문 보고용): 학습 끝난 정책을 greedy(ε=0)로 별도 시드·서로 다른 n_ep 에피소드에 돌려
    스텝 F1/정밀도/재현율/FPR, 사건 탐지율·지연, 그룹별 recall 을 낸다. 학습·push 없음. (탐지기 자체 성능 = 다른 논문 지표와 비교 가능)"""
    from sim.surrogate import SurrogateEnv
    sd0 = agent.steps_done; ep_steps = exp.cfg.episode_max_steps
    genv = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps, exp.cfg.seed + seed_off, reseed_per_episode=True)
    ob = copy.deepcopy(ob_template)
    tp = fp = fn = tn = 0; by_cls = {}; ev_seen = ev_det = 0; dels = []; crashes = 0; fa_eps = 0; n_clean = 0
    for k in range(n_ep):
        genv.ep_idx = exp.cfg.max_episodes - 1 + k           # 에피소드마다 다른 (시드, ep) 난수 + 바람 schedule 은 마지막 체제
        genv.reset(); ob.reset(); prev_a = 0; ep_fp = 0; det_t = {}
        for _t in range(ep_steps):
            v, g, atk = genv.nis(prev_a)
            s = ob.push(v, g, prev_a, key=(2, k, _t))
            if s is not None:
                if atk:
                    tt = max(genv.t - (1 if genv.knn else 0), 0); c = genv.plan.cls_at(tt); b = by_cls.setdefault(c, [0, 0])
                    ev = int(genv.plan.bstart[tt]) if hasattr(genv.plan, 'bstart') else 0
                    det_t.setdefault(ev, None)                 # 사건 등록(시작 스텝 = id)
                    if prev_a == 1:
                        tp += 1; b[0] += 1
                        if det_t[ev] is None: det_t[ev] = genv.attack_delay()   # 사건 첫 탐지 지연
                    else:
                        fn += 1; b[1] += 1
                elif prev_a == 1: fp += 1; ep_fp += 1
                else: tn += 1
                prev_a = agent.act(s, 0.0, greedy=True)
            if genv.step(): break
        for ev, d in det_t.items():
            ev_seen += 1
            if d is not None: ev_det += 1; dels.append(d)
        crashes += int(genv.crashed)
        if not genv.plan.has_attack: n_clean += 1; fa_eps += int(ep_fp > 0)
    agent.steps_done = sd0
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(n_ep=n_ep, f1=(2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec=prec, rec=rec,
                fpr=(fp / (fp + tn) if fp + tn else 0.0), event_det=(ev_det / ev_seen if ev_seen else float('nan')), n_events=ev_seen,
                delay=(float(np.mean(dels)) if dels else float('nan')), delay_med=(float(np.median(dels)) if dels else float('nan')),
                fa_episode_rate=(fa_eps / n_clean if n_clean else float('nan')), crash=crashes,
                rec_by_cls={k: (v[0] / (v[0] + v[1]) if v[0] + v[1] else None) for k, v in sorted(by_cls.items())})


def run_surrogate(exp, log=print):
    from sim.surrogate import SurrogateEnv
    cfg = exp.cfg
    ob = ObsBuilder(exp.obs)
    tracker = RewardTracker(exp.reward)
    agent = make_agent(exp)
    ep_steps = cfg.episode_max_steps
    genv = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps, cfg.seed + 777)
    if genv.cfg.theta:                                      # ★09-24 θ 코퓰러 출처 기록(config / pool / J32a 기본값)
        log(f'[θ] surrogate θ 채널 코퓰러 {tuple(round(x, 4) for x in genv._theta_cop)} (출처 {genv.theta_copula_src})')
    dwell = int(cfg.hover_dwell)
    pe, pn = int(exp.log['probe_every']), int(exp.log['probe_n'])
    hist = []
    for ep in range(cfg.max_episodes):
        t0 = time.time()
        genv.reset(); ob.reset(); tracker.reset()
        prev_s = None; prev_a = 0; dw_left = 0
        tp = fp = fn = tn = 0; wtp = wfn = stp = sfn = 0
        epr = 0.0; epr_train = 0.0; epF = 0.0; ths = []; losses = []; qs = []; tv = []; innov = []; adapt = []; nisf = []; flip = []; n_r = 0; by_cls = {}
        det_delay = None; t = 0
        for t in range(ep_steps):
            v, g, atk = genv.nis(prev_a)
            s = ob.push(v, g, prev_a, key=(0, ep, t))
            if s is None:
                if genv.step(): break
                continue
            adly = genv.attack_delay()
            # ★09-23 P2′: surrogate θ 채널(env.surrogate.theta)을 Isaac 과 같은 코드·같은 동결 규칙으로 (roll=θ, pitch=0)
            r = tracker.step(prev_a, atk, adly, terminated=genv.crashed,
                             roll=genv.theta, pitch=(0.0 if genv.theta is not None else None))
            n_r += 1
            if genv.theta is not None: ths.append(genv.theta)
            if tracker.shaping: epr_train += r; epF += tracker.last_F
            if atk:
                _c = genv.plan.cls_at(max(genv.t - (1 if genv.knn else 0), 0)); _b = by_cls.setdefault(_c, [0, 0])
                _b[0 if prev_a == 1 else 1] += 1                       # 그룹별 [TP, FN]
                weak = float(genv.plan.delta[max(genv.t - (1 if genv.knn else 0), 0)]) < 0.35   # ★09-21 검토: knn 풀은 라벨=δ[t−1] → 약/강 recall 로깅 인덱스 정합(학습 무관)
                if prev_a == 1:
                    tp += 1; wtp += weak; stp += (not weak)
                    if det_delay is None: det_delay = adly
                else:
                    fn += 1; wfn += weak; sfn += (not weak)
            else:
                if prev_a == 1: fp += 1
                else: tn += 1
            epr += tracker.last_rG                     # 보고용 = r^G (성형 끔이면 r 과 같다)
            if prev_s is not None:
                agent.push(prev_s, prev_a, r, s, bool(genv.crashed))
                out = agent.learn()
                if out is not None and out[0]:
                    losses.append(out[0])
                    if len(out) > 2 and out[2] is not None: tv.append(float(out[2]))
                    innov.append(float(getattr(agent, '_last_innov', 0.0) or 0.0))
                    adapt.append(float(getattr(agent, '_last_adapt', 0.0) or 0.0))
                    nisf.append(float(getattr(agent, '_last_nis', 0.0) or 0.0))
                    flip.append(float(getattr(agent, '_last_argmax_flip', 0.0) or 0.0))
            if (len(losses) & 15) == 0:
                try: qs.append(float(max(agent.get_q_values(s))))
                except Exception: pass
            a = agent.act(s, agent.get_epsilon())
            if dwell > 0:                               # hover 확약: 진입 후 D−1 스텝 강제 hover
                if dw_left > 0: a = 1; dw_left -= 1
                elif a == 1 and prev_a == 0: dw_left = dwell - 1
            prev_s = s; prev_a = a
            if genv.step(): break
        agent.end_episode(epr, t + 1)
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        mean = lambda x: float(np.mean(x)) if x else 0.0
        row = dict(ep=ep, reward=epr, f1=(2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec=prec, rec=rec,
                   fpr=(fp / (fp + tn) if fp + tn else 0.0), delay=(float(det_delay) if det_delay is not None else float('nan')),
                   loss=mean(losses), qmax=(float(np.max(qs)) if qs else None), qavg=(float(np.mean(qs)) if qs else None),
                   tvar=(float(np.mean(tv)) if tv else None), has_atk=int(tp + fn > 0), crashed=int(genv.crashed),
                   dmax=genv.plan.dmax, cls=genv.plan.cls, n_events=len(genv.plan.events) or int(genv.plan.has_attack), ws=genv.ws, n_upd=len(innov), innov=mean(innov), adapt=mean(adapt),
                   nisf=mean(nisf), aflip=mean(flip), kgain=float(getattr(agent, '_last_kgain', 0.0) or 0.0),
                   pmax=float(getattr(agent, '_last_pmax', 0.0) or 0.0),
                   reward_cost=epr - exp.reward.scale * exp.reward.alive * n_r,     # 생존 보상 뺀 부분(오경보·지연·탐지)
                   rec_by_cls={k: (v[0] / (v[0] + v[1]) if v[0] + v[1] else None) for k, v in by_cls.items()},
                   wrec=(wtp / (wtp + wfn) if wtp + wfn else float('nan')), srec=(stp / (stp + sfn) if stp + sfn else float('nan')),
                   sec=time.time() - t0)
        if tracker.shaping:                             # ★09-23 P2′: 학습 보상 합·성형항 합(지표는 r^G 인 'reward')
            row.update(reward_train=epr_train, shape_F=epF)
        if ths:
            row.update(theta_mean=float(np.mean(ths)))
        if pe > 0 and ep % pe == 0:
            row.update(probe(agent, exp, ob, pn, episode=ep))
        hist.append(row)
        if ep % 10 == 0 or ep == cfg.max_episodes - 1:
            log(f"ep{ep:4d} R={epr:8.2f} loss={row['loss']:.4f} F1={row['f1']:.3f} FPR={row['fpr']:.3f} "
                f"qmax={row['qmax'] if row['qmax'] is None else round(row['qmax'], 2)} eps={agent.get_epsilon():.3f} ({row['sec']:.1f}s)")
    ne = int(exp.log.get('eval_n', 0))
    if ne > 0:                                          # ★09-19 최종 greedy 평가 + 모델 저장
        ev = evaluate(agent, exp, ob, ne)
        with open(os.path.join(cfg.outdir, 'eval.json'), 'w') as f: json.dump(ev, f, default=float, ensure_ascii=False)
        log(f"[eval] greedy {ne}ep: F1={ev['f1']:.3f} P={ev['prec']:.3f} R={ev['rec']:.3f} FPR={ev['fpr']:.4f} 사건탐지 {ev['event_det']:.3f} 지연 {ev['delay']:.2f} "
            + ' '.join(f"{k}={v:.2f}" for k, v in ev['rec_by_cls'].items() if v is not None))
        try: agent.save(os.path.join(cfg.outdir, 'final_model.pt'))
        except Exception as e: log(f"[eval] 모델 저장 실패: {e}")
    return hist


def main(argv=None):
    ap = add_cli(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args(argv)
    exp = load_experiment(args.config, args.set)
    out = exp.run['outdir']
    save_resolved(exp, out)
    logf = open(os.path.join(out, 'train.log'), 'a')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    c = exp.cfg
    log(f"[train] env={exp.env_kind} agent={exp.agent_type} γ={c.gamma} n={c.n_step_size if c.use_n_step else 1} "
        f"reward={exp.reward.mode}×{exp.reward.scale} obs={exp.obs.compress}/clip{exp.obs.clip}/div{exp.obs.div}/W{exp.obs.window} "
        f"attack={exp.scenario.attack.family} wind={exp.scenario.wind.mode} seed={c.seed} ep={c.max_episodes}×{c.episode_max_steps}")
    if float(exp.reward.shape_tilt) > 0:
        log(f"[train] P2′ 성형 λ={exp.reward.shape_tilt} θ0={exp.reward.tilt_ref} 동결={exp.reward.shape_freeze} γ={exp.reward.shape_gamma}")
    if exp.agent_type != 'adam':
        log(f"[train] SWIRL form={c.state_form} anchor={c.anchor_type} argmax={c.ddqn_argmax}/{c.h0_online_moving_init} "
            f"pΔ={c.p_delta_init} huber={c.huber_c} N={c.N_horizon} R={c.r_init} q={c.q_init} α={c.alpha} τ={c.tau_srrhuif} ui={c.update_interval}")
    t0 = time.time()
    if exp.env_kind == 'surrogate':
        hist = run_surrogate(exp, log)
    else:
        from online_rl_main import run_isaac
        hist = run_isaac(exp, log)
    rec = dict(name=exp.run['name'], sec=time.time() - t0, hist=hist)
    tmp = os.path.join(out, 'hist.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f, default=float)
    os.replace(tmp, os.path.join(out, 'hist.json'))
    log(f"[train] 완료 {rec['sec'] / 60:.1f}분 → {out}")
    return hist


if __name__ == '__main__':
    main()
