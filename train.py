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
            s = ob.push(v, g, prev_a)
            if s is not None:
                a = agent.act(s, 0.0)
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


def run_surrogate(exp, log=print):
    from sim.surrogate import SurrogateEnv
    cfg = exp.cfg
    ob = ObsBuilder(exp.obs)
    tracker = RewardTracker(exp.reward)
    agent = make_agent(exp)
    ep_steps = cfg.episode_max_steps
    genv = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps, cfg.seed + 777)
    dwell = int(cfg.hover_dwell)
    pe, pn = int(exp.log['probe_every']), int(exp.log['probe_n'])
    hist = []
    for ep in range(cfg.max_episodes):
        t0 = time.time()
        genv.reset(); ob.reset(); tracker.reset()
        prev_s = None; prev_a = 0; dw_left = 0
        tp = fp = fn = tn = 0; wtp = wfn = stp = sfn = 0
        epr = 0.0; losses = []; qs = []; tv = []; innov = []; adapt = []; nisf = []; flip = []
        det_delay = None; t = 0
        for t in range(ep_steps):
            v, g, atk = genv.nis(prev_a)
            s = ob.push(v, g, prev_a)
            if s is None:
                if genv.step(): break
                continue
            adly = genv.attack_delay()
            r = tracker.step(prev_a, atk, adly, terminated=genv.crashed)
            if atk:
                weak = float(genv.plan.delta[genv.t]) < 0.35
                if prev_a == 1:
                    tp += 1; wtp += weak; stp += (not weak)
                    if det_delay is None: det_delay = adly
                else:
                    fn += 1; wfn += weak; sfn += (not weak)
            else:
                if prev_a == 1: fp += 1
                else: tn += 1
            epr += r
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
                   wrec=(wtp / (wtp + wfn) if wtp + wfn else float('nan')), srec=(stp / (stp + sfn) if stp + sfn else float('nan')),
                   sec=time.time() - t0)
        if pe > 0 and ep % pe == 0:
            row.update(probe(agent, exp, ob, pn, episode=ep))
        hist.append(row)
        if ep % 10 == 0 or ep == cfg.max_episodes - 1:
            log(f"ep{ep:4d} R={epr:8.2f} loss={row['loss']:.4f} F1={row['f1']:.3f} FPR={row['fpr']:.3f} "
                f"qmax={row['qmax'] if row['qmax'] is None else round(row['qmax'], 2)} eps={agent.get_epsilon():.3f} ({row['sec']:.1f}s)")
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
