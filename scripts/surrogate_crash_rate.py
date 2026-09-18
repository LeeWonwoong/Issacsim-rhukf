#!/usr/bin/env python3
"""scripts/surrogate_crash_rate.py — surrogate 추락률 점검(항상 track).  python3 scripts/surrogate_crash_rate.py <설정.yaml,덮어쓰기.yaml>"""
# surrogate 추락률: 항상 track 정책. 강공격(δ≥0.7) 사건에 track 노출 20+ 인 에피소드 중 추락 비율 (Isaac 2/52 ≈ 4%)
import sys, numpy as np
sys.path.insert(0, '.')
from cfgload import load_experiment
from sim.surrogate import SurrogateEnv
exp = load_experiment(sys.argv[1].split(','), [])
ep_steps = exp.cfg.episode_max_steps
env = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps, 12345)
n_exp = n_crash = n_all_crash = 0; tcr = []
for e in range(3000):
    env.reset()
    while True:
        env.nis(0)
        if env.step(): break
    P = env.plan
    strong = [(ev["start"], ev["end"]) for ev in P.events if P.delta[ev["start"]:ev["end"]].max() >= 0.7 and ev["end"] - ev["start"] >= 20] if P.has_attack else []
    n_all_crash += env.crashed
    if strong:
        n_exp += 1; n_crash += env.crashed
        if env.crashed: tcr.append(env.t - strong[0][0])
print(f'{sys.argv[1]}: 강공격 노출 20+ 에피소드 {n_exp} 중 추락 {n_crash} ({n_crash/max(n_exp,1):.1%}) · 전체 3000 중 추락 {n_all_crash} · 온셋→추락 {np.percentile(tcr,[10,50,90]) if tcr else "-"}')
