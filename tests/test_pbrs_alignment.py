"""P2′ 성형 — **실제로 push 되는 전이**의 보상 정렬 (리뷰 1 시험을 저장소로 옮김, 09-24):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_pbrs_alignment.py

  RewardTracker 는 에피소드 첫 호출 행에서 F=0·Φ_prev 초기화만 한다. 이것이 설계(Φ_0 를 첫 F 에도 적용)와 같은
  결과가 되려면 호출부가 '첫 행은 전이를 만들지 않는다'는 불변식을 지켜야 한다 — 이 파일이 그것을 단언한다.
  push 를 가로채 (s, a, r, s′) 를 state 벡터로 steps 행과 맞춰
    r = reward_train(행 j), r − r^G = γ·φ_j − φ_{j−1}, j = i+1(연속 행), prev_action(행 j) = a, terminal 이면 φ_j = 0
  를 확인한다. 경로: 구 벽시계 · SIMCLOCK 인라인(+REENGAGE_FIX) · SIMCLOCK+LEARNER_PROC+보류 격자 · surrogate(train.run_surrogate).
"""
import atexit
import glob
import math
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TMP = tempfile.mkdtemp(prefix='pbrs_align_')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


# ───────────────────────── Isaac 노드 경로 (rclpy·px4_msgs 필요) ─────────────────────────
def _harness():
    pytest.importorskip('rclpy'); pytest.importorskip('px4_msgs')
    import _isaac_node_harness as H
    os.chdir(H.ROOT)
    return H


def _tilt_plant(H):
    class TiltPlant(H.Plant):
        def gt_msg(self, us):
            m = super().gt_msg(us)
            a = 0.05 * math.sin(0.05 * self.k) + 0.02; b = 0.04 * math.cos(0.031 * self.k)
            w = math.sqrt(max(0.0, 1 - (a / 2) ** 2 - (b / 2) ** 2))
            m.pose.pose.orientation = NS(w=w, x=a / 2, y=b / 2, z=0.0)
            return m

    class StaticTilt(TiltPlant):
        def step(self, node, dt=0.02):
            self.k += 1
    return TiltPlant, StaticTilt


def _exp(out, *extra):
    from cfgload import load_experiment
    return load_experiment(['configs/isaac_v5_adam.yaml'], ['run.device=cpu', 'agent.batch=16', 'run.ep_steps=150', f'run.outdir={out}',
                                                            'env.kind=isaac', 'reward.shape_tilt=4', 'log.steps=true', *extra])


def _check(out, pushes, g):
    files = sorted(glob.glob(os.path.join(out, 'steps', 'ep*.npz')))
    assert files
    rows = []
    for f in files:
        z = np.load(f); cols = [str(c) for c in z['cols']]
        c = {n: i for i, n in enumerate(cols)}
        R = z['rows']
        sc = [c[f's{i}'] for i in range(sum(1 for n in cols if n.startswith('s') and n[1:].isdigit()))]
        for j in range(len(R)):
            rows.append((os.path.basename(f), j, tuple(R[j, sc]), R[j, c['reward']], R[j, c['reward_train']], R[j, c['phi']],
                         R[j, c['prev_action']], R[j, c['done']]))
    idx = {}
    for r in rows:
        idx.setdefault(r[2], []).append(r)
    n_ok = n_amb = n_sw = 0
    for (s, a, r, s2, term) in pushes:
        k1 = tuple(float(x) for x in s); k2 = tuple(float(x) for x in s2)
        c1, c2 = idx.get(k1, []), idx.get(k2, [])
        if len(c1) != 1 or len(c2) != 1:
            n_amb += 1; continue
        (f1, j1, _, _, _, phi1, pa1, _), (f2, j2, _, rG2, rT2, phi2, pa2, _) = c1[0], c2[0]
        assert f1 == f2 and j2 == j1 + 1, (f1, j1, f2, j2)                   # 연속 행
        assert int(pa2) == int(a)                                           # 행 j 의 prev_action = push 된 행동
        assert math.isclose(r, rT2, rel_tol=0, abs_tol=1e-9), (r, rT2)       # 학습 보상 = steps reward_train
        assert math.isclose(r - rG2, g * phi2 - phi1, abs_tol=1e-9), (r - rG2, g * phi2 - phi1)
        if term:
            assert phi2 == 0.0
        n_sw += int(pa1 != pa2)
        n_ok += 1
    return n_ok, n_amb, len(pushes), n_sw


def _wrap_push(agent, store):
    orig = agent.push

    def p(s, a, r, s2, term):
        store.append((np.array(s, float).copy(), int(a), float(r), np.array(s2, float).copy(), bool(term)))
        return orig(s, a, r, s2, term)
    agent.push = p


def test_legacy_push_alignment():
    H = _harness(); TiltPlant, _ = _tilt_plant(H)
    mod = H.load_module(os.path.join(H.ROOT, 'online_rl_main.py'), 'orm_pa_leg')
    out = os.path.join(_TMP, 'leg'); H.seed_all(42)
    exp = _exp(out); exp.cfg.eps_start = 0.6
    pushes = []
    with H.patched_ros(mod, []):
        node = H.make_node(mod, exp, {}, sync_threads=True)
        _wrap_push(node.agent, pushes)
        node._start_new_episode(); node.flight_state = 'STABILIZE'
        H.run(node, TiltPlant(), 2600)
    res = _check(out, pushes, exp.reward.shape_gamma)
    assert res[0] > 100 and res[3] > 0, res


def test_simclock_inline_push_alignment():
    H = _harness(); TiltPlant, _ = _tilt_plant(H)
    mod = H.load_module(os.path.join(H.ROOT, 'online_rl_main.py'), 'orm_pa_sc')
    out = os.path.join(_TMP, 'sc'); H.seed_all(42)
    exp = _exp(out); exp.cfg.eps_start = 0.6
    pushes = []
    with H.patched_ros(mod, []):
        node = H.make_node(mod, exp, {'SIMCLOCK_UKF': 1, 'TIMING_LOG': 1, 'REENGAGE_FIX': 1})
        _wrap_push(node.agent, pushes)
        node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
        H.run(node, TiltPlant(), 2200, tick=False)
        assert node._fatal is None
    res = _check(out, pushes, exp.reward.shape_gamma)
    assert res[0] > 100 and res[3] > 0, res


def test_simclock_lp_deferred_push_alignment():
    H = _harness(); _, StaticTilt = _tilt_plant(H)
    mod = H.load_module(os.path.join(H.ROOT, 'online_rl_main.py'), 'orm_pa_def')
    out = os.path.join(_TMP, 'def'); exp = _exp(out); exp.cfg.eps_start = 0.6; H.seed_all(int(exp.cfg.seed))
    knobs = {'SIMCLOCK_UKF': 1, 'LEARNER_PROC': 1, 'ACT_LAT_SIM': 0.04, 'TIMING_LOG': 1, 'SENSOR_NOISE_SCALE': 0}
    pushes = []
    with H.patched_ros(mod, []):
        node = H.make_node(mod, exp, knobs)
        px = node.agent
        _wrap_push(px, pushes)
        hold = [0]; _orig_ready = px.ready_to_act; _orig_la = px.learn_async
        px.ready_to_act = lambda: _orig_ready() and hold[0] <= 0
        n_upd = [0]

        def _la():
            r = _orig_la()
            if r[1]:
                n_upd[0] += 1
                if n_upd[0] % 2 == 0:
                    hold[0] = 7                                           # 학습 회신 보류 → 보류 격자 경로를 탄다
            return r
        px.learn_async = _la
        plant = StaticTilt()
        try:
            node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
            for i in range(1, 2201):
                us = i * 20000
                plant.step(node); plant.feed_sensors(node)
                hold[0] -= 1
                node._cb_gt(plant.gt_msg(us))
                if us % 100000 == 0:
                    node._cb_gps(plant.gps_msg(us))
                t_end = time.time() + 60
                while px._inflight > 0:
                    node._sc_drain(); time.sleep(0.0005)
                    assert time.time() < t_end
                node._sc_drain(); node._tick()
            assert node._fatal is None
        finally:
            px.close()
    res = _check(out, pushes, exp.reward.shape_gamma)
    assert res[0] > 100 and res[3] > 0, res


# ───────────────────────── surrogate 경로 ─────────────────────────
def test_surrogate_push_alignment():
    import test_surrogate_theta as TS
    if TS.MAIN is None:
        pytest.skip('원본 풀 없음(결과 폴더 없는 서버)')
    ROOT = TS.ROOT
    os.chdir(ROOT)
    from cfgload import load_experiment
    import env.reward as ER
    import build_pool_knn as B
    src = [str(s) for s in np.load(os.path.join(TS.MAIN, TS.POOL_REL))['sources']]
    files = [s if s.startswith('/') else os.path.join(TS.MAIN, s) for s in src][::10]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 60:
        pytest.skip('원본 에피소드 파일 부족')
    out = os.path.join(_TMP, 'pool_v4_align.npz'); B.build(out, files)
    exp = load_experiment(['configs/newenv_v5.yaml', 'configs/overlays/surrogate_knn_v3clean.yaml'],
                          [f'env.surrogate.pool={out}', 'run.device=cpu', 'run.episodes=6', f'run.outdir={_TMP}/surr_o',
                           'reward.shape_tilt=4', 'env.surrogate.theta=true', 'log.eval_n=0', 'log.probe_every=0',
                           'agent.batch=16', 'scenario.attack.deadline_steps=3'])
    g = exp.reward.shape_gamma
    rows = []; pushes = []; ends = []
    orig_step = ER.RewardTracker.step

    def st(self, *a, **k):
        r = orig_step(self, *a, **k)
        rows.append((r, self.last_rG, self.last_F, self.last_phi, bool(k.get('terminated', False)), a[0]))
        return r
    ER.RewardTracker.step = st
    import train
    orig_make = train.make_agent

    def mk(e):
        ag = orig_make(e); op = ag.push; oe = ag.end_episode

        def p(s, a, r, s2, d):
            pushes.append((len(rows) - 1, a, r, d)); return op(s, a, r, s2, d)

        def ee(*x):
            ends.append(len(rows)); return oe(*x)
        ag.push = p; ag.end_episode = ee
        return ag
    train.make_agent = mk
    try:
        hist = train.run_surrogate(exp, log=lambda *a, **k: None)
    finally:
        ER.RewardTracker.step = orig_step; train.make_agent = orig_make
    starts = [0] + ends[:-1]
    first_rows = set(starts)
    n = n_term = 0
    for (ri, a, r, d) in pushes:
        assert ri not in first_rows, ri                      # 에피 첫 행은 전이를 만들지 않는다(호출부 불변식)
        r_, rG, F, phi, term, pa = rows[ri]
        phi0 = rows[ri - 1][3]
        assert r == r_ and pa == a and d == term
        assert abs((r - rG) - (g * phi - phi0)) < 1e-9
        if d:
            assert phi == 0.0; n_term += 1
        n += 1
    for s in starts:
        assert rows[s][2] == 0.0
    for i, h in enumerate(hist):                             # 보고 reward = Σ r^G, reward_train = Σ 학습 보상
        e0, e1 = starts[i], ends[i]
        assert abs(h['reward'] - sum(x[1] for x in rows[e0:e1])) < 1e-6
        assert abs(h['reward_train'] - sum(x[0] for x in rows[e0:e1])) < 1e-6
    assert n > 500
