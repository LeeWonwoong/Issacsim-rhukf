"""REENGAGE_FIX 합성 시험 (ROS 노드·Isaac 없음, 오프라인 하네스):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_reengage_fix.py

  _learning_setpoint 를 원(circle) 궤적에서 직접 구동한다. 기체는 명령 설정점을 1차 지연으로 따라간다.
  (a) 노브 0(구 동작): 2 m 안에서 hover 를 풀면 _did_hover 가 남고, 매 track 틱 판정이 궤적 시계를 240 위상 격자로 재동기한다(버그 재현)
  (b) 노브 1: 2 m 안 해제 → 첫 track 틱에 _did_hover=False, 재동기 0 회, 시계 = 정지 시각 + dt·k, 설정점 증분이 R·ω·dt 로 연속
  (c) 노브 1: 멀리(6 m) 해제 → 재접근 동안 시계 불변, 도달 틱에서 재동기 정확히 1 회, 이후 dt 씩 전진
  (d) 노브 1 + 구 벽시계 경로(_legacy_track_dt): hover 동안 기준 스탬프가 따라가 해제 첫 틱 dt = 0.02 (노브 0 은 hover 길이만큼 뜀)
  (e) SIMCLOCK 통합 스모크: REENGAGE_FIX + P2′ 성형(λ4) 켜고 노드 구동 → steps 열(reward_train·theta_eff·phi)·타이밍 열 불변식·
      F = γΦ_t − Φ_{t−1} 을 steps 로 재구성
  (f) 기본값(노브 0·성형 0): SIMCLOCK 인라인 경로가 7f77609 와 비트 동일(구 벽시계 경로는 test_isaac_node_offline 이 d089dc5 기준으로 본다)
"""
import glob
import math
import os
import sys
import tempfile
import shutil
import atexit
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('px4_msgs')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaac_node_harness import ROOT, Plant, load_module, make_node, patched_ros, run, seed_all   # noqa: E402

os.chdir(ROOT)
from cfgload import load_experiment   # noqa: E402

_TMP = tempfile.mkdtemp(prefix='reengage_fix_')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
DT = 0.02


def _node(mod, knobs, *extra, tag='x'):
    exp = load_experiment(['configs/isaac_v5_adam.yaml'], ['run.device=cpu', 'agent.batch=16', 'run.ep_steps=150',
                                                          f'run.outdir={os.path.join(_TMP, tag)}', *extra])
    node = make_node(mod, exp, knobs)
    node._reset_episode_state()
    node.scenario = {'pattern': 'circle'}
    node._hover_alt = -2.5
    node.cur_pos[:] = [0.0, 0.0, -2.5]
    return node


class Drive:
    """_learning_setpoint 를 틱마다 호출. 기체 xy 는 control_sp 로 1차 지연(τ 0.15 s)."""
    def __init__(self, node, legacy=False):
        self.n = node; self.legacy = legacy; self.sim_t = 0.0
        self.log = []                                        # (hover, sim_flight_t, did_hover, reengaging, control_sp)
        self.resync_calls = 0; self.dts = []
        orig = node._reengage_nearest

        def _wrap(resync=True):
            before = (node._sim_flight_t, node._wp_s)
            out = orig(resync) if 'resync' in orig.__code__.co_varnames else orig()
            if (node._sim_flight_t, node._wp_s) != before:
                self.resync_calls += 1
            return out
        node._reengage_nearest = _wrap
        if hasattr(node, '_apply_nearest_phase'):
            ap = node._apply_nearest_phase

            def _ap():
                self.resync_calls += 1; ap()
            node._apply_nearest_phase = _ap

    def tick(self, hover, push=None):
        n = self.n
        self.sim_t += DT
        n._gt_sim_time = self.sim_t
        if self.legacy:
            def td():
                v = n._legacy_track_dt(); self.dts.append(v); return v
        else:
            td = lambda: DT
        c, _ = n._learning_setpoint(hover, td)
        k = 1.0 - math.exp(-DT / 0.15)
        n.cur_pos[0] += k * (c[0] - n.cur_pos[0]); n.cur_pos[1] += k * (c[1] - n.cur_pos[1])
        if push is not None:
            n.cur_pos[0] += push[0]; n.cur_pos[1] += push[1]
        self.log.append((hover, n._sim_flight_t, bool(getattr(n, '_did_hover', False)), bool(getattr(n, '_reengaging', False)), c))
        return c


def _mod(tag):
    return load_module(os.path.join(ROOT, 'online_rl_main.py'), f'orm_rf_{tag}')


def _circle_step(node):
    return node.cfg.flight_radius * node.cfg.flight_omega * DT


def test_old_behavior_resyncs_every_tick_after_near_release():
    mod = _mod('old')
    with patched_ros(mod, []):
        node = _node(mod, {}, tag='old')
        d = Drive(node)
        for _ in range(150): d.tick(False)
        for _ in range(10): d.tick(True)
        d.resync_calls = 0
        post = [d.tick(False) for _ in range(60)]
    tail = d.log[-60:]
    assert all(x[2] for x in tail)                              # _did_hover 가 끝까지 남는다(은닉 상태)
    assert d.resync_calls >= 55                                 # 매 track 틱 판정이 시계를 재동기(부수효과)
    dts = np.diff([x[1] for x in tail])
    assert np.max(np.abs(dts - DT)) > 1e-6                      # 시계 증분이 dt 가 아니다(위상 격자 양자화)


def test_fix_near_release_continuous_no_resync():
    mod = _mod('near')
    with patched_ros(mod, []):
        node = _node(mod, {'REENGAGE_FIX': 1}, tag='near')
        d = Drive(node)
        pre = [d.tick(False) for _ in range(150)]
        t_stop = node._sim_flight_t
        for _ in range(10): d.tick(True)
        assert node._sim_flight_t == t_stop                     # hover 동안 시계 정지
        post = [d.tick(False) for _ in range(60)]
    tail = d.log[-60:]
    assert not tail[0][2] and not any(x[2] or x[3] for x in tail)   # 첫 track 틱에 이력 해제, 재접근 없음
    assert d.resync_calls == 0
    exp_t = t_stop
    for x in tail:
        exp_t += DT
        assert x[1] == exp_t                                    # 시계 = 정지 시각 + dt·k (비트 단위로 같은 누적)
    step = _circle_step(node)
    sp = np.array([c[:2] for c in pre + post])
    jumps = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    assert np.max(jumps) <= step * 1.001, (np.max(jumps), step)   # hover 를 건너서도 설정점 연속(한 dt 만큼)


def test_fix_far_release_single_resync_on_arrival():
    mod = _mod('far')
    with patched_ros(mod, []):
        node = _node(mod, {'REENGAGE_FIX': 1}, tag='far')
        d = Drive(node)
        for _ in range(150): d.tick(False)
        t_stop = node._sim_flight_t
        # hover 직전 원 바깥쪽(중심 (−R,0) 기준)으로 6 m 밀려남 — 누수홀드 앵커는 밀린 자리에서 잡힌다
        node._hover_anchor = None
        c0 = np.array(node.cur_pos[:2]) - np.array([-node.cfg.flight_radius, 0.0]); out = c0 / np.linalg.norm(c0)
        node.cur_pos[0] += 6.0 * out[0]; node.cur_pos[1] += 6.0 * out[1]
        for _ in range(10): d.tick(True)
        assert node._sim_flight_t == t_stop
        d.resync_calls = 0
        n_re = 0
        for i in range(400):
            d.tick(False)
            if d.log[-1][3]:
                n_re += 1
                assert node._sim_flight_t == t_stop            # 재접근 중 시계 불변
            if not d.log[-1][2] and not d.log[-1][3]:
                break
        assert n_re > 3                                         # 실제로 재접근 기동을 했다
        assert d.resync_calls == 1                              # 도달 틱에서 1 회
        t_arr = node._sim_flight_t
        assert t_arr != t_stop
        for _ in range(20): d.tick(False)
        assert d.resync_calls == 1 and not node._did_hover
        assert math.isclose(node._sim_flight_t, t_arr + 20 * DT, abs_tol=1e-9)


@pytest.mark.parametrize('fix', [0, 1])
def test_legacy_clock_freeze(fix):
    mod = _mod(f'leg{fix}')
    with patched_ros(mod, []):
        node = _node(mod, {'REENGAGE_FIX': 1} if fix else {}, tag=f'leg{fix}')
        d = Drive(node, legacy=True)
        for _ in range(150): d.tick(False)
        t_stop = node._sim_flight_t
        for _ in range(15): d.tick(True)
        d.dts.clear()
        d.tick(False)
    jump = node._sim_flight_t - t_stop
    if fix:
        assert d.dts == [pytest.approx(DT)] and math.isclose(jump, DT, abs_tol=1e-9)
    else:
        assert d.dts == [pytest.approx(16 * DT)]                 # 구 경로: 해제 첫 틱 dt = hover 길이 + 1틱(0.32 s)
        assert abs(jump - DT) > 0.05                             # (그 위에 최근접 재동기까지 섞여) 시계 불연속


class TiltPlant(Plant):
    """자세가 움직이는 가짜 플랜트(성형 θ 가 0 이 아니도록)."""
    def gt_msg(self, us):
        m = super().gt_msg(us)
        a = 0.05 * math.sin(0.05 * self.k) + 0.02; b = 0.04 * math.cos(0.031 * self.k)   # |roll|,|pitch| < 0.1 (STABILIZE 통과)
        # 작은 각 쿼터니언(x≈roll/2, y≈pitch/2) — 부호·프레임은 _quat_to_euler 가 처리, θ̂ 는 크기만 본다
        w = math.sqrt(max(0.0, 1 - (a / 2) ** 2 - (b / 2) ** 2))
        m.pose.pose.orientation = NS(w=w, x=a / 2, y=b / 2, z=0.0)
        return m


def test_simclock_integration_shaping_and_fix():
    mod = _mod('sc')
    out = os.path.join(_TMP, 'sc_int'); seed_all(42)
    exp = load_experiment(['configs/isaac_v5_adam.yaml'], ['run.device=cpu', 'agent.batch=16', 'run.ep_steps=150',
                                                          f'run.outdir={out}', 'env.kind=isaac', 'reward.shape_tilt=4', 'log.steps=true'])
    exp.cfg.eps_start = 0.6                                                     # hover 가 섞이게
    log = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, {'SIMCLOCK_UKF': 1, 'TIMING_LOG': 1, 'REENGAGE_FIX': 1})
        node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
        run(node, TiltPlant(), 2200, tick=False)
        assert node._fatal is None
    files = sorted(glob.glob(os.path.join(out, 'steps', 'ep*.npz')))
    assert files
    g = exp.reward.shape_gamma
    n_rows = n_hov = 0
    for f in files:
        z = np.load(f); cols = [str(c) for c in z['cols']]; R = z['rows']; c = {n: i for i, n in enumerate(cols)}
        assert cols[-3:] == ['reward_train', 'theta_eff', 'phi']
        i0 = cols.index('t_gps_us')
        assert np.all(R[:, c['n_pred']] == 5) and np.all(R[:, c['dt_gps_us']] == 100000)   # 타이밍 열이 제자리(행 끝 기준 아님)
        assert np.all(np.isfinite(R[:-1, c['act_wait_ms']]))
        F = R[:, c['reward_train']] - R[:, c['reward']]
        phi = R[:, c['phi']]
        assert F[0] == 0.0
        np.testing.assert_allclose(F[1:], g * phi[1:] - phi[:-1], atol=1e-9)
        th = np.hypot(R[:, c['roll']], R[:, c['pitch']])
        pa = R[:, c['prev_action']]
        sw = np.r_[False, pa[1:] != pa[:-1]]
        free = np.ones(len(R), bool)
        for s in np.flatnonzero(sw):
            free[s:s + 3] = False
        np.testing.assert_allclose(R[free, c['theta_eff']], th[free], atol=1e-12)      # 동결 밖은 원값
        assert np.any(th > 0.02)
        n_rows += len(R); n_hov += int(pa.sum())
    assert n_hov > 0
    with open(os.path.join(out, 'metrics_adam.csv')) as fh:
        head = fh.readline().strip().split(',')
    assert 'reward_train' in head and 'shape_F' in head and head.index('reward') < head.index('reward_train')


@pytest.mark.parametrize('kind', ['adam', 'swirl'])
def test_simclock_default_bit_identical_to_7f77609(kind):
    """노브·성형 기본값: SIMCLOCK 인라인 경로가 변경 전 커밋(7f77609)과 비트 동일(벽시계 ms 열만 제외 — 런마다 다른 값)."""
    import subprocess
    import test_isaac_node_offline as TO
    ref = os.path.join(_TMP, f'orm_7f77609_{kind}.py')
    with open(ref, 'wb') as f:
        f.write(subprocess.check_output(['git', 'show', '7f77609:online_rl_main.py'], cwd=ROOT))

    def go(path, tag):
        mod = load_module(path, f'orm_bit_{tag}'); out = os.path.join(_TMP, f'bit_{tag}'); seed_all(42)
        exp = TO._exp(kind, out, 'log.steps=true')
        with patched_ros(mod, []):
            node = make_node(mod, exp, {'SIMCLOCK_UKF': 1, 'TIMING_LOG': 1})
            node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
            run(node, Plant(), 1800, tick=False)
        mf = [f for f in os.listdir(out) if f.startswith('metrics_')][0]
        with open(os.path.join(out, mf)) as fh:
            met = fh.read()
        return dict(sp=TO._setpoints(node), atk=TO._attacks(node), steps=TO._steps(out), metrics=met, ep=node.episode)

    a = go(ref, f'{kind}_ref'); b = go(os.path.join(ROOT, 'online_rl_main.py'), f'{kind}_new')
    assert a['ep'] == b['ep'] and a['steps']
    assert np.array_equal(a['sp'], b['sp'], equal_nan=True) and a['atk'] == b['atk'] and a['metrics'] == b['metrics']
    assert a['steps'].keys() == b['steps'].keys()
    for k in a['steps']:
        ca, ra = a['steps'][k]; cb, rb = b['steps'][k]
        assert int(ra[:, ca.index('prev_action')].sum()) > 0                      # hover·재접근 경로를 실제로 탔다
        keep = [i for i, n in enumerate(ca) if n not in TO._WALL]
        assert ca == cb and np.array_equal(ra[:, keep], rb[:, keep], equal_nan=True), k
