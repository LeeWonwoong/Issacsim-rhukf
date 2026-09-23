"""Isaac 노드 오프라인 테스트 (ROS 노드·Isaac·PX4 없음 — rclpy.init 도 안 한다):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_isaac_node_offline.py

  1) 노브 0(구 벽시계 틱 경로)이 변경 전 커밋(_LEGACY_REF)의 online_rl_main.py 와 **비트 동일**: 같은 합성 GT/GPS 열·시드에서
     발행한 궤적 setpoint·공격 δ·steps npz·metrics csv·최종 θ 가 같다(학습 스레드는 두 쪽 모두 동기 실행으로 고정).
  2) SIMCLOCK_UKF+LEARNER_PROC+ACT_LAT_SIM+TIMING_LOG: 실시간 흉내(GT 간 벽시계 대기) 하에서
     n_pred==5·Σdt 0.1·RL 간격 100000 µs·gap 0·적용 지연 0.04 고정(overrun 은 따로 계수), 에피 종료·리셋·재시작 정상.
  3) SIMCLOCK_UKF 인라인 동기 학습(SWIRL): 같은 타이밍 불변식 + 적용 지연 = 다음 GT(0.02).
  4) rev2 격자 분리: 학습기가 격자 한 칸(0.1 s) 넘게 늦어도(ready_to_act 를 일부러 늦춤) 격자 처리·δ 발행은 sim 시각대로,
     정지 플랜트에서 행동·스텝 행·δ·θ·metrics 가 늦추지 않은 런과 같다(보류 RL 앞부분의 스냅샷 복원 검증).
  5) 치명 경로: run_sim 이 SIMCLOCK 이 아닐 때(float 스탬프) 조용히 돌지 않고 _fatal+SystemExit · LEARNING 중 스탬프 역행 → HARD
     · run_isaac 의 knob/sim_env 불일치·인라인+speed>1 거부.
"""
import atexit
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip('rclpy')        # ROS 없는 원격(surrogate) 서버에서는 수집 단계에서 건너뛴다
pytest.importorskip('px4_msgs')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaac_node_harness import ROOT, Plant, load_module, make_node, patched_ros, run, seed_all   # noqa: E402

os.chdir(ROOT)
from cfgload import load_experiment   # noqa: E402

_LEGACY_REF = 'd089dc5'              # SIMCLOCK·LEARNER_PROC 도입 직전 커밋 — 커밋 뒤에도 비교 기준이 새 코드 자신이 되지 않게 고정
_TMP = tempfile.mkdtemp(prefix='isaac_node_offline_')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_TIMING11 = ['t_gps_us', 'dt_gps_us', 'n_pred', 'sum_dt_pred', 'gps_gap', 'gt_gap', 'node_lag_sim',
             'act_wait_ms', 'learn_ms', 'act_applied_sim_lat', 'overrun']
_WALL = {'act_wait_ms', 'learn_ms', 'gt_rx_lag_ms', 'imu_age_ms', 'u_age_ms'}


def _exp(kind, outdir, *extra):
    f = 'configs/isaac_v5_swirl.yaml' if kind == 'swirl' else 'configs/isaac_v5_adam.yaml'
    return load_experiment([f], ['run.device=cpu', 'agent.batch=16', 'run.ep_steps=150', f'run.outdir={outdir}', *extra])


def _setpoints(node):
    return np.array([list(m.position) + list(m.velocity) + [m.yaw] + list(m.acceleration) for m in node.pub_traj.msgs], float)


def _attacks(node):
    return [(bool(m.active), tuple(float(x) for x in m.torque), float(m.thrust)) for m in node.pub_actuator_attack.msgs]


def _steps(outdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(outdir, 'steps', 'ep*.npz'))):
        z = np.load(f, allow_pickle=False)
        out[os.path.basename(f)] = (list(z['cols']), z['rows'])
    return out


def _legacy_run(mod, outdir):
    seed_all(42)
    exp = _exp('adam', outdir)
    log = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, {}, sync_threads=True)
        node._start_new_episode(); node.flight_state = 'STABILIZE'
        run(node, Plant(), 2600)                                   # 52 s sim: STABILIZE → 에피 1(15 s) → SOFT → 에피 2 …
        sd = {k: v.detach().numpy().copy() for k, v in node.agent.net.state_dict().items()}
        with open(os.path.join(outdir, 'metrics_adam.csv')) as f:
            metrics = f.read()
        return dict(sp=_setpoints(node), atk=_attacks(node), steps=_steps(outdir), metrics=metrics, theta=sd,
                    ep=node.episode, step=node.step_count, state=node.flight_state)


def test_legacy_path_bit_identical_to_ref():
    ref = os.path.join(_TMP, 'online_rl_main_ref.py')
    with open(ref, 'wb') as f:
        f.write(subprocess.check_output(['git', 'show', f'{_LEGACY_REF}:online_rl_main.py'], cwd=ROOT))
    old = _legacy_run(load_module(ref, 'orm_ref'), os.path.join(_TMP, 'legacy_ref'))
    new = _legacy_run(load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_new'), os.path.join(_TMP, 'legacy_new'))
    assert old['ep'] >= 2 and len(old['steps']) >= 1, (old['ep'], list(old['steps']))
    assert (old['ep'], old['step'], old['state']) == (new['ep'], new['step'], new['state'])
    assert old['sp'].shape == new['sp'].shape and np.array_equal(old['sp'], new['sp'], equal_nan=True)
    assert old['atk'] == new['atk']
    assert old['steps'].keys() == new['steps'].keys()
    for k in old['steps']:
        assert old['steps'][k][0] == new['steps'][k][0] and np.array_equal(old['steps'][k][1], new['steps'][k][1])
    assert old['metrics'] == new['metrics']
    assert all(np.array_equal(old['theta'][k], new['theta'][k]) for k in old['theta'])


def _check_timing(outdir, lat_expect, allow_overrun_frac=0.0):
    st = _steps(outdir)
    assert st, 'no steps npz'
    rows_all = []
    for name, (cols, rows) in st.items():
        i0 = cols.index('t_gps_us')
        assert cols[i0:i0 + 11] == _TIMING11 and 'act_eff_lat' in cols and 'deferred' in cols
        c = {n: i for i, n in enumerate(cols)}
        r = rows
        assert np.all(r[:, c['n_pred']] == 5) and np.allclose(r[:, c['sum_dt_pred']], 0.1)
        assert np.all(r[:, c['dt_gps_us']] == 100000)
        assert np.all(r[:, c['gps_gap']] == 0) and np.all(r[:, c['gt_gap']] == 0)
        assert np.all(np.diff(r[:, c['t_gps_us']]) == 100000)
        assert np.all(np.diff(r[:, c['step']]) == 1)
        lat = r[:-1, c['act_applied_sim_lat']]; ov = r[:-1, c['overrun']]   # 마지막(종료) 행은 행동 없음 → NaN
        assert np.isnan(r[-1, c['act_applied_sim_lat']])
        ok = ov == 0
        assert np.allclose(lat[ok], lat_expect), np.unique(lat)
        assert np.all(lat[~ok] > lat_expect)
        assert np.all(r[:-1, c['act_eff_lat']] >= lat - 1e-9)
        rows_all.append(r)
    r = np.concatenate(rows_all); c = {n: i for i, n in enumerate(st[next(iter(st))][0])}
    frac = float(np.nanmean(r[:, c['overrun']]))
    assert frac <= allow_overrun_frac, frac
    return r, c


def test_simclock_learner_proc_fixed_latency():
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_sc_lp')
    out = os.path.join(_TMP, 'sc_lp'); seed_all(42)
    exp = _exp('adam', out)
    log = []
    knobs = {'SIMCLOCK_UKF': 1, 'LEARNER_PROC': 1, 'ACT_LAT_SIM': 0.04, 'TIMING_LOG': 1}
    with patched_ros(mod, log):
        node = make_node(mod, exp, knobs)
        try:
            assert node._sc and node._lp and node._scq.act_lat_us == 40000
            node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
            run(node, Plant(), 1500, pace_s=0.004)                  # ≈ speed 5 실시간 흉내: 에피 1 끝 + SOFT → 에피 2 시작
            assert node.episode >= 2
            px = node.agent
            assert px.n_mismatch == 0 and px.n_errors == 0 and px._proc.is_alive()
        finally:
            node.agent.close()
        assert not node.agent._proc.is_alive()
    r, c = _check_timing(out, 0.04, allow_overrun_frac=0.10)
    assert np.nanmax(r[:, c['learn_ms']]) > 0                         # 갱신 스텝의 learn 시간이 기록됨
    assert np.all(np.isfinite(r[:, c['gt_rx_lag_ms']])) and np.all(np.isfinite(r[:, c['imu_age_ms']]))
    assert any('timing: n_pred={5:' in ln for ln in log), [ln for ln in log if 'timing' in ln][:3]
    assert any('simclock: overrun=' in ln for ln in log)
    with open(os.path.join(out, 'metrics_adam.csv')) as f:
        head = f.readline().strip().split(',')
    assert head[-6:] == ['overrun', 'n_apply', 'overrun_frac', 'node_lag_max', 'act_lag_max', 'act_eff_lat_max']


def test_simclock_inline_swirl():
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_sc_inline')
    out = os.path.join(_TMP, 'sc_inline'); seed_all(42)
    exp = _exp('swirl', out)
    log = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, {'SIMCLOCK_UKF': 1, 'TIMING_LOG': 1})
        node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
        run(node, Plant(), 1000, tick=False)                        # 틱 없이도(콜백 드레인만) 돈다
    r, c = _check_timing(out, 0.02)
    assert np.nanmax(r[:, c['learn_ms']]) > 0
    assert not any('LEARN ERROR' in ln for ln in log)


class _StaticPlant(Plant):
    """setpoint 를 따라가지 않는 정지 플랜트 — 행동 적용 시각이 달라도 센서·NIS 가 같다(보류 경로 비교용)."""
    def step(self, node, dt=0.02):
        self.k += 1


def _deferral_run(kind, tag, stall):
    """LEARNER_PROC · 정지 플랜트 · 센서 노이즈 0. 두 경우 모두 GT 사이에 실제 학습기 회신을 전부 받는다(CPU 속도 무관·결정적).
    stall>0: 갱신 요청 둘 중 하나 뒤 GT stall 개 동안 ready_to_act=False(학습기가 격자 한 칸 넘게 늦은 것과 같은 결정 시각)."""
    import time
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), f'orm_def_{tag}')
    out = os.path.join(_TMP, f'def_{tag}'); exp = _exp(kind, out); seed_all(int(exp.cfg.seed))
    knobs = {'SIMCLOCK_UKF': 1, 'LEARNER_PROC': 1, 'ACT_LAT_SIM': 0.04, 'TIMING_LOG': 1, 'SENSOR_NOISE_SCALE': 0}
    log = []; acts = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, knobs)
        px = node.agent
        hold = [0]; _orig_ready = px.ready_to_act; _orig_la = px.learn_async
        px.ready_to_act = lambda: _orig_ready() and hold[0] <= 0

        n_upd = [0]

        def _la():
            r = _orig_la()
            if stall and r[1]:
                n_upd[0] += 1
                if n_upd[0] % 2 == 0:                                   # 갱신 요청 둘 중 하나: 다음 격자를 넘겨 늦게 결정(지연은 쌓이지 않게)
                    hold[0] = stall
            return r
        px.learn_async = _la
        _atk = []; _pa = node.pub_actuator_attack.publish                # δ 발행을 에피소드 번호와 함께
        node.pub_actuator_attack.publish = lambda m: (_atk.append((node.episode, bool(m.active), tuple(float(x) for x in m.torque))), _pa(m))[1]
        _orig_dec = node._sc_decide

        def _wrap():
            _orig_dec(); acts.append((node.episode, node.step_count - 1, int(node.prev_action)))
        node._sc_decide = _wrap
        plant = _StaticPlant()
        try:
            node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
            for i in range(1, 2201):
                us = i * 20000
                plant.step(node); plant.feed_sensors(node)
                hold[0] -= 1
                node._cb_gt(plant.gt_msg(us))
                if us % 100000 == 0:
                    node._cb_gps(plant.gps_msg(us))
                # 실제 학습기 회신은 GT 사이에 전부 받는다(무한히 빠른 학습기) → 늦음은 hold 로만(결정적). stall==0 이면 hold 도 없음.
                t_end = time.time() + 60
                while px._inflight > 0 or (not stall and not _orig_ready()):
                    node._sc_drain(); time.sleep(0.0005)
                    assert time.time() < t_end
                node._sc_drain()
                node._tick()
            d = px.get_theta()
            res = dict(acts=acts, th=d['theta'], ep=node.episode, atk_ep=_atk, atk=_attacks(node), mis=(px.n_mismatch, px.n_errors),
                       deferred=sum(1 for ln in log if 'deferred=' in ln and 'deferred=0 ' not in ln), fatal=node._fatal)
        finally:
            px.close()
    res['steps'] = _steps(out)
    with open(os.path.join(out, f'metrics_{"adam" if kind == "adam" else "rhukf"}.csv')) as f:
        res['metrics'] = f.read()
    return res


@pytest.mark.parametrize('kind', ['adam', 'swirl'])
def test_deferred_grid_equivalence(kind):
    a = _deferral_run(kind, f'{kind}_fast', 0)
    b = _deferral_run(kind, f'{kind}_slow', 7)                        # 갱신 요청 둘 중 하나: GT 7개(0.14 s > RL 주기) 동안 결정 불가 → 다음 격자 보류
    # 에피소드 1 만 비교: 느린 런은 에피 1 종료를 늦게 알아채 에피 2 가 다른 GT 위상(합성 센서)에서 시작하므로 그 뒤는 입력 자체가 다르다
    assert a['ep'] >= 2 and b['ep'] >= 2
    assert b['deferred'] > 0 and a['mis'] == b['mis'] == (0, 0) and a['fatal'] is None and b['fatal'] is None
    common = ['ep0001.npz']; n_ep = 1
    ea = [x for x in a['acts'] if x[0] == 1]; eb = [x for x in b['acts'] if x[0] == 1]
    assert len(ea) > 100 and ea == eb and any(x[2] == 1 for x in ea)   # hover 결정(적용 때 캡처)도 포함
    da = [x for x in a['atk_ep'] if x[0] == 1]; db = [x for x in b['atk_ep'] if x[0] == 1]
    assert len(da) > 0 and da == db                                   # δ 발행 열(격자 시각에 묶임)이 같다
    for k in common:
        ca, ra = a['steps'][k]; cb, rb = b['steps'][k]
        keep = [i for i, n in enumerate(ca) if n not in _WALL | {'gt_err', 'act_applied_sim_lat', 'overrun', 'act_eff_lat',
                                                                   'deferred', 'node_lag_sim'}]
        assert ca == cb and np.array_equal(ra[:, keep], rb[:, keep], equal_nan=True), k
        c = {n: i for i, n in enumerate(cb)}
        assert np.all(rb[:, c['n_pred']] == 5) and np.all(rb[:, c['dt_gps_us']] == 100000)
        assert np.nansum(rb[:, c['deferred']]) > 0 and np.nanmax(rb[:, c['act_applied_sim_lat']]) >= 0.1 - 1e-9
    ma = [l.split(',') for l in a['metrics'].splitlines()][:n_ep + 1]; mb = [l.split(',') for l in b['metrics'].splitlines()][:n_ep + 1]
    hdr = ma[0]; keep = [i for i, n in enumerate(hdr) if n not in ('overrun', 'n_apply', 'overrun_frac', 'node_lag_max',
                                                                   'act_lag_max', 'act_eff_lat_max')]
    assert [[r[i] for i in keep] for r in ma] == [[r[i] for i in keep] for r in mb]


def test_simclock_mismatch_is_fatal():
    """노드 SIMCLOCK=1 인데 run_sim 은 구 동작(float 스탬프·float GPS 게이트): 조용히 학습하지 않고 _fatal + SystemExit."""
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_mis')
    out = os.path.join(_TMP, 'mis'); exp = _exp('adam', out); seed_all(42)
    log = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, {'SIMCLOCK_UKF': 1})
        node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
        plant = Plant(); sim_t = 0.0; last_gps = 0.0; step = 0
        with pytest.raises(SystemExit):
            for _ in range(250 * 30):
                sim_t += 0.004; step += 1
                if step % 5 == 0:
                    plant.step(node); plant.feed_sensors(node)
                    sec = int(sim_t); us = sec * 1_000_000 + int((sim_t - sec) * 1e9) // 1000   # run_sim 구 float 스탬프
                    node._cb_gt(plant.gt_msg(us))
                if sim_t - last_gps >= 0.1:
                    node._cb_gps(plant.gps_msg(int(sim_t * 1e6))); last_gps = sim_t
        assert node._fatal and 'SIMCLOCK' in node._fatal


def test_regress_in_learning_triggers_hard_reset():
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_reg')
    out = os.path.join(_TMP, 'reg'); exp = _exp('adam', out); seed_all(42)
    log = []
    with patched_ros(mod, log):
        node = make_node(mod, exp, {'SIMCLOCK_UKF': 1})
        node._start_new_episode(); node.flight_state = 'STABILIZE'; node._sc_enter_stabilize()
        plant = Plant()
        run(node, plant, 400, tick=False)
        assert node.flight_state == 'LEARNING'
        node.sim_mgr.restart = lambda: None
        node._cb_gt(plant.gt_msg(20000))                               # 스탬프 역행
        assert node.flight_state == 'HARD_RESET' and node._fatal is None
        assert not node._scq.q and node._sc_ctx is None and not node._sc_defq


def test_run_isaac_rejects_knob_mismatch():
    mod = load_module(os.path.join(ROOT, 'online_rl_main.py'), 'orm_ri')
    from env.knobs import set_knobs

    def _passed(*a, **k):                                             # 검사를 통과해 버리면 XRCE·Isaac 을 띄우기 전에 멈춘다
        raise AssertionError('run_isaac 이 불일치 설정을 거부하지 않았다')
    mod._ensure_xrce_agent = _passed; mod.rclpy = type('R', (), {'init': staticmethod(_passed)})
    for knobs, env, speed in (({'SIMCLOCK_UKF': 1}, {'SIMCLOCK_UKF': '0'}, 1.0),      # 노드 켬 · sim 끔
                              ({'SIMCLOCK_UKF': 1}, {}, 2.0)):                         # 인라인 + speed 2
        exp = _exp('adam', os.path.join(_TMP, 'ri'), 'env.kind=isaac', f'env.isaac.speed={speed}')
        exp.isaac['sim_env'] = dict(env)
        set_knobs(knobs)
        with pytest.raises(ValueError):
            mod.run_isaac(exp, log=lambda *a, **k: None)
    set_knobs({})


if __name__ == '__main__':
    import time
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            t0 = time.time()
            if k == 'test_deferred_grid_equivalence':
                f('adam'); f('swirl')
            else:
                f()
            print('ok', k, f'{time.time() - t0:.1f}s', flush=True)
