"""Isaac 노드 오프라인 하네스 — ROS 노드를 띄우지 않고(rclpy.init·DDS 없음) OnlineRLNode 를 합성 메시지로 구동.

rclpy Node 의 생성자·create_*·get_logger 를 가짜로 바꾸고, SimProcessManager.start 를 막는다.
가짜 플랜트: 기체 위치가 마지막 궤적 setpoint 를 1차 지연으로 따라간다(결정적). 센서는 결정적 합성 값.
"""
import contextlib
import importlib.util
import os
import random
import sys
import time
import threading as _real_threading
import types
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakePub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class FakeLog:
    def __init__(self, sink=None):
        self.lines = [] if sink is None else sink

    def _w(self, m):
        self.lines.append(str(m))
    info = warn = warning = error = debug = _w


class _SyncThread:
    """구 경로 비트 비교용: learn 스레드를 start() 즉시 동기 실행(두 코드 모두 같은 규칙)."""
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._t, self._a, self._k = target, args, (kwargs or {})

    def start(self):
        self._t(*self._a, **self._k)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def patched_ros(mod, log_sink):
    import rclpy.node
    N = rclpy.node.Node
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(N, '__init__', lambda self, *a, **k: None))
        st.enter_context(mock.patch.object(N, 'create_publisher', lambda self, *a, **k: FakePub()))
        st.enter_context(mock.patch.object(N, 'create_subscription', lambda self, *a, **k: None))
        st.enter_context(mock.patch.object(N, 'create_timer', lambda self, *a, **k: None))
        _lg = FakeLog(log_sink)
        st.enter_context(mock.patch.object(N, 'get_logger', lambda self: _lg))
        st.enter_context(mock.patch.object(mod.SimProcessManager, 'start', lambda self: None))
        st.enter_context(mock.patch.object(mod.SimProcessManager, 'stop', lambda self: None))
        yield


def make_node(mod, exp, knobs_d, sync_threads=False):
    from env.knobs import set_knobs
    set_knobs(knobs_d)
    mod._apply_module_knobs()
    if sync_threads:
        mod.threading = types.SimpleNamespace(Thread=_SyncThread, Lock=_real_threading.Lock)
    node = mod.OnlineRLNode(exp.cfg, exp)
    for n in ('pub_traj', 'pub_att', 'pub_offboard', 'pub_cmd', 'pub_actuator_attack'):
        setattr(node, n, FakePub())
    node._fmu_ready = True; node._first_gt_received = True
    return node


class Plant:
    """결정적 가짜 플랜트(NED). 위치가 마지막 궤적 setpoint 로 1차 지연(τ=0.3 s) 추종."""
    def __init__(self, alt=2.5):
        self.p = np.array([0.0, 0.0, -alt]); self.v = np.zeros(3); self.k = 0

    def step(self, node, dt=0.02):
        self.k += 1
        tgt = self.p
        if node.pub_traj.msgs:
            m = node.pub_traj.msgs[-1]
            tgt = np.array([float(x) for x in m.position])
        v = (tgt - self.p) / 0.3
        n = np.linalg.norm(v[:2])
        if n > 2.0:
            v[:2] *= 2.0 / n
        self.v = v; self.p = self.p + v * dt

    def gt_msg(self, us):
        e, n_, u = self.p[1], self.p[0], -self.p[2]                 # NED → ENU
        return NS(header=NS(stamp=NS(sec=us // 1_000_000, nanosec=(us % 1_000_000) * 1000)),
                  pose=NS(pose=NS(position=NS(x=e, y=n_, z=u), orientation=NS(w=1.0, x=0.0, y=0.0, z=0.0)),
                          covariance=[time.time()] + [0.0] * 35),          # run_sim SIMCLOCK: [0] = 발행 벽시계
                  twist=NS(twist=NS(linear=NS(x=self.v[1], y=self.v[0], z=-self.v[2]))))

    def gps_msg(self, us):
        lat0, lon0, alt0, R = 47.397742, 8.545594, 488.0, 6371000.0
        e, n_ = self.p[1], self.p[0]
        return NS(latitude_deg=lat0 + np.degrees(n_ / R), longitude_deg=lon0 + np.degrees(e / (R * np.cos(np.radians(lat0)))),
                  altitude_msl_m=alt0 - self.p[2], vel_n_m_s=self.v[0], vel_e_m_s=self.v[1], vel_d_m_s=self.v[2],
                  timestamp=int(us))

    def feed_sensors(self, node):
        w = 0.05 * np.sin(0.37 * self.k)
        node._cb_sensor(NS(accelerometer_m_s2=[0.0, 0.0, -9.81], gyro_rad=[w, -0.5 * w, 0.1 * w], timestamp=self.k * 20000))
        node._cb_odometry(NS(position=list(self.p), velocity=list(self.v)))
        node._cb_thrust(NS(xyz=[0.0, 0.0, -0.36]))
        node._cb_torque(NS(xyz=[0.01 * w, 0.0, 0.0]))


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run(node, plant, n_gt, t0_us=0, tick=True, pace_s=0.0, stop=None):
    """GT 스탬프 n_gt 개 발행: 플랜트 → 센서 → GT → (격자면) GPS → 틱. pace_s>0 이면 GT 사이 벽시계 대기(실시간 흉내)."""
    import time
    for i in range(1, n_gt + 1):
        us = t0_us + i * 20000
        plant.step(node)
        plant.feed_sensors(node)
        node._cb_gt(plant.gt_msg(us))
        if us % 100000 == 0:
            node._cb_gps(plant.gps_msg(us))
        if tick:
            node._tick()
        if pace_s:
            time.sleep(pace_s)
        if stop is not None and stop(node):
            return i
    return n_gt
