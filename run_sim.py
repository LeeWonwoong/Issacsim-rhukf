"""
run_sim.py — Isaac Sim + PX4 물리 엔진 구동
=============================================
역할:
  1. PhysX 물리 시뮬레이션 (250Hz)
  2. GPS 센서 퍼블리시 (10Hz) — 순수 노이즈만, 센서 공격 없음
  3. Ground Truth Odometry 퍼블리시 (250Hz)
  4. 액추에이터 공격 주입 (config attack_form: additive 가산바이어스[기본] | multiplicative 곱셈LoE, + Ramp)
  5. 환경 외란 (바람, 충돌)
  6. 에피소드 리셋 제어

Baro/Flow/Distance 센서 퍼블리시 제거 — UKF는 GPS+IMU만 사용.
센서 공격 (GPS FDI) 제거 — Actuator-only 위협 모델.
"""
import carb
import os
import argparse
from isaacsim import SimulationApp

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--headless', dest='headless', action='store_true')
_pre_parser.add_argument('--no-headless', dest='headless', action='store_false')
_pre_parser.add_argument('--px4-ns', dest='px4_ns', default='auto')
_pre_parser.add_argument('--speed', dest='speed', type=float, default=1.0,
                         help='sim 속도배율(>1=실시간보다 빠름; lockstep 한계까지)')
_pre_parser.set_defaults(headless=False)
_pre_args, _ = _pre_parser.parse_known_args()
simulation_app = SimulationApp({"headless": _pre_args.headless})

import time
import math
import json
import numpy as np
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig)
from pegasus.simulator.logic.vehicles.multirotor import (
    Multirotor, MultirotorConfig)
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
from px4_msgs.msg import SensorGps, VehicleThrustSetpoint, VehicleTorqueSetpoint

from swrl_config import compute_attack_ramp, compute_attack_forces
from env.ukf_filter import load_calibration


# ══════════════════════════════════════════════════════════════
#  바람 모델 (기존 px4.py에서 이식)
# ══════════════════════════════════════════════════════════════
class WindModel:
    def __init__(self, scenario='none', params=None):
        self.scenario = scenario
        p = params or {}
        self.rng = np.random.default_rng(int(time.time()) % 2**32)
        self.A, self.Cd, self.rho = 0.04, 1.28, 1.225
        self.ws = p.get('wind_speed', 5.0)
        self.wd = np.deg2rad(p.get('wind_dir', 0.0))
        self.gs = p.get('gust_start', 10.0)
        self.gd = p.get('gust_duration', 3.0)
        self.ti = p.get('turbulence_intensity', 0.5)
        self.tb = p.get('turbulence_bandwidth', 2.0)
        self._ts = np.zeros(3)
        # ── 랜덤 충격(반복 임펄스): 주행 중 돌풍/부딪힘. env SHOCK_WS>0 활성 (2026-08-21) ──
        #   POMDP 애매성용: benign 전이가 공격 온셋과 닮게 만듦. 바람 위에 가산.
        self._shock_ws = float(os.environ.get('SHOCK_WS', '0') or 0.0)   # 임펄스 등가풍속 [m/s]
        self._shock_hz = float(os.environ.get('SHOCK_HZ', '0.3') or 0.3) # 초당 발생확률
        self._shock_dur = float(os.environ.get('SHOCK_DUR', '0.4') or 0.4)
        self._shock_end = -1.0; self._shock_start = -1.0; self._shock_dir = np.zeros(3)
        # ── 바람 회전 모멘트 (env WIND_MOMENT_ARM>0): 바람력이 CP(COM 위 arm[m])에 작용 → 모멘트 ──
        #   τw = [−arm·Fy, arm·Fx, 0] = wind→ω 직접경로 (실기 선풍기 weathervaning, Dryden 회전성분).
        #   ★arm은 물리 기하로 고정. 회전 모멘트는 바람력 F(ws²·난류)에 비례 → ws 스윕만으로 병진+회전 동반증가. (2026-08-21)
        self._wind_moment_arm = float(os.environ.get('WIND_MOMENT_ARM', '0') or 0.0)  # CP 유효 레버 [m]

    def _drag(self, w):
        v = np.linalg.norm(w)
        if v < 1e-6:
            return np.zeros(3)
        return 0.5 * self.rho * v**2 * self.Cd * self.A * (w / v)

    def _shock(self, t, dt):
        """랜덤 충격 임펄스 (1-cos 펄스). SHOCK_WS>0일 때만."""
        if self._shock_ws <= 0.0:
            return np.zeros(3)
        # 활성 충격 없으면 확률적으로 새로 발생
        if t > self._shock_end and self.rng.random() < self._shock_hz * dt:
            self._shock_start = t; self._shock_end = t + self._shock_dur
            ang = self.rng.random() * 2 * np.pi
            self._shock_dir = np.array([np.cos(ang), np.sin(ang), 0.0])
        if self._shock_start <= t <= self._shock_end:
            V = (self._shock_ws / 2) * (1 - np.cos(2 * np.pi * (t - self._shock_start) / self._shock_dur))
            return self._drag(V * self._shock_dir)
        return np.zeros(3)

    def wind_moment(self, force):
        """바람력이 CP(COM 위 arm)에 작용해 생기는 roll/pitch 모멘트 [N·m]. 힘 F에 비례(∴ws²·난류)."""
        a = self._wind_moment_arm
        if a <= 0.0:
            return np.zeros(3)
        return np.array([-a * force[1], a * force[0], 0.0])   # τ = r_cp × F, r_cp=[0,0,a]

    def get_force(self, t, dt=0.004):
        d = np.array([np.cos(self.wd), np.sin(self.wd), 0.0])
        if self.scenario == 'wind_constant':
            base = self._drag(self.ws * d)
        elif self.scenario == 'wind_gust':
            if t < self.gs or t > self.gs + self.gd:
                base = np.zeros(3)
            else:
                V = (self.ws / 2) * (1 - np.cos(np.pi * (t - self.gs) / self.gd))
                base = self._drag(V * d)
        elif self.scenario == 'wind_turbulence':
            # OU(Ornstein-Uhlenbeck) 이산화. 노이즈계수는 sqrt(1-a^2)여야
            # 정상상태 std(ts)=ti*ws (=난류강도 정의)가 샘플레이트 무관하게 보존된다.
            # (기존 (1-a)는 고샘플레이트에서 std를 sqrt((1-a)/(1+a))≈0.063배로 압착 → 사실상 상수풍 버그)
            a = np.exp(-self.tb * dt)
            self._ts = a * self._ts + \
                       np.sqrt(1.0 - a * a) * self.ti * self.ws * self.rng.standard_normal(3)
            base = self._drag(self.ws * d + self._ts)
        else:
            base = np.zeros(3)
        return base + self._shock(t, dt)   # 랜덤 충격 가산 (SHOCK_WS>0시)

    def reconfigure(self, scenario, wind_speed):
        """에피소드마다 동적으로 바람 시나리오 변경"""
        self.scenario = scenario
        self.ws = wind_speed
        self._ts = np.zeros(3)


# ══════════════════════════════════════════════════════════════
#  메인 시뮬레이션 앱
# ══════════════════════════════════════════════════════════════
class PegasusApp:
    def __init__(self, args):
        self.args = args
        self.sim_time = 0.0
        self.physics_dt = 1.0 / 250.0
        self._step_us = int(round(self.physics_dt * 1e6))   # 4000 µs — SIMCLOCK_UKF 정수 스탬프 단위

        self.attack_active = False
        self.attack_type = 'none'
        self.attack_intensity = 0.0       # ramp 목표 강도(0~1). additive=바이어스 스케일 / mult=LoE 비율 α
        self.attack_start_time = 0.0
        self.attack_ramp_duration = 0.1

        # 공격 형태 + 가산 바이어스 크기 (online_rl_main이 /attack_config로 전달; 기본=가산)
        self.attack_form = 'additive'
        self.bias_torque_xy = 0.12
        self.bias_torque_z = 0.0
        self.bias_thrust_n = 2.0

        # 곱셈형 LoE용: PX4가 명령한 추력/토크 (u_ref) + calib 계수
        self.cmd_thrust = np.zeros(3)
        self.cmd_torque = np.zeros(3)
        self.calib = load_calibration('calibration.json')
        self._C_thrust = self.calib['C_thrust']
        self._C_torque_xy = self.calib['C_torque_xy']
        self._C_torque_z = self.calib['C_torque_z']

        self.last_gps_time = 0.0
        self.gps_noise_state = np.zeros(3)

        self.home_lat = 47.397742
        self.home_lon = 8.545594
        self.home_alt = 488.0
        self.earth_radius = 6371000.0

        self.needs_reset = False

        rclpy.init()
        self.ros_node = Node('sim_engine')
        # ★2026-09-23 SIMCLOCK_UKF=1 (노드 knob 이 sim_env 로 복사됨): GT·GPS 스탬프를 정수 step_counter×4000 µs 로,
        #   GPS 를 정수 게이트(step_counter%25==0, GT 블록 안·같은 state)로, 두 토픽을 RELIABLE·KEEP_LAST 50 으로.
        #   노드가 스탬프 순서대로 UKF 를 드레인한다(GPS 당 예측 5·RL 간격 sim 0.100 s 고정). 0 이면 구 동작 그대로.
        self._simclock = os.environ.get('SIMCLOCK_UKF', '0') not in ('', '0')
        if self._simclock:
            _sc_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                                 history=HistoryPolicy.KEEP_LAST, depth=50)
            self.gt_pub = self.ros_node.create_publisher(Odometry, '/gt/odometry', _sc_qos)
            self.pub_gps = self.ros_node.create_publisher(SensorGps, '/sim/sensor_gps', _sc_qos)
            carb.log_warn("[run_sim] SIMCLOCK_UKF=1: GT/GPS 정수 스탬프(step×4000µs)·GPS step%25·RELIABLE/50")
        else:
            self.gt_pub = self.ros_node.create_publisher(
                Odometry, '/gt/odometry', 10)
            self.pub_gps = self.ros_node.create_publisher(
                SensorGps, '/sim/sensor_gps', 10)

        self.ros_node.create_subscription(
            String, '/attack_config', self._cb_attack_config, 10)
        self.ros_node.create_subscription(
            String, '/scenario_config', self._cb_scenario_config, 10)
        self.ros_node.create_subscription(
            String, '/sim_control', self._cb_sim_control, 10)

        # PX4가 명령한 actuator setpoint 구독 (곱셈형 LoE = -α·u_ref 주입용)
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        _ns = self._resolve_ns(_pre_args.px4_ns)   # 'auto'면 자동감지(best-effort), 아니면 그대로
        self.ros_node.create_subscription(
            VehicleThrustSetpoint, f'{_ns}/fmu/out/vehicle_thrust_setpoint',
            self._cb_thrust, px4_qos)
        self.ros_node.create_subscription(
            VehicleTorqueSetpoint, f'{_ns}/fmu/out/vehicle_torque_setpoint',
            self._cb_torque, px4_qos)
        carb.log_warn(f"[run_sim] PX4 namespace = '{_ns}' (thrust/torque 구독)")

        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # ══════════════════════════════════════════════════════════
        #  ★ 실기 제어기 제한값을 SITL 에 주입 (2026-08-07)
        # ══════════════════════════════════════════════════════════
        #  왜: sim 이 실기보다 **훨씬 빠르고 급격하게** 난다. MC_* 게인은 같은데
        #      MPC_* 속도·가속 제한이 통째로 다르기 때문이다(실측 9개 항목).
        #        MPC_XY_VEL_MAX  실기 2.0  vs  PX4기본 12.0   ← 6배
        #        MPC_VEL_MANUAL  실기 2.0  vs        10.0    ← 5배 (수동조종 감각도 다름)
        #      이 상태로는 어떤 궤적을 써도 sim/실기가 갈리고, Isaac 수동조종 검증도
        #      실기와 다른 감각이 된다. **sim 을 실기에 맞춘다**(sim2real 방향).
        #  방법: PX4 posix rcS 의 PX4_PARAM_* 환경변수 오버라이드.
        #        rcS 가 에어프레임 기동보다 **먼저** 읽으므로 안전하다.
        #        Pegasus 는 os.environ 을 그대로 PX4 프로세스에 넘긴다
        #        (px4_launch_tool.py:44,63).
        #  출처: px4_field/PX4_6.params (실기 6호기 실측 덤프)
        #  ⚠ 값을 바꾸면 플랜트가 아니라 **폐루프**가 바뀐다 → 밴드 재측정 대상.
        _PX4_LIMITS = {
            # ★★ 2026-08-13: 2.5 로 올렸다가 **2.0 으로 되돌림**.
            #   속도 스파이크의 원인은 상한이 아니라 **궤적 불연속**이었다(aggressive phase 경계
            #   2.24/4.48 m 순간이동, waypoint 레그당 고정 4.0s → 10 m 레그가 2.50 m/s).
            #   둘 다 궤적 쪽에서 고쳤으므로 전 패턴 ≤1.41 m/s → 2.0 대비 여유 30%.
            #   상한을 올려도 패턴 속도는 안 바뀐다(반경·각속도가 정한다: R2.8×ω0.5 = 1.40 m/s).
            #   바뀌는 것은 **오차 보정 여유**뿐: (2.5−1.40)/2.5 = 44%  (구 설정은 14% 였다)
            #   08-12 실기에서 명령속도가 2.00 에 상시 포화 → 뒤처져도 못 따라잡아 RMSE 1.4~2.9 m.
            #   ∴ 상한↑ 는 RMSE 를 **낮춘다**. NIS 는 속도와 무관(구간별 실측 확인).
            'MPC_XY_VEL_MAX':   2.0,    # 수평 최대속도  (기본 12.0)
            'MPC_XY_CRUISE':    2.0,    # 순항속도       (기본  5.0)
            'MPC_VEL_MANUAL':   2.0,    # 수동조종 속도  (기본 10.0)
            'MPC_XY_VEL_ALL':   2.0,    # 수평속도 일괄키 — 실제 클램프값. 개별키를 덮어쓴다
            'MPC_Z_VEL_ALL':    3.0,    # 수직속도 일괄키(기본  -3.0 = 미사용). ★>0 이면 아래 UP/DN 을 이 값으로 덮어씀
            'MPC_Z_VEL_MAX_UP': 3.0,    # 상승 최대속도  (실기와 동일. _ALL 이 어차피 3.0 으로 강제하나 명시)
            'MPC_Z_VEL_MAX_DN': 2.25,   # 하강 최대속도  (기본  1.5)
            'MPC_Z_V_AUTO_UP':  3.0,    # 자동 상승속도  (실기 3.0)
            'MPC_Z_V_AUTO_DN':  2.25,   # 자동 하강속도  (기본  1.5)
            'MPC_TKO_SPEED':    1.8,    # 이륙속도       (기본  1.5)
            # ★2026-09-07 실기(PX4_6.params) 정합 추가 — jerk·게인 미주입이었다(가감속 du/dt=aliasing 불일치 원인).
            'MPC_ACC_HOR_MAX':  5.0,    # 실기 5.0
            'MPC_JERK_AUTO':    4.0,    # 실기 4.0 (가감속 급격함 = 기동 aliasing 강도 결정)
            'MPC_JERK_MAX':     8.0,    # 실기 8.0
            'MPC_XY_P':         0.95,   # 실기 0.95 (위치 P게인)
            'MPC_XY_VEL_P_ACC': 1.8,    # 실기 1.8 (속도 P게인)
            'MPC_LAND_SPEED':   1.5,    # 착륙속도       (기본  0.7)
            # ★ 추력곡선 정합 (2026-08-11): Iris 는 T∝ω²(곡률 1.68) 이라 호버 0.56.
            #   실기 Q250 은 거의 선형(곡률 0.93)·호버 0.33. THR_MDL_FAC 로 선형화해 맞춘다.
            #   T/W(3.16)는 이미 실기(~3)와 비슷 → 곡선만 선형화하면 호버가 ~0.33 으로 내려온다.
            # ★ env THR_MDL_FAC 로 튜닝(호버 정합). 0.9→호버0.295, 낮추면 호버↑(0.78→~0.33).
            'THR_MDL_FAC':      float(os.environ.get('THR_MDL_FAC', '0.78')),
            # ★ PX4 오토튠 활성(검증용). APPLY=0=로깅전용(게인 안바뀜) → 항상 켜도 안전.
            #   MC_AT_START=1 을 autotune_go.py 로 쏴야 실제 실행. τon 상태 G_DC 검증용.
            'MC_AT_EN':         1,
            'MC_AT_APPLY':      0,
            'MC_AT_SYSID_AMP':  0.7,
            # ★★ 2026-08-13: ATK_EN=1 — 이게 없어서 **모든 headless 스윕이 무공격이었다.**
            #   apply_attack() 은 ATK_EN=0 ∨ !armed 면 즉시 리턴한다(기본 0=stock).
            #   08-09 SITL 검증은 pxh 에서 손으로 켠 것이고, 스윕은 매번 fresh SITL 이라 0.
            #   증거: δ 0.46 → 2.29(권한 229%) 로 5배 올려도 NIS p50/p95·max_roll·생존율 불변.
            #   RC 경로는 headless 에 manual_control_setpoint 가 없어 자동으로 기여 0 이다.
            'ATK_EN':           1,
            # ★2026-09-02 수평가속 한계 env 노브 (기동 회전 sharpness 의 실제 다이얼).
            #   PX4 기본 = 실기 정합 3.0. 상향 시 실기도 같은 값으로 재정합해서 날 것.
            'MPC_ACC_HOR':      float(os.environ.get('MPC_ACC_HOR', '3.0')),
            # ★2026-09-08 B′실험: rate 적분기 한계(=지속 토크바이어스 트림 상한). 기본 0.3(실기 동일).
            #   INT_LIM 상향 → PX4 가 δ~1.0 까지 트림 가능 = "호버가 전 권한을 자세안정화에" 경로.
            'MC_RR_INT_LIM':    float(os.environ.get('MC_INT_LIM', '0.30')),
            'MC_PR_INT_LIM':    float(os.environ.get('MC_INT_LIM', '0.30')),
            # ★2026-09-08 C실험(제어권한 상향): 위치루프 보정 한계 — 공격유발가속 7.6 대비 기본 3.0 이
            #   호버 사수 실패의 병목. TILTMAX 상향 = counter-tilt 명령 허용폭.
            'MPC_TILTMAX_AIR':  float(os.environ.get('MPC_TILTMAX', '45.0')),
        }
        # ★ 2026-08-17 진단용: SPEED_SCALE 로 수평속도 캡 상향(궤적 omega 도 online_rl 에서 같은 배율).
        _spd_scale = float(os.environ.get('SPEED_SCALE', '1.0'))
        if _spd_scale != 1.0:
            for _vk in ('MPC_XY_VEL_MAX', 'MPC_XY_CRUISE', 'MPC_VEL_MANUAL', 'MPC_XY_VEL_ALL'):
                _PX4_LIMITS[_vk] = round(_PX4_LIMITS[_vk] * _spd_scale, 3)
            carb.log_warn(f'[SPEED_SCALE] 수평속도 캡 ×{_spd_scale} → MPC_XY_VEL_ALL={_PX4_LIMITS["MPC_XY_VEL_ALL"]}')
        # ★2026-08-27 PX4_VEL_CAP: 수평속도캡만 상향(mission omega 무관 — online_rl 은 안 건드림).
        #   높은 캡 ≠ 빠른 비행(궤적 setpoint 가 속도 결정). 캡은 '필요할 때 빠른 보정'만 허용.
        #   → 임무는 gentle, failsafe 호버만 여유 권한으로 hijack 반격.
        _vcap = float(os.environ.get('PX4_VEL_CAP', '0') or 0)
        if _vcap > 0:
            for _vk in ('MPC_XY_VEL_MAX', 'MPC_XY_CRUISE', 'MPC_VEL_MANUAL', 'MPC_XY_VEL_ALL'):
                _PX4_LIMITS[_vk] = _vcap
            carb.log_warn(f'[PX4_VEL_CAP] 수평속도캡 → {_vcap} (mission 궤적속도 불변)')
        for _k, _v in _PX4_LIMITS.items():
            os.environ[f'PX4_PARAM_{_k}'] = str(_v)
        carb.log_warn('[PX4LIM] 실기 제한값 주입: ' +
                      ', '.join(f'{k}={v}' for k, v in _PX4_LIMITS.items()))

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        config_multirotor = MultirotorConfig()

        # ★★ 로터/항력 정합 (2026-08-12).  ⚠ **여기가 진짜 설정 지점이다.**
        #   run_sim 은 IrisConfig 가 아니라 MultirotorConfig() 를 직접 만든다.
        #   그래서 iris.py 를 고쳐도 **전혀 반영되지 않는다**(구 ROTOR_K 편집도 죽은 코드였다).
        #   MultirotorConfig 기본값: rotor_constant 8.54858e-6, rolling_moment_coefficient 1e-6,
        #                            LinearDrag([0.50, 0.30, 0.0])
        _rk = float(os.environ.get('ROTOR_K', '8.54858e-6'))   # 추력 T = k·ω². T/W 3.15 ≈ 실기 3.0
        # 요 권한: τ_z = Σ c_m·ω²·dir  → C_torque_z 는 c_m 에 선형.
        #   sim 실측(로터로그 회귀, c_m=1e-6): C_torque_z ≈ 1.92   목표(실기 오토튠) 1.378
        #   → c_m = 1e-6 × 1.378/1.92 = 7.2e-7
        #   ※ 구 문서의 "sim C_torque_z 4.41" 은 spin-up 오염값이었다. 직접 회귀는 ~1.9.
        _cm = float(os.environ.get('ROTOR_CM', '7.2e-7'))
        # 병진 항력: 실기 F4(등속법) x 0.352 / y 0.397. 구 [0.50,0.30] 은 Iris 기본값(미검증)이고
        #   비등방 방향이 실기와 반대였다. ⚠ x 는 가속법과 갈려 미확정 — 무풍일 재측정 대상.
        _dg = [float(v) for v in os.environ.get('DRAG_XYZ', '0.35,0.40,0.0').split(',')]
        config_multirotor.thrust_curve = QuadraticThrustCurve({
            'rotor_constant': [_rk] * 4,
            'rolling_moment_coefficient': [_cm] * 4,
        })
        config_multirotor.drag = LinearDrag(_dg)
        carb.log_warn(f"[ROTOR] rotor_k={_rk:.5e}  rolling_moment_c={_cm:.5e}  drag={_dg}")
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0, "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        self.vehicle = Multirotor(
            "/World/quadrotor", ROBOTS['Iris'], 0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config_multirotor)

        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()

        self.wind = WindModel('none')
        self._wind_dbg = os.environ.get('WIND_DBG', '') not in ('', '0')
        # 바람 회전 gust는 WindModel.get_torque (env WIND_ROT_SIGMA) — Dryden 회전성분.

        # ── 로터 캘리브레이션 로그 (env ROTOR_LOG=경로.npz 로 활성) ──
        #   PX4 정규화 명령(thrust/torque setpoint)과 **실제 적용된 로터 각속도**를 함께 기록.
        #   로터 각속도를 알면 적용 추력/토크가 Pegasus 모델(T=k·ω², τ_z=Σc·ω²·dir, τ_xy=Σ T×r)
        #   로 정확히 계산되므로, 동역학·폐루프 편향 없는 **정적** C_thrust/C_torque 적합이 가능하다.
        #   (자세 rate loop 폐루프에서 ω̇~τ 회귀는 편향된다 — 2026-07-28 진단)
        self._rotor_log_path = os.environ.get('ROTOR_LOG', '')
        self._rotor_rows = []

        # ── 모터 1차 지연 (env, 미설정/0 = 비활성) ──
        #   MOTOR_TAU=초        : 대칭(상승=하강). 하위호환 별칭.
        #   MOTOR_TAU_UP/DOWN=초: 비대칭. 실기는 스핀업(모터 토크)보다 스핀다운(프로펠러
        #                         공력항력만, 액티브 브레이킹 없으면)이 2~3배 느리다.
        # ★★ 기본값 0.023→0.014 (2026-08-11 오토튠 캘리브): MOTOR_TAU 는 유효지연을 **2배**로 만든다
        #   (up+down 양방향, τ_eff≈4ms커플링+2×MOTOR_TAU). 실기 오토튠 a2=0.67(τ~23ms) 를
        #   sim 오토튠으로 맞춘 값이 0.014 → sim roll a2=0.663 (실기 0.67, 1%이내). 검증:
        #     MOTOR_TAU  0→a2 0.153 / 0.003→0.435 / 0.014→0.663✓ / 0.023→0.853(2배과다).
        #   0.023 은 τ_eff 50ms=실기 2배 → 오토튠 G_DC 를 불신영역으로 밀었었다. (끄려면 MOTOR_TAU=0)
        _tau_sym = float(os.environ.get('MOTOR_TAU', '0.014') or 0.0)
        _tau_up = float(os.environ.get('MOTOR_TAU_UP', '0') or 0.0) or _tau_sym
        _tau_dn = float(os.environ.get('MOTOR_TAU_DOWN', '0') or 0.0) or _tau_sym
        self._install_motor_lag(_tau_up, _tau_dn)

        self.body_view = None
        self.rotor_view = None
        self._setup_body_view()
        self._apply_body_calib('[init]')   # 플랜트 질량/관성 = calibration.json (UKF와 동일)

        # ── GUI 보기용: chase-cam + 컬러 조명 (headless엔 무영향, 실패해도 무해) ──
        self._cam_follow = (not args.headless)
        self._cam_offset = np.array([-10.0, -10.0, 10.0])   # 드론 뒤·위 오프셋(ENU)
        self._cam_eye = None
        self._add_colored_lights()

        self.stop_sim = False

    def _cb_attack_config(self, msg):
        try:
            cfg = json.loads(msg.data)
            self.attack_active = cfg['active']
            self.attack_type = cfg.get('type', 'none')
            self.attack_intensity = float(cfg.get('intensity', 0.0))
            self.attack_ramp_duration = float(cfg.get('ramp_duration', 0.1))
            self.attack_form = cfg.get('form', 'additive')
            self.bias_torque_xy = float(cfg.get('bias_torque_xy', self.bias_torque_xy))
            self.bias_torque_z = float(cfg.get('bias_torque_z', self.bias_torque_z))
            self.bias_thrust_n = float(cfg.get('bias_thrust_n', self.bias_thrust_n))
            if self.attack_active:
                self.attack_start_time = self.sim_time
            carb.log_warn(f"[ATTACK] {cfg}")
        except Exception as e:
            carb.log_error(f"Attack config parse error: {e}")

    def _cb_scenario_config(self, msg):
        try:
            cfg = json.loads(msg.data)
            self.wind.reconfigure(
                cfg.get('disturbance_type', 'none'),
                cfg.get('wind_speed', 0.0))
            carb.log_warn(f"[SCENARIO] Wind: {cfg}")
        except Exception as e:
            carb.log_error(f"Scenario config parse error: {e}")

    def _cb_sim_control(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'reset':
            self.needs_reset = True
            carb.log_warn("[SIM] Reset requested")

    def _cb_thrust(self, msg):
        self.cmd_thrust[:] = msg.xyz[:3]

    def _cb_torque(self, msg):
        self.cmd_torque[:] = msg.xyz[:3]

    def _resolve_ns(self, configured):
        """'auto'면 ROS 그래프에서 '*/fmu/out/vehicle_odometry'의 살아있는 ns를 best-effort 감지.
        run_sim __init__ 시점엔 PX4가 막 떠 등록 전일 수 있어 폴백 가능(공격주입용이라 GT엔 무관)."""
        if configured != 'auto':
            return configured.rstrip('/')
        suffix = '/fmu/out/vehicle_odometry'
        for _ in range(20):   # ~6s best-effort (run_sim 기동 지연 최소화)
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            topics = self.ros_node.get_topic_names_and_types()
            live = [t[:-len(suffix)] for t, _ in topics
                    if t.endswith(suffix) and self.ros_node.count_publishers(t) > 0]
            if live:
                return sorted(live)[0]
            time.sleep(0.2)
        carb.log_warn("[run_sim] px4 ns auto 감지 실패 → bare '/fmu' 폴백(공격주입만 영향, GT 무관)")
        return ''

    def _setup_body_view(self):
        for path in ["/World/quadrotor/body", "/World/quadrotor"]:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                self.body_path = path
                try:
                    self.body_view = RigidPrimView(
                        prim_paths_expr=path, name="attack_body")
                    self.world.scene.add(self.body_view)
                    self.body_view.initialize()
                except Exception:
                    pass
                break

    def _rotor_compensation(self):
        """로터 4개(리볼루트 조인트로 바디에 붙은 별개 강체)의 질량/관성 기여를 반환.

        Iris USD 는 rotor0~3 이 각각 RigidBody 라서 실제 비행질량 = body + Σrotor 다
        (기본 에셋: 1.5 + 0.1186 = 1.6186kg — '기본값 1.5kg'은 바디만의 값).
        따라서 '총 비행질량 = 1.372kg' 을 맞추려면 바디에 (1.372 - Σrotor) 를 써야 한다.
        관성도 동일: 바디 COM 기준 평행축 정리로 로터 기여를 빼준다.
        반환: (Σm_rotor, I_contrib[3])
        """
        try:
            if getattr(self, 'rotor_view', None) is None:
                self.rotor_view = RigidPrimView(
                    prim_paths_expr="/World/quadrotor/rotor[0-3]", name="calib_rotors")
                self.world.scene.add(self.rotor_view)
                self.rotor_view.initialize()
            rm = np.asarray(self.rotor_view.get_masses(), dtype=float).ravel()
            ri = np.asarray(self.rotor_view.get_inertias(), dtype=float).reshape(len(rm), 9)
            rp, _ = self.rotor_view.get_world_poses()
            bp, _ = self.body_view.get_world_poses()
            rel = np.asarray(rp, dtype=float).reshape(len(rm), 3) - \
                np.asarray(bp, dtype=float).reshape(3)
            I = ri[:, [0, 4, 8]].sum(axis=0)              # 로터 자체 관성(작음)
            for i in range(len(rm)):
                x, y, z = rel[i]
                I = I + rm[i] * np.array([y*y + z*z, x*x + z*z, x*x + y*y])   # 평행축
            # ★ 로터 중심(바디 원점 기준). COM 정합에 쓴다 — 아래 _apply_body_calib 참조.
            self._rotor_centroid = rel.mean(axis=0)
            return float(rm.sum()), I
        except Exception as e:
            carb.log_error(f"[MASS] 로터 기여 계산 실패(보정 없이 진행): {e}")
            return 0.0, np.zeros(3)

    def _apply_body_calib(self, tag=''):
        """물리 바디의 질량/관성을 calibration.json(drone)에 강제 정렬.

        calibration.json 의 drone 블록 = **총 비행질량/총 관성**(로터 포함, UKF가 모델링하는 대상).
        Iris USD 기본값은 body 1.5kg + rotor 0.1186kg = 1.6186kg 이었고, UKF 는 1.372kg 를
        쓰고 있었다 → 플랜트와 탐지기 모델이 어긋난 채로 돌아갔다(2026-07-28 발견).
        여기서 body 에 (총목표 - 로터기여) 를 써서 **총합이 목표와 일치**하도록 만든다.
        USD 어트리뷰트와 PhysX 뷰 양쪽에 써 넣고 읽어서 로그로 확인한다.

        ⚠ 질량이 바뀌면 호버 동작점(u_norm)이 이동하므로 C_thrust/C_torque/drag 재캘리브레이션이
          **반드시 함께** 가야 한다(안 하면 정합이 오히려 악화).
        """
        d = self.calib['drone']
        m_tot = float(d['mass'])
        I_tot = np.array([float(d['Ixx']), float(d['Iyy']), float(d['Izz'])])
        path = getattr(self, 'body_path', '/World/quadrotor/body')

        m_r, I_r = self._rotor_compensation()
        m_body = m_tot - m_r
        I_body = I_tot - I_r
        if m_body <= 0.0 or np.any(I_body <= 0.0):
            carb.log_error(f"[MASS] 로터 보정 후 값이 비물리적 "
                           f"(m_body={m_body:.4f}, I_body={I_body}) → 보정 생략")
            m_body, I_body = m_tot, I_tot
            m_r, I_r = 0.0, np.zeros(3)

        # (1) USD 어트리뷰트 (리셋/재파싱 후에도 유지되는 소스)
        try:
            prim = self.stage.GetPrimAtPath(path)
            mass_api = (UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI)
                        else UsdPhysics.MassAPI.Apply(prim))
            mass_api.CreateMassAttr().Set(float(m_body))
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in I_body]))
            self._apply_com_align(mass_api, tag, m_body / m_tot)
        except Exception as e:
            carb.log_error(f"[MASS] USD 어트리뷰트 설정 실패: {e}")

        # (2) PhysX 런타임 뷰 (이미 파싱된 바디에 즉시 반영)
        if self.body_view is not None:
            try:
                self.body_view.set_masses(np.array([m_body], dtype=np.float32))
                self.body_view.set_inertias(np.array(
                    [[I_body[0], 0, 0, 0, I_body[1], 0, 0, 0, I_body[2]]], dtype=np.float32))
            except Exception as e:
                carb.log_error(f"[MASS] PhysX 뷰 설정 실패: {e}")

        # (3) 읽어서 확인 — 로그는 '실효 총합' 기준(목표와 직접 비교 가능)
        try:
            rb_m = float(np.asarray(self.body_view.get_masses()).ravel()[0])
            rb_I = np.asarray(self.body_view.get_inertias()).ravel()[[0, 4, 8]]
            eff_m = rb_m + m_r
            eff_I = rb_I + I_r
            ok = abs(eff_m - m_tot) < 1e-3 and np.all(np.abs(eff_I - I_tot) < 1e-5)
            lvl = carb.log_warn if ok else carb.log_error
            lvl(f"[MASS]{tag} {'OK' if ok else 'MISMATCH'} path={path} "
                f"body=({rb_m:.6f}kg, {rb_I[0]:.6f},{rb_I[1]:.6f},{rb_I[2]:.6f}) "
                f"+rotor=({m_r:.6f}kg, {I_r[0]:.6f},{I_r[1]:.6f},{I_r[2]:.6f}) "
                f"= 총({eff_m:.6f}kg, {eff_I[0]:.6f},{eff_I[1]:.6f},{eff_I[2]:.6f}) "
                f"목표({m_tot:.6f}kg, {I_tot[0]:.6f},{I_tot[1]:.6f},{I_tot[2]:.6f})")
        except Exception as e:
            carb.log_error(f"[MASS] readback 실패: {e}")

    def _apply_com_align(self, mass_api, tag='', mass_frac=1.0):
        """★ 추력-토크 결합(torque_thrust_coupling)을 실기와 같게 만든다 — 바디 COM 이동.

        왜 필요한가
          로터 추력의 작용점이 COM 에서 어긋나 있으면 **총추력이 상시 토크**를 만든다.
          PX4 는 트림으로 상쇄하지만, UKF 는 그 트림을 진짜 토크로 오해해 평시 잔차가 남는다.
          calibration.json 의 `torque_thrust_coupling`(K, N·m per N) 이 그 항을 UKF 에 알려준다.
          그런데 K 는 **기하 상수**라, UKF 에 실기 K 를 넣었으면 sim 플랜트 기하도 같아야 한다.
          안 그러면 sim 에서만 보정이 틀려 평시 NIS 가 반대로 올라간다(2026-08-12 실측 확인).

        수식 (바디 FRD, 추력은 -z)
          τ = r̄ × F,  F = (0,0,-T),  r̄ = 로터중심 − COM
            τ_x = −T·ȳ  →  K_roll  = −ȳ
            τ_y = +T·x̄  →  K_pitch = +x̄

        ⚠ 좌표계 (2026-08-12 실측으로 잡은 함정)
          `_rotor_compensation` 의 로터중심 c 는 **월드 ENU = 스폰 자세에서 바디 FLU** 이고,
          위 유도는 **FRD** 다. FRD y = −FLU y → **y 항의 부호가 뒤집힌다.**
          USD centerOfMass 도 바디 로컬(FLU) 이므로 최종식은
              COM_FLU = ( c_x − K_pitch,  c_y − K_roll,  c_z )
          (부호를 +K_roll 로 뒀다가 롤 트림이 실기와 반대로 나왔다 — 실측으로 확인)

        ⚠ 로터 질량 희석 (2026-08-12)
          로터 4개가 **별개 강체**라, 바디 COM 을 옮겨도 **시스템 COM** 은 덜 움직인다:
              r̄_system = (m_body / m_total) · r̄_body      (≈0.911)
          그래서 목표 r̄ 를 mass_frac 으로 나눠 과보정한다.

        env COM_ALIGN=0 이면 건너뛴다(구 동작 유지).
        ⚠ 적용 후 sim 호버 트림을 재서 실기 트림(τ_norm ≈ [-0.0073, -0.0554, +0.0042])과
          대조할 것. 부호를 뒤집기 쉬운 항이라 반드시 실측으로 확인한다.
        """
        if os.environ.get('COM_ALIGN', '1') == '0':
            return
        K = self.calib.get('torque_thrust_coupling')
        c = getattr(self, '_rotor_centroid', None)
        if not K or c is None:
            carb.log_warn(f"[COM]{tag} 건너뜀 (K={bool(K)}, centroid={c is not None})")
            return
        f = max(float(mass_frac), 1e-3)
        com = np.array([c[0] - float(K[1]) / f, c[1] - float(K[0]) / f, c[2]], dtype=float)
        try:
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in com]))
        except Exception as e:
            carb.log_error(f"[COM]{tag} USD 설정 실패: {e}")
        # ★★ USD 어트리뷰트만으로는 **이미 파싱된 바디에 안 먹는다**(2026-08-12 실측:
        #   피치는 절반만 이동, 롤은 전혀 안 움직였다). 질량/관성처럼 PhysX 런타임 뷰에도 써야 한다.
        rb = None
        try:
            if self.body_view is not None and hasattr(self.body_view, 'set_coms'):
                self.body_view.set_coms(np.array([com], dtype=np.float32))
                rb = np.asarray(self.body_view.get_coms()[0]).ravel()[:3]
        except Exception as e:
            carb.log_error(f"[COM]{tag} PhysX 뷰 설정 실패: {e}")
        carb.log_warn(f"[COM]{tag} 로터중심=({c[0]:+.5f},{c[1]:+.5f},{c[2]:+.5f}) "
                      f"→ COM=({com[0]:+.5f},{com[1]:+.5f},{com[2]:+.5f}) m  "
                      f"목표 K=({K[0]:+.5f},{K[1]:+.5f}) N·m/N  희석보정 /{f:.3f}"
                      + (f"  readback=({rb[0]:+.5f},{rb[1]:+.5f},{rb[2]:+.5f})" if rb is not None
                         else "  ⚠ readback 없음(set_coms 미지원?)"))

    def _install_motor_lag(self, tau_up, tau_down=None):
        """로터 명령에 1차 지연을 넣는다 (실기 ESC+모터 시상수 모사). 둘 다 <=0 이면 비활성.

        Pegasus 의 QuadraticThrustCurve 는 명령을 **즉시** 로터 각속도로 적용한다
        (원 소스 주석: "instanenous model - no delay introduced"). 실기는 ESC+모터가
        1차 지연(τ≈20~40ms)을 가지므로, 이걸 빼놓으면 제어권한의 동적 한계가 낙관적으로
        나오고 밴드가 실제보다 관대하게 측정된다.

        ★ 비대칭 (2026-07-29): 실기는 상승과 하강의 시상수가 다르다.
          - 스핀업   = 모터가 전기 토크로 능동 가속        → 빠르다
          - 스핀다운 = 프로펠러 공력항력에만 의존
                       (ESC 액티브 브레이킹/damped light 없으면) → 보통 τ_down ≈ 2~3·τ_up
          이 프로젝트에 직결되는 이유: bias 주입은 **일부 로터 up / 일부 로터 down** 이고,
          온셋 엣지(t=1~2 스파이크 = RL 정당화의 핵심 근거)는 **느린 쪽이 지배**한다.
          대칭 τ 하나로는 온셋 모양이 실기와 다르다.
        ⚠ τ 는 RPM 의존이다(τ ≈ J_prop/(2·c_d·ω₀)). 상수 τ 는 한 동작점의 선형화이므로
          실측은 **호버 RPM 근방**에서 할 것 — 평시 NIS 기준선을 지배하는 동작점이므로.
        ⚠ ESC 제어모드를 먼저 동결하고 실측할 것. τ_down 이 거기 좌우된다.

        구현: thrusters 인스턴스의 두 메서드만 감싼다. 백엔드가 쓰는 원 명령은 `_raw_ref` 로
        따로 보관하고, update() 직전에 지연 상태를 전진시켜 `_input_reference` 에 넣는다
        (원 값과 필터 출력을 분리해 이중 필터링을 방지).
        τ 값은 실기 ESC 텔레메트리(RPM 계단응답)에서 실측해 넣는 것이 목표다.
        """
        tau_up = float(tau_up or 0.0)
        tau_down = float(tau_up if tau_down is None else (tau_down or 0.0))
        if tau_up <= 0.0 and tau_down <= 0.0:
            self._motor_tau_up = self._motor_tau_down = 0.0
            self._motor_tau = 0.0
            return
        # 한쪽만 준 경우 나머지는 같은 값으로 (대칭 폴백)
        if tau_up <= 0.0:
            tau_up = tau_down
        if tau_down <= 0.0:
            tau_down = tau_up

        tc = self.vehicle._thrusters
        n = len(tc._input_reference)
        tc._raw_ref = list(tc._input_reference)
        tc._lag_state = np.asarray(tc._input_reference, dtype=float).copy()
        _orig_set = tc.set_input_reference
        _orig_update = tc.update

        def _set_ref(ref):
            tc._raw_ref = list(ref)

        def _update(state, dt):
            raw = np.asarray(tc._raw_ref, dtype=float)
            st = tc._lag_state
            # 로터별로 상승/하강을 판정해 서로 다른 시상수를 적용
            tau = np.where(raw >= st, tau_up, tau_down)
            a = dt / (tau + dt)                      # 1차 이산화 (로터별 벡터)
            st += a * (raw - st)
            tc._input_reference = list(st)
            return _orig_update(state, dt)

        tc.set_input_reference = _set_ref
        tc.update = _update
        self._motor_tau_up = tau_up
        self._motor_tau_down = tau_down
        self._motor_tau = tau_up                     # 하위호환 필드
        self._orig_set_input_reference = _orig_set   # 참조 유지
        _sym = '대칭' if abs(tau_up - tau_down) < 1e-12 else '비대칭'
        carb.log_warn(f"[MOTOR] 1차 지연 활성({_sym}) τ_up={tau_up*1000:.1f}ms "
                      f"τ_down={tau_down*1000:.1f}ms "
                      f"(로터 {n}개, 물리 dt={self.physics_dt*1000:.1f}ms)")

    def _log_rotors(self, step_counter):
        """PX4 정규화 명령 + 실제 적용 로터 각속도를 기록(정적 캘리브레이션용).
        run_sim 은 SIGKILL 로 종료되므로 주기적으로 덮어써 저장한다."""
        try:
            vel = np.asarray(self.vehicle._thrusters.velocity, dtype=float)
            if vel.size < 4 or float(np.max(vel)) <= 0.0:
                return                       # 아직 시동 전
            st = self.vehicle.state
            self._rotor_rows.append(np.concatenate([
                [self.sim_time], self.cmd_thrust[:3], self.cmd_torque[:3], vel[:4],
                np.asarray(st.linear_velocity, dtype=float),      # ENU
                np.asarray(st.angular_velocity, dtype=float),     # 바디
                [float(self.attack_active)],
            ]))
            if step_counter % 2500 == 0 and self._rotor_rows:
                arr = np.asarray(self._rotor_rows, dtype=np.float64)
                np.savez(self._rotor_log_path, data=arr, dt=self.physics_dt,
                         cols='t,cmd_thr_x,cmd_thr_y,cmd_thr_z,cmd_tq_x,cmd_tq_y,cmd_tq_z,'
                              'w0,w1,w2,w3,vE,vN,vU,wx,wy,wz,attack')
                carb.log_warn(f"[ROTOR] {self._rotor_log_path} 저장 (rows={len(arr)})")
        except Exception as e:
            carb.log_error(f"[ROTOR] 로깅 실패(1회 후 비활성): {e}")
            self._rotor_log_path = ''

    def _update_chase_cam(self, drone_pos):
        """GUI 뷰포트 카메라가 드론을 부드럽게 추적(chase-cam). 실패 시 1회 경고 후 비활성."""
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
            target = np.asarray(drone_pos, dtype=float)
            desired_eye = target + self._cam_offset
            if self._cam_eye is None:
                self._cam_eye = desired_eye.copy()
            else:
                self._cam_eye = 0.85 * self._cam_eye + 0.15 * desired_eye   # 저역통과(부드럽게)
            set_camera_view(self._cam_eye.tolist(), target.tolist(),
                            camera_prim_path="/OmniverseKit_Persp")
        except Exception as e:
            carb.log_warn(f"[chase-cam] 비활성화: {e}")
            self._cam_follow = False

    def _add_colored_lights(self):
        """무대 조명을 컬러로. (1) 기존 dome/distant 틴트 + (2) 컬러 sphere light 추가.
        Isaac 버전마다 intensity/속성명이 달라 전부 try/except로 감쌈(실패해도 무해)."""
        try:
            from pxr import UsdLux
            stage = self.stage
            # (1) 기존 조명 살짝 틴트
            for prim in stage.Traverse():
                t = prim.GetTypeName()
                if t in ("DomeLight", "DistantLight"):
                    try:
                        UsdLux.LightAPI(prim).CreateColorAttr().Set(Gf.Vec3f(0.55, 0.65, 1.0))
                    except Exception:
                        pass
            # (2) 컬러 sphere light 추가 (위치/색/밝기는 취향껏 조정)
            specs = [
                ("/World/StageLights/Red",   (0.0, 0.0, 9.0),    (1.0, 0.15, 0.15)),
                ("/World/StageLights/Blue",  (8.0, 8.0, 9.0),    (0.2, 0.3, 1.0)),
                ("/World/StageLights/Green", (-8.0, -8.0, 9.0),  (0.2, 1.0, 0.3)),
            ]
            for path, pos, color in specs:
                light = UsdLux.SphereLight.Define(stage, Sdf.Path(path))
                light.CreateRadiusAttr(0.5)
                light.CreateIntensityAttr(50000.0)     # 너무 어두우면 ↑, 너무 밝으면 ↓
                light.CreateColorAttr(Gf.Vec3f(*color))
                UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*pos))
            carb.log_warn("[lights] colored stage lights 추가됨")
        except Exception as e:
            carb.log_error(f"[lights] 실패: {e}")

    def _do_reset(self):
        self.attack_active = False
        self.attack_type = 'none'
        self.attack_intensity = 0.0
        # attack_form / bias_* 는 다음 _cb_attack_config에서 갱신되므로 유지(리셋 불필요)

        self.world.reset()
        self._apply_body_calib('[reset]')   # world.reset() 이 기본값을 되살릴 경우 대비 재적용
        self.needs_reset = False
        carb.log_warn("[SIM] World reset complete")

    def run(self):
        self.timeline.play()

        render_fps = 60
        physics_hz = int(1.0 / self.physics_dt)
        render_interval = max(1, int(physics_hz / render_fps))
        step_counter = 0

        # ── 실시간 페이싱 앵커 (headless엔 렌더 스로틀이 없어 루프가 폭주→PX4 lockstep 붕괴) ──
        #    speed_factor>1이면 sim_time을 그만큼 압축해 더 빨리 진행(lockstep이 따라오는 한).
        self.speed_factor = max(0.1, float(getattr(_pre_args, 'speed', 1.0)))
        wall_start = time.time()

        while simulation_app.is_running() and not self.stop_sim:

            if self.needs_reset:
                self._do_reset()
                wall_start = time.time() - self.sim_time / self.speed_factor   # 재앵커(배율 반영)
                continue

            wf = self.wind.get_force(self.sim_time, self.physics_dt)

            # ── [WINDDBG] 바람 인가 검증용 진단 로그 (env WIND_DBG=1일 때만) ──
            if getattr(self, '_wind_dbg', False) and step_counter % 100 == 0:
                try:
                    st = self.vehicle.state
                    v = np.asarray(st.linear_velocity, dtype=float)
                    vh = float(np.hypot(v[0], v[1]))
                    p = np.asarray(st.position, dtype=float)
                    q = np.asarray(st.attitude, dtype=float)  # [x,y,z,w]
                    # 수직으로부터 틸트각(도): body z축과 world z축 사이 각
                    x_, y_, z_, w_ = q
                    zbz = 1.0 - 2.0*(x_*x_ + y_*y_)          # R[2,2] = body-z의 world-z 성분
                    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, zbz)))))
                    line = (f'[WINDDBG] t={self.sim_time:6.2f} scen={self.wind.scenario} ws={self.wind.ws:.1f} '
                            f'|wf|={np.linalg.norm(wf):.3f}N wfx={wf[0]:+.3f} '
                            f'vhoriz={vh:.3f} vx={v[0]:+.3f} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) '
                            f'tilt={tilt:5.2f}deg atk={self.attack_active} bv={self.body_view is not None}')
                    print(line, flush=True)
                    _dbgpath = os.environ.get('WIND_DBG_FILE', '')
                    if _dbgpath:
                        with open(_dbgpath, 'a') as _f:
                            _f.write(line + '\n')
                except Exception as _e:
                    print(f'[WINDDBG] state read fail: {_e}', flush=True)

            # ── 공격 주입은 더 이상 여기서 하지 않는다 (2026-08-08) ──────────────
            #  ★ 외부 wrench 방식 폐기. 공격은 이제 실기와 **동일 경로**로 주입된다:
            #       companion(online_rl/스윕) → DDS actuator_attack → PX4 SITL ControlAllocator
            #       → c[0] 에 정규화 δ 가산 → actuator_motors → Pegasus 로터모델 → Isaac
            #  이렇게 해야 배분·포화·MOTOR_TAU·추력곡선 결합이 sim·실기에서 같아진다
            #  (외부 wrench 는 그 전부를 우회했다). 근거: ATTACK_INJECTION.md §6-2.
            #  self.attack_active / bias_* 는 여전히 /attack_config 로 받아 **GT 라벨링에만** 쓴다.
            total_force = wf                        # 바람만 외부 힘으로 유지

            if self.body_view and np.any(total_force):
                forces = np.array([total_force], dtype=np.float32)
                tau_w = self.wind.wind_moment(total_force)   # 바람 회전 모멘트 (WIND_MOMENT_ARM>0시 활성)
                if np.any(tau_w):
                    try:
                        self.body_view.apply_forces_and_torques_at_pos(
                            forces=forces, torques=np.array([tau_w], dtype=np.float32), is_global=True)
                    except Exception:
                        self.body_view.apply_forces(forces, is_global=True)  # 폴백(구 API)
                else:
                    self.body_view.apply_forces(forces, is_global=True)

            do_render = (not _pre_args.headless) and (step_counter % render_interval == 0)
            if do_render and self._cam_follow:
                self._update_chase_cam(self.vehicle.state.position)
            self.world.step(render=do_render)

            # ── 달성 배속(RTF) 주기 로그: 요청 speed 대비 실제 도달 배속(=compute 한계 지표) ──
            if step_counter % (physics_hz * 2) == 0:   # 2 sim-초마다
                _now = time.time()
                if not hasattr(self, '_rtf_t0'):
                    self._rtf_t0 = _now; self._rtf_sim0 = self.sim_time
                else:
                    _dw = _now - self._rtf_t0; _ds = self.sim_time - self._rtf_sim0
                    if _dw > 0.5:
                        _rtf = _ds / _dw
                        _flag = '' if _rtf >= 0.95 * self.speed_factor else '  ← compute 한계(요청 미달)'
                        print(f'[RTF] 달성={_rtf:.2f}x / 요청={self.speed_factor:.1f}x '
                              f'(sim={self.sim_time:.0f}s){_flag}', flush=True)
                    self._rtf_t0 = _now; self._rtf_sim0 = self.sim_time

            self.sim_time += self.physics_dt
            step_counter += 1

            if self._rotor_log_path:
                self._log_rotors(step_counter)

            if step_counter % 5 == 0:
                state = self.vehicle.state
                msg = Odometry()
                # ★2026-08-27 sim 시간 스탬프: 제어노드가 궤적 전진을 sim-dt 로 하도록.
                #   (구: wall clock → speed N 에서 궤적이 sim 기준 1/RTF 로 기어가는 버그의 반쪽)
                if self._simclock:          # 정수 스탬프(step_counter 하나로 통일, float sim_time 과 섞지 않음)
                    _us = step_counter * self._step_us
                    msg.header.stamp.sec = _us // 1_000_000
                    msg.header.stamp.nanosec = (_us % 1_000_000) * 1000
                else:
                    msg.header.stamp.sec = int(self.sim_time)
                    msg.header.stamp.nanosec = int((self.sim_time - int(self.sim_time)) * 1e9)
                msg.header.frame_id = "world"
                msg.pose.pose.position.x = float(state.position[0])
                msg.pose.pose.position.y = float(state.position[1])
                msg.pose.pose.position.z = float(state.position[2])
                msg.pose.pose.orientation.x = float(state.attitude[0])
                msg.pose.pose.orientation.y = float(state.attitude[1])
                msg.pose.pose.orientation.z = float(state.attitude[2])
                msg.pose.pose.orientation.w = float(state.attitude[3])
                msg.twist.twist.linear.x = float(state.linear_velocity[0])
                msg.twist.twist.linear.y = float(state.linear_velocity[1])
                msg.twist.twist.linear.z = float(state.linear_velocity[2])
                msg.twist.twist.angular.x = float(state.angular_velocity[0])
                msg.twist.twist.angular.y = float(state.angular_velocity[1])
                msg.twist.twist.angular.z = float(state.angular_velocity[2])
                if self._simclock:          # 발행 벽시계(epoch s) — 노드가 GT 전달 지연·IMU/u 짝 어긋남을 계측(TIMING_LOG gt_rx_lag_ms). 공분산은 쓰는 곳 없음
                    msg.pose.covariance[0] = time.time()
                self.gt_pub.publish(msg)
                if self._simclock and step_counter % 25 == 0:   # GPS 10 Hz = GT 5개마다, 같은 state·같은 스탬프
                    self._publish_gps(state, step_counter * self._step_us)
                    self.last_gps_time = self.sim_time

            if (not self._simclock) and self.sim_time - self.last_gps_time >= 0.1:
                self._publish_gps(state, int(self.sim_time * 1e6))
                self.last_gps_time = self.sim_time

            rclpy.spin_once(self.ros_node, timeout_sec=0)

            # ── 실시간 페이싱: sim_time/speed_factor가 wall-clock을 앞서면 그만큼 sleep ──
            #   (speed_factor>1이면 목표 wall-time을 압축 → 더 빨리 진행. headless 폭주 캡 유지.
            #    루프가 이미 그 속도보다 느리면 sleep=0이라 무해.)
            _ahead = (wall_start + self.sim_time / self.speed_factor) - time.time()
            if _ahead > 0:
                time.sleep(_ahead)

        carb.log_warn("PegasusApp closing.")
        self.timeline.stop()
        simulation_app.close()
        rclpy.shutdown()

    def _publish_gps(self, state, timestamp_us):
        """/sim/sensor_gps 1회 발행 (구 run() 인라인 블록을 그대로 옮김 — 난수 소비 순서 동일)."""
        dp = state.position

        msg_gps = SensorGps()
        msg_gps.timestamp = timestamp_us

        raw_noise = np.random.normal(0, 1.0, 3)
        self.gps_noise_state = 0.9 * self.gps_noise_state + 0.1 * raw_noise

        # ★2026-08-20 센서 σ 정합: 실기 GPS × 0.9(10% 약하게), SENSOR_NOISE_SCALE 로 스윕.
        _sns = float(os.environ.get('SENSOR_NOISE_SCALE', '1.0'))
        gps_noise_n = self.gps_noise_state[0] * 0.27 * _sns   # 0.30→0.27 (×0.9)
        gps_noise_e = self.gps_noise_state[1] * 0.27 * _sns
        gps_noise_alt = self.gps_noise_state[2] * 0.49 * _sns  # raw GPS 수직 ×0.9 (융합 안 씀, raw 센서 일관)

        lat_rad = math.radians(self.home_lat)
        lat_offset = math.degrees(
            (float(dp[1]) + gps_noise_n) / self.earth_radius)
        lon_offset = math.degrees(
            (float(dp[0]) + gps_noise_e) /
            (self.earth_radius * math.cos(lat_rad)))

        msg_gps.latitude_deg = float(self.home_lat + lat_offset)
        msg_gps.longitude_deg = float(self.home_lon + lon_offset)
        msg_gps.altitude_msl_m = float(
            self.home_alt + float(dp[2]) + gps_noise_alt)

        # ★2026-08-20 GPS vel σ: 실측 field [.088,.077,.243]×0.9. 단 수직(D)은 실기가
        #   RNG(EKF2_RNG_CTRL=2)로 고도 잡고 GPS 수직 안 씀 → .219 대신 RNG-clean 0.06.
        msg_gps.vel_n_m_s = float(
            state.linear_velocity[1] + np.random.normal(0, 0.079 * _sns))  # 실기.088×0.9
        msg_gps.vel_e_m_s = float(
            state.linear_velocity[0] + np.random.normal(0, 0.069 * _sns))  # 실기.077×0.9
        msg_gps.vel_d_m_s = float(
            -state.linear_velocity[2] + np.random.normal(0, 0.219 * _sns))  # 실기 raw GPS .243×0.9 (융합X, raw 일관)
        msg_gps.vel_m_s = math.sqrt(
            msg_gps.vel_n_m_s**2 + msg_gps.vel_e_m_s**2 +
            msg_gps.vel_d_m_s**2)

        msg_gps.eph = 0.5
        msg_gps.epv = 0.8
        msg_gps.satellites_used = 12
        msg_gps.fix_type = 3

        self.pub_gps.publish(msg_gps)


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim + PX4 Engine")
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    parser.add_argument('--px4-ns', dest='px4_ns', default='auto')
    parser.add_argument('--speed', dest='speed', type=float, default=1.0)  # ★ pre_parser와 동일하게(없으면 strict parse 에러)
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    PegasusApp(args).run()


if __name__ == "__main__":
    main()
