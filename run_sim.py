"""
run_sim.py — Isaac Sim + PX4 물리 엔진 구동
=============================================
역할:
  1. PhysX 물리 시뮬레이션 (250Hz)
  2. GPS 센서 퍼블리시 (10Hz) — 순수 노이즈만, 센서 공격 없음
  3. Ground Truth Odometry 퍼블리시 (250Hz)
  4. 액추에이터 공격 주입 (곱셈형 LoE: actual=(1-α)·u_ref, -α·u_ref 외력/토크 주입 + Ramp)
  5. 환경 외란 (바람, 충돌)
  6. 에피소드 리셋 제어

Baro/Flow/Distance 센서 퍼블리시 제거 — UKF는 GPS+IMU만 사용.
센서 공격 (GPS FDI) 제거 — Actuator-only 위협 모델.
"""
import carb
import argparse
from isaacsim import SimulationApp

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--headless', dest='headless', action='store_true')
_pre_parser.add_argument('--no-headless', dest='headless', action='store_false')
_pre_parser.add_argument('--px4-ns', dest='px4_ns', default='auto')
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
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
from px4_msgs.msg import SensorGps, VehicleThrustSetpoint, VehicleTorqueSetpoint

from swrl_config import compute_attack_ramp
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

    def _drag(self, w):
        v = np.linalg.norm(w)
        if v < 1e-6:
            return np.zeros(3)
        return 0.5 * self.rho * v**2 * self.Cd * self.A * (w / v)

    def get_force(self, t, dt=0.004):
        d = np.array([np.cos(self.wd), np.sin(self.wd), 0.0])
        if self.scenario == 'wind_constant':
            return self._drag(self.ws * d)
        elif self.scenario == 'wind_gust':
            if t < self.gs or t > self.gs + self.gd:
                return np.zeros(3)
            V = (self.ws / 2) * (1 - np.cos(np.pi * (t - self.gs) / self.gd))
            return self._drag(V * d)
        elif self.scenario == 'wind_turbulence':
            a = np.exp(-self.tb * dt)
            self._ts = a * self._ts + \
                       (1 - a) * self.ti * self.ws * self.rng.standard_normal(3)
            return self._drag(self.ws * d + self._ts)
        return np.zeros(3)

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

        self.attack_active = False
        self.attack_type = 'none'
        self.attack_intensity = 0.0       # 이제 LoE 비율 α (0~1)
        self.attack_start_time = 0.0
        self.attack_ramp_duration = 0.1

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

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        config_multirotor = MultirotorConfig()
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

        self.body_view = None
        self._setup_body_view()

        # ── GUI 보기용: chase-cam + 컬러 조명 (headless엔 무영향, 실패해도 무해) ──
        self._cam_follow = (not args.headless)
        self._cam_offset = np.array([-7.0, -7.0, 4.0])   # 드론 뒤·위 오프셋(ENU)
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
                try:
                    self.body_view = RigidPrimView(
                        prim_paths_expr=path, name="attack_body")
                    self.world.scene.add(self.body_view)
                    self.body_view.initialize()
                except Exception:
                    pass
                break

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

        self.world.reset()
        self.needs_reset = False
        carb.log_warn("[SIM] World reset complete")

    def run(self):
        self.timeline.play()

        render_fps = 60
        physics_hz = int(1.0 / self.physics_dt)
        render_interval = max(1, int(physics_hz / render_fps))
        step_counter = 0

        # ── 실시간 페이싱 앵커 (headless엔 렌더 스로틀이 없어 루프가 폭주→PX4 lockstep 붕괴) ──
        wall_start = time.time()

        while simulation_app.is_running() and not self.stop_sim:

            if self.needs_reset:
                self._do_reset()
                wall_start = time.time() - self.sim_time   # 리셋 동안 흐른 wall-time 보정(재앵커)
                continue

            wf = self.wind.get_force(self.sim_time, self.physics_dt)

            attack_force = np.zeros(3)
            attack_torque = np.zeros(3)

            if self.attack_active and self.attack_type != 'none':
                t_since = self.sim_time - self.attack_start_time
                alpha = compute_attack_ramp(
                    t_since, self.attack_intensity, self.attack_ramp_duration)
                # ── 곱셈형 LoE: 플랜트가 (1-α)·u_ref 에 반응하도록 -α·u_ref 주입 ──
                #   u_ref(물리량) = PX4 명령 setpoint × calib. (덧셈 고정크기 → 명령 비례로 변경)
                #   주입 프레임/적용방식은 기존(월드, is_global=True)과 동일 유지.
                #   ※ 부호 규약: 원본 loe_thrust가 force[2]=-mag(하향=고도손실)였던 것과 동일.
                #     만약 너 환경에서 원본이 상향이었다면 부호만 뒤집으면 됨.
                f_thrust = abs(self.cmd_thrust[2]) * self._C_thrust   # 명령 추력 크기(N)
                attack_force[2] = -alpha * f_thrust                   # 추력 LoE (월드 z 하향)
                if self.attack_type == 'loe_combined':
                    # 토크 채널에도 동일 LoE: -α·(명령 토크). 명령 비례라 크기가 자동 유계.
                    attack_torque[0] = -alpha * (self.cmd_torque[0] * self._C_torque_xy)
                    attack_torque[1] = -alpha * (self.cmd_torque[1] * self._C_torque_xy)
                    attack_torque[2] = -alpha * (self.cmd_torque[2] * self._C_torque_z)

            total_force = wf + attack_force

            if self.body_view:
                has_force = np.any(total_force)
                has_torque = np.any(attack_torque)

                if has_force or has_torque:
                    forces = np.array([total_force], dtype=np.float32) if has_force else np.zeros((1, 3), dtype=np.float32)

                    if has_torque:
                        torques = np.array([attack_torque], dtype=np.float32)
                        self.body_view.apply_forces_and_torques_at_pos(
                            forces=forces, torques=torques, is_global=True)
                    else:
                        self.body_view.apply_forces(forces, is_global=True)

            do_render = (not _pre_args.headless) and (step_counter % render_interval == 0)
            if do_render and self._cam_follow:
                self._update_chase_cam(self.vehicle.state.position)
            self.world.step(render=do_render)

            self.sim_time += self.physics_dt
            step_counter += 1
            if step_counter % 5 == 0:
                state = self.vehicle.state
                msg = Odometry()
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
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
                self.gt_pub.publish(msg)

            if self.sim_time - self.last_gps_time >= 0.1:
                timestamp_us = int(self.sim_time * 1e6)
                dp = state.position

                msg_gps = SensorGps()
                msg_gps.timestamp = timestamp_us

                raw_noise = np.random.normal(0, 1.0, 3)
                self.gps_noise_state = 0.9 * self.gps_noise_state + 0.1 * raw_noise

                gps_noise_n = self.gps_noise_state[0] * 0.3
                gps_noise_e = self.gps_noise_state[1] * 0.3
                gps_noise_alt = self.gps_noise_state[2] * 0.6

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

                msg_gps.vel_n_m_s = float(
                    state.linear_velocity[1] + np.random.normal(0, 0.1))
                msg_gps.vel_e_m_s = float(
                    state.linear_velocity[0] + np.random.normal(0, 0.1))
                msg_gps.vel_d_m_s = float(
                    -state.linear_velocity[2] + np.random.normal(0, 0.1))
                msg_gps.vel_m_s = math.sqrt(
                    msg_gps.vel_n_m_s**2 + msg_gps.vel_e_m_s**2 +
                    msg_gps.vel_d_m_s**2)

                msg_gps.eph = 0.5
                msg_gps.epv = 0.8
                msg_gps.satellites_used = 12
                msg_gps.fix_type = 3

                self.pub_gps.publish(msg_gps)
                self.last_gps_time = self.sim_time

            rclpy.spin_once(self.ros_node, timeout_sec=0)

            # ── 실시간 페이싱: sim_time이 wall-clock을 앞서면 그만큼 sleep ──
            #   (GUI는 렌더가 ~60fps로 묶어주지만 headless는 폭주 → 여기서 캡.
            #    루프가 이미 실시간보다 느리면 sleep=0이라 무해.)
            _ahead = (wall_start + self.sim_time) - time.time()
            if _ahead > 0:
                time.sleep(_ahead)

        carb.log_warn("PegasusApp closing.")
        self.timeline.stop()
        simulation_app.close()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim + PX4 Engine")
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    parser.add_argument('--px4-ns', dest='px4_ns', default='auto')
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    PegasusApp(args).run()


if __name__ == "__main__":
    main()
