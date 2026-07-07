"""
online_rl_main.py — 온라인 RL 제어 루프 + 평가 시스템
======================================================
3단계 리셋: SOFT_RECOVERY / WARM_RESET / HARD_RESET
평가: eval_interval마다 고정 시나리오 5개 순회 (learn OFF, greedy)

★ 제어 루프는 50Hz 타이머 (setpoint 규칙적 발행)
★ GT 콜백은 상태 갱신만 (제어에 영향 없음)
★ learn()은 비동기 스레드 (제어 루프 블로킹 없음)

이번 개정:
  - agent_type 'rhukf' | 'adam'(Adam+Huber baseline) 선택
  - 버스트 LoE 공격 (on-off-on) 스케줄링
  - crash_drift 유예(drift_patience) — transient 스파이크 보호
  - use_logical_done 게이트 (기본 False=물리 crash만 종료) + terminated 부트스트랩 분리
  - eval crash도 reason별 리셋 라우팅 + SOFT_RECOVERY 타임아웃 에스컬레이션
"""
import rclpy
import numpy as np
import math
import json
import collections
import os
import signal
import subprocess
import time as pytime
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    SensorCombined, VehicleOdometry,
    VehicleThrustSetpoint, VehicleTorqueSetpoint, SensorGps
)
from nav_msgs.msg import Odometry as GroundTruthOdometry
from std_msgs.msg import String

import torch
from swrl_config import Config, sample_episode_scenario, sweep_bias_vector
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration, to_physical_u
from env.reward import calculate_reward
from rl.agent import OnlineRHUKFAgent


# 물리적 crash만 부트스트랩 terminal(가치=0)+종단 페널티. drift는 비행 중 이탈→truncation(SOFT 리셋, 페널티 X, 부트스트랩 유지).
PHYSICAL_TERMINALS = ('crash_altitude', 'crash_flip')


# ══════════════════════════════════════════════════════════════
#  Simulator Process Manager
# ══════════════════════════════════════════════════════════════
class SimProcessManager:
    def __init__(self, sim_script='run_sim.py', headless=True,
                 log_dir='./results', sim_launcher='~/isaacsim/python.sh',
                 px4_ns='', kill_stale=True, speed_factor=1.0):
        self.sim_script = sim_script
        self.headless = headless
        self.log_dir = log_dir
        self.sim_launcher = os.path.expanduser(sim_launcher)
        self.px4_ns = px4_ns
        self.kill_stale = kill_stale
        self.speed_factor = float(speed_factor)
        self.process = None
        self._log_file = None
        os.makedirs(log_dir, exist_ok=True)

    def start(self):
        # ── 좀비 PX4 정리 (이전 실행이 남긴 bin/px4가 포트 잡으면 새 PX4가 못 붙음→GT 정지) ──
        if self.kill_stale:
            for pat in ('bin/px4',):
                try:
                    subprocess.run(['pkill', '-9', '-f', pat],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            pytime.sleep(1.0)
        launcher_path = os.path.expanduser('~/isaacsim/python.sh')
        cmd = [launcher_path, self.sim_script]
        if self.headless:
            cmd.append('--headless')
        else:
            cmd.append('--no-headless')
        cmd += ['--px4-ns', self.px4_ns]   # 항상 전달(빈 값이면 bare /fmu)→컨트롤러와 정합 보장
        if self.speed_factor and self.speed_factor != 1.0:
            cmd += ['--speed', str(self.speed_factor)]
        log_path = os.path.join(self.log_dir, 'sim_process.log')
        self._log_file = open(log_path, 'w')
        self.process = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=self._log_file,
            preexec_fn=os.setsid)
        print(f"  [SimManager] Started PID={self.process.pid} "
              f"(cmd={' '.join(cmd[:3])}..., log={log_path})")

    def stop(self):
        if self.process is None:
            return
        pid = self.process.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            self.process.wait(timeout=1)
            print(f"  [SimManager] Instantly killed PID={pid} without waiting.")
        except Exception:
            pass

        self.process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def restart(self):
        self.stop()
        pytime.sleep(5)
        self.start()


# ══════════════════════════════════════════════════════════════
#  Main Node
# ══════════════════════════════════════════════════════════════
class OnlineRLNode(Node):
    def __init__(self, cfg):
        super().__init__('online_rl_controller')
        self.cfg = cfg
        self.step_dt = 0.02  # 50Hz

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # ── Simulator ──
        self.sim_mgr = SimProcessManager(
            'run_sim.py', cfg.headless,
            log_dir=cfg.outdir, sim_launcher=cfg.sim_launcher,
            px4_ns=cfg.px4_namespace,
            kill_stale=getattr(cfg, 'kill_stale_px4_on_start', True),
            speed_factor=getattr(cfg, 'sim_speed_factor', 1.0))
        self.sim_mgr.start()
        self.get_logger().info(
            '  Sim 기동 대기: 첫 GT(/gt/odometry) 수신까지 IDLE 유지 '
            '(헤드리스 콜드 로딩 몇 분 걸려도 죽이지 않음)')
        self._qos = qos

        # ── 첫 GT 전까지 안 죽이기 위한 게이트 ──
        self._first_gt_received = False
        self._fmu_ready = False
        self._startup_t = pytime.time()

        # ── /fmu IO는 첫 GT 이후 _setup_fmu_io()에서 생성 (네임스페이스 정확 감지 위해) ──
        self.pub_offboard = None
        self.pub_traj = None
        self.pub_cmd = None
        self.px4_ns = None

        # ── 비-PX4 토픽 (run_sim 자체 발행/구독; 즉시 생성) ──
        self.pub_attack = self.create_publisher(String, '/attack_config', 10)
        self.pub_scenario = self.create_publisher(String, '/scenario_config', 10)
        self.pub_sim_ctrl = self.create_publisher(String, '/sim_control', 10)
        self.create_subscription(SensorGps, '/sim/sensor_gps', self._cb_gps, qos)
        self.create_subscription(GroundTruthOdometry, '/gt/odometry', self._cb_gt, qos)

        # ── UKF + Agent ──
        self.calib = load_calibration('calibration.json')
        self._ukf_q_gate = getattr(cfg, 'ukf_q_gate_gyro', 0.0)
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
        # ── UKF 오프라인 튜닝용 (z,u) 로깅 (opt-in; sim 1회만 돌려 수집) ──
        self._log_zu = bool(getattr(cfg, 'log_zu', False))
        self._zu_rows = []
        if getattr(cfg, 'agent_type', 'rhukf') == 'adam':
            from rl.agent_adam import OnlineAdamAgent
            self.agent = OnlineAdamAgent(cfg)
        else:
            self.agent = OnlineRHUKFAgent(cfg)
        self.window_buffer = collections.deque(maxlen=cfg.window_size)

        # ── Sensor state ──
        self.cur_accel = np.zeros(3); self.cur_gyro = np.zeros(3)
        self.cur_pos = np.zeros(3); self.cur_vel = np.zeros(3)
        self.cur_euler = np.zeros(3)
        self.cur_thrust = np.zeros(3); self.cur_torque = np.zeros(3)
        self.gt_pos = np.zeros(3); self.gt_vel = np.zeros(3)
        self.obs_gps_pos = np.zeros(3); self.obs_gps_vel = np.zeros(3)
        self.home_lat = None; self.home_lon = None; self.home_alt = None
        self.earth_radius = 6371000.0; self.gps_updated = False
        self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)

        # ── Episode state ──
        self.flight_state = 'IDLE'
        self.episode = 0; self.scenario = None
        self.step_count = 0; self.tick_count = 0
        self.init_counter = 0; self.stable_counter = 0; self.theta = 0.0
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.is_ukf_initialized = False
        self.attack_active_flag = False

        # ── 공격 버스트 상태 ──
        self.attack_bursts = []
        self._cur_burst_start = 0
        self._last_burst_end = None
        self.drift_counter = 0

        # ── HOVER 위치 고정 (★ 떨림 방지) ──
        self._hover_pos = np.zeros(2)  # HOVER 전환 시 위치 저장
        self._hover_alt = 0.0          # ★ HOVER 전환 시 고도 저장 (고도 스냅 과도 제거)
        self._hover_yaw = 0.0  # ★ HOVER 전환 시 Yaw 저장용

        # ── Detection tracking ──
        self.first_hover_step = None
        self.hover_before_attack_count = 0

        # ── Evaluation mode ──
        self.eval_mode = False
        self.eval_scenario_idx = 0
        self.current_eval_results = []
        self.eval_history = []

        # ── Heartbeat ──
        self.last_gt_time = pytime.time()
        self.heartbeat_timeout = 40.0   # 20→40: headless 기동/리셋 여유(GT 첫 수신 지연 흡수)
        self._is_airborne = False

        # ── Async Learning ──
        self._learn_lock = threading.Lock()
        self._is_learning_bg = False

        # ── Logging ──
        self.episode_losses = []
        self.last_learn_dt = 0.0
        self.train_start_time = pytime.time()
        self.hard_reset_count = 0

        # ★ 50Hz 타이머 (GT 콜백이 아닌 규칙적 타이머로 제어)
        self.timer = self.create_timer(self.step_dt, self._tick)

        self.last_z_var = 0.0  # Z 분산 저장용

        # ── α-SWEEP 상태 (sweep_mode일 때만) ──
        self.sweep_mode = getattr(cfg, 'sweep_mode', False)
        if self.sweep_mode:
            self._sweep_setup()

        self.get_logger().info(
            f'[INIT] agent={getattr(cfg, "agent_type", "rhukf")} | dimS={cfg.dimS} | '
            f'eval_interval={cfg.eval_interval} | max_ep={cfg.max_episodes} | '
            f'attack_mode={getattr(cfg, "attack_mode", "single")} | '
            f'use_logical_done={getattr(cfg, "use_logical_done", False)} | '
            f'PER={"ON" if cfg.use_per else "off"}')

    # ══════════════════════════════════════════════════════════
    #  PX4 namespace 자동감지
    # ══════════════════════════════════════════════════════════
    def _resolve_px4_ns(self, configured):
        """configured가 'auto'면 ROS 그래프에서 '*/fmu/out/vehicle_odometry'를 찾아
        '살아있는 publisher가 있는' 네임스페이스를 고른다(죽은 px4_1 ghost 자동 제외).
        못 찾으면 bare ''로 폴백. 'auto'가 아니면 그대로 사용."""
        if configured != 'auto':
            ns = configured.rstrip('/')
            self.get_logger().info(f'  [PX4 ns] 고정 사용: "{ns or "(bare /fmu)"}"')
            return ns
        suffix = '/fmu/out/vehicle_odometry'
        cands = []
        for _ in range(40):   # 최대 ~20s 재시도 (PX4 등록 대기)
            try:
                topics = self.get_topic_names_and_types()
            except Exception:
                topics = []
            cands = [t for t, _ in topics if t.endswith(suffix)]
            live = [t[:-len(suffix)] for t in cands if self.count_publishers(t) > 0]
            if live:
                ns = sorted(live)[0]
                self.get_logger().info(
                    f'  [PX4 ns] 자동감지: "{ns}" (live publisher) | 후보={cands}')
                return ns
            pytime.sleep(0.5)
        if cands:
            ns = sorted(cands)[0][:-len(suffix)]
            self.get_logger().warn(f'  [PX4 ns] live 없음 → 후보 첫번째 "{ns}" 사용 | {cands}')
            return ns
        self.get_logger().error(
            '  [PX4 ns] /fmu/out/vehicle_odometry 토픽을 못 찾음! '
            'MicroXRCEAgent/PX4 미연결 의심. bare "/fmu"로 폴백.')
        return ''

    # ══════════════════════════════════════════════════════════
    #  Sensor Callbacks (★ 상태 갱신만, 제어 로직 없음)
    # ══════════════════════════════════════════════════════════
    def _cb_gps(self, msg):
        if self.home_lat is None:
            self.home_lat = msg.latitude_deg; self.home_lon = msg.longitude_deg; self.home_alt = msg.altitude_msl_m
        lat_rad = math.radians(self.home_lat)
        self.obs_gps_pos[:] = [
            math.radians(msg.longitude_deg - self.home_lon) * self.earth_radius * math.cos(lat_rad),
            math.radians(msg.latitude_deg - self.home_lat) * self.earth_radius,
            msg.altitude_msl_m - self.home_alt]
        self.obs_gps_vel[:] = [msg.vel_e_m_s, msg.vel_n_m_s, -msg.vel_d_m_s]
        self.gps_updated = True

    def _cb_sensor(self, msg):
        self.cur_accel[:] = msg.accelerometer_m_s2[:3]; self.cur_gyro[:] = msg.gyro_rad[:3]

    def _cb_odometry(self, msg):
        self.cur_pos[:] = msg.position[:3]; self.cur_vel[:] = msg.velocity[:3]

    def _cb_thrust(self, msg): self.cur_thrust[:] = msg.xyz[:3]
    def _cb_torque(self, msg): self.cur_torque[:] = msg.xyz[:3]

    def _cb_gt(self, msg):
        """★ 상태 갱신만 — _tick() 호출 안 함"""
        self.gt_pos[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.gt_vel[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        q = msg.pose.pose.orientation
        self.cur_euler[:] = self._quat_to_euler(q.w, q.x, q.y, q.z)
        self.last_gt_time = pytime.time()
        if not self._first_gt_received:
            self._first_gt_received = True
            self.get_logger().info('  ✅ 첫 GT 수신 — sim 기동 완료. /fmu IO 셋업 진행')

    def _setup_fmu_io(self):
        """첫 GT 이후 호출: PX4 네임스페이스 확정(live publisher 기준) + /fmu pub/sub 생성."""
        if self._fmu_ready:
            return
        ns = self._resolve_px4_ns(self.cfg.px4_namespace)
        self.px4_ns = ns
        q = self._qos
        def _fmu(t):
            return f'{ns}{t}'
        self.pub_offboard = self.create_publisher(OffboardControlMode, _fmu('/fmu/in/offboard_control_mode'), q)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, _fmu('/fmu/in/trajectory_setpoint'), q)
        self.pub_cmd = self.create_publisher(VehicleCommand, _fmu('/fmu/in/vehicle_command'), q)
        self.create_subscription(SensorCombined, _fmu('/fmu/out/sensor_combined'), self._cb_sensor, q)
        self.create_subscription(VehicleOdometry, _fmu('/fmu/out/vehicle_odometry'), self._cb_odometry, q)
        self.create_subscription(VehicleThrustSetpoint, _fmu('/fmu/out/vehicle_thrust_setpoint'), self._cb_thrust, q)
        self.create_subscription(VehicleTorqueSetpoint, _fmu('/fmu/out/vehicle_torque_setpoint'), self._cb_torque, q)
        self._fmu_ready = True
        self.get_logger().info(f'  [PX4 ns] /fmu IO 생성 완료 (ns="{ns or "(bare)"}")')

    # ══════════════════════════════════════════════════════════
    #  Utilities
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _quat_to_euler(w, x, y, z):
        return [np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2)),
                np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                np.arctan2(2*(w*z+x*y), 1-2*(y**2+z**2))]

    def _send_setpoint(self, x, y, z, yaw, vx=float('nan'), vy=float('nan'), vz=float('nan')):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.timestamp = 0
        self.pub_traj.publish(msg)

    def _vehicle_cmd(self, command, p1, p2=0.0):
        msg = VehicleCommand(); msg.command = command; msg.param1 = float(p1); msg.param2 = float(p2)
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        msg.timestamp = 0
        self.pub_cmd.publish(msg)

    def _publish_offboard(self):
        off = OffboardControlMode(); off.position = True
        off.timestamp = 0
        self.pub_offboard.publish(off)

    def _send_attack_cmd(self, active, attack_type='none', intensity=0.0,
                         bias_torque_xy=None, bias_torque_z=None, bias_thrust_n=None):
        # bias_* 인자가 주어지면 그 값으로 override(주로 b-sweep용), 아니면 cfg 기본 사용.
        gx = getattr(self.cfg, 'bias_torque_xy', 0.12) if bias_torque_xy is None else bias_torque_xy
        gz = getattr(self.cfg, 'bias_torque_z', 0.0)   if bias_torque_z  is None else bias_torque_z
        gt = getattr(self.cfg, 'bias_thrust_n', 2.0)   if bias_thrust_n  is None else bias_thrust_n
        msg = String(); msg.data = json.dumps({'active': active, 'type': attack_type,
            'intensity': intensity, 'ramp_duration': self.cfg.attack_ramp_duration,
            'form': getattr(self.cfg, 'attack_form', 'additive'),
            'bias_torque_xy': gx, 'bias_torque_z': gz, 'bias_thrust_n': gt})
        self.pub_attack.publish(msg)

    def _send_scenario_cmd(self):
        msg = String(); msg.data = json.dumps({'disturbance_type': self.scenario['disturbance_type'],
            'wind_speed': self.scenario['wind_speed']})
        self.pub_scenario.publish(msg)

    def _send_sim_reset(self):
        msg = String(); msg.data = 'reset'; self.pub_sim_ctrl.publish(msg)

    def _check_heartbeat(self):
        return (pytime.time() - self.last_gt_time) < self.heartbeat_timeout

    def _is_attack_step(self, step):
        """현재 step이 어떤 버스트 ON 구간 [s, e)에 속하는지."""
        for (s, e) in self.attack_bursts:
            if s <= step < e:
                return True
        return False

    # ══════════════════════════════════════════════════════════
    #  Flight Patterns
    # ══════════════════════════════════════════════════════════
    def _compute_setpoint(self):
        """궤도 setpoint 및 속도(Feedforward) 계산 + theta 전진."""
        alt = -abs(self.cfg.flight_altitude); dt = self.step_dt
        t = self.tick_count * dt; R = self.cfg.flight_radius; w = self.cfg.flight_omega
        pattern = self.scenario['pattern'] if self.scenario else 'hover'

        if pattern == 'hover':
            return (0.0, 0.0, alt, 0.0, 0.0, 0.0, 0.0)

        elif pattern == 'circle':
            x = R*np.cos(self.theta) - R          # 중심 (-R,0): theta=0에서 원점 통과 (시작 갭 제거)
            y = R*np.sin(self.theta)
            vx = -R*w*np.sin(self.theta); vy = R*w*np.cos(self.theta)
            yaw = self.theta + np.pi/2; self.theta += w*dt
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'figure8':
            x = R*np.sin(w*t); y = (R/2)*np.sin(2*w*t)
            vx = R*w*np.cos(w*t); vy = R*w*np.cos(2*w*t)
            yaw = np.arctan2(vy, vx) if (vx!=0 or vy!=0) else 0.0
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'waypoint':
            wps = np.array([[0,0],[5,0],[5,5],[-5,5],[-5,0],[0,0]]); seg = 4.0
            t_mod = t%(seg*(len(wps)-1)); idx = int(t_mod/seg); f = (t_mod%seg)/seg
            dx = wps[idx+1][0]-wps[idx][0]; dy = wps[idx+1][1]-wps[idx][1]
            x = wps[idx][0] + dx*f; y = wps[idx][1] + dy*f
            vx = dx / seg; vy = dy / seg
            yaw = np.arctan2(dy, dx)
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'aggressive':
            spp = int(5.0/dt); phase = (self.tick_count//spp)%4; f = (self.tick_count%spp)/spp
            nan = float('nan')
            if phase == 0: return (0.0, 0.0, alt-3.0*np.sin(np.pi*f), 0.0, nan, nan, nan)
            elif phase == 1: a = np.pi*f; return (4*np.cos(a), 4*np.sin(a), alt, a, nan, nan, nan)
            elif phase == 2: return (4.0, 0.0, alt+2.0*np.sin(np.pi*f), np.pi, nan, nan, nan)
            else: a = np.pi*(1-f); return (4*np.cos(a), 4*np.sin(a), alt, a, nan, nan, nan)

        return (0.0, 0.0, alt, 0.0, 0.0, 0.0, 0.0)

    # ══════════════════════════════════════════════════════════
    #  Episode State Reset
    # ══════════════════════════════════════════════════════════
    def _reset_episode_state(self):
        self.step_count = 0; self.tick_count = 0; self.stable_counter = 0; self.theta = 0.0
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.episode_losses = []; self.attack_active_flag = False
        self.window_buffer.clear(); self.gps_updated = False
        self.first_hover_step = None; self.hover_before_attack_count = 0
        self._is_airborne = False; self._hover_pos[:] = 0;  self._hover_yaw = 0.0
        self._hover_alt = -abs(self.cfg.flight_altitude)  # ★ 기본값(미전환 상태용)
        self.cur_pos[:] = 0; self.cur_vel[:] = 0; self.cur_euler[:] = 0
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
        self.is_ukf_initialized = False; self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)
        self.continuous_fp_count = 0
        self.drift_counter = 0
        self.attack_bursts = []
        self._cur_burst_start = 0
        self._last_burst_end = None
        # 에피소드 confusion/지연 메트릭 (TP=공격중hover, FP=평시hover, FN=공격중track, TN=평시track)
        self._ep_tp = self._ep_fp = self._ep_fn = self._ep_tn = 0
        self._ep_det_delay = None
        self._ep_learn_dts = []   # 에피소드 내 learn-step 시간(ms) — speed 한계 판단용

    def _start_new_episode(self):
        if self.sweep_mode:
            self._start_sweep_episode(); return
        if self.eval_mode:
            self.scenario = self.cfg.eval_scenarios[self.eval_scenario_idx]
            label = f'EVAL {self.eval_scenario_idx+1}/{len(self.cfg.eval_scenarios)}'
        else:
            self.episode += 1
            if self.episode > self.cfg.max_episodes:
                self._finish_training(); return
            self.scenario = sample_episode_scenario(self.episode, self.cfg)
            label = f'TRAIN Ep {self.episode}/{self.cfg.max_episodes}'

        atk = self.scenario
        self.get_logger().info(
            f'\n{"="*60}\n  {label}\n'
            f'  Pattern: {atk["pattern"]} | Attack: {atk["attack_type"]} '
            f'(int={atk["attack_intensity"]:.3f}, start={atk["attack_start_step"]}) | '
            f'Wind: {atk.get("disturbance_type","none")} ({atk.get("wind_speed",0):.1f} m/s)\n{"="*60}')
        self._send_scenario_cmd()
        self._reset_episode_state(); self.home_lat = None; self.init_counter = 0

        # ── 공격 버스트 일정 확정 (burst 우선, 없으면 단일 구간) ──
        self.attack_bursts = self.scenario.get('attack_bursts')
        if self.attack_bursts is None:
            if self.scenario.get('attack_type', 'none') != 'none':
                s = self.scenario.get('attack_start_step', 0)
                e = self.scenario.get('attack_end_step', 99999)
                self.attack_bursts = [(s, e)]
            else:
                self.attack_bursts = []
        self._cur_burst_start = 0
        self._last_burst_end = None

    def _check_done(self, trajectory_sp):
        dist = math.hypot(self.cur_pos[0]-trajectory_sp[0], self.cur_pos[1]-trajectory_sp[1])
        # drift는 순간 스파이크가 아니라 지속 이탈일 때만 종료 (transient 보호)
        if dist >= self.cfg.max_error:
            self.drift_counter += 1
        else:
            self.drift_counter = 0
        if self.drift_counter >= self.cfg.drift_patience:
            return True, 'crash_drift'
        if self.cur_pos[2] > self.cfg.min_altitude: return True, 'crash_altitude'
        if abs(self.cur_euler[0]) > 1.05 or abs(self.cur_euler[1]) > 1.05: return True, 'crash_flip'
        if self.step_count >= self.cfg.episode_max_steps: return True, 'timeout'
        return False, None

    def _finish_training(self):
        total = pytime.time() - self.train_start_time
        self.get_logger().info(f'\n{"#"*60}\n  Training Complete | {total:.0f}s ({total/60:.1f}min)\n'
            f'  Episodes: {self.episode-1} | Hard Resets: {self.hard_reset_count}\n{"#"*60}')
        self.agent.save(os.path.join(self.cfg.outdir, 'final_model.pt'))
        if self.eval_history:
            np.savez(os.path.join(self.cfg.outdir, 'eval_history.npz'),
                     eval_history=self.eval_history)
        self._autoplot()
        self.sim_mgr.stop(); raise SystemExit("Training complete")

    def _autoplot(self):
        """학습 종료 시 단일-에이전트 plot 자동 생성 (plot_results.py 서브프로세스)."""
        try:
            import subprocess
            agent = getattr(self.cfg, 'agent_type', 'rhukf')
            mpath = os.path.join(self.cfg.outdir, f'metrics_{agent}.csv')
            if not os.path.exists(mpath):
                return
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_results.py')
            subprocess.run(['python3', script, mpath, '--outdir', self.cfg.outdir], timeout=120)
            self.get_logger().info(f'[PLOT] 자동 생성 → {self.cfg.outdir}/metrics_{agent}.png')
        except Exception as e:
            self.get_logger().warn(f'[PLOT] 자동 plot 실패(무시): {e}')

    def _trigger_hard_reset(self):
        self._send_attack_cmd(False)
        self._reset_episode_state()
        self.cur_pos[:] = 0; self.cur_vel[:] = 0; self.cur_euler[:] = 0
        self.home_lat = None; self.init_counter = 0; self.flight_state = 'HARD_RESET'

    def _apply_reset(self, reason):
        """crash 종류에 따라 SOFT / WARM / HARD 리셋 선택 (train·eval 공통)."""
        if reason == 'crash_flip':
            self._trigger_hard_reset()
        elif reason == 'crash_altitude':
            self._reset_episode_state(); self.home_lat = None
            self.init_counter = 0; self.flight_state = 'WARM_RESET'
        else:  # crash_drift, timeout, (use_logical_done 시 논리종료)
            self._reset_episode_state(); self.stable_counter = 0
            self.init_counter = 0; self.flight_state = 'SOFT_RECOVERY'

    # ══════════════════════════════════════════════════════════
    #  Evaluation System
    # ══════════════════════════════════════════════════════════
    def _start_eval_round(self):
        self.eval_mode = True; self.eval_scenario_idx = 0; self.current_eval_results = []
        self.get_logger().info(
            f'\n  ╔═══ EVAL Round @ Ep {self.episode} ({len(self.cfg.eval_scenarios)} scenarios) ═══╗')

    def _record_eval_result(self, reason):
        attack_start = self.scenario.get('attack_start_step', 0)
        det_delay = -1
        if self.first_hover_step is not None and attack_start > 0:
            det_delay = max(0, self.first_hover_step - attack_start)
        fa_rate = 0.0
        if attack_start > 0:
            pre = min(self.step_count, attack_start)
            fa_rate = self.hover_before_attack_count / max(pre, 1)
        result = {
            'scenario_idx': self.eval_scenario_idx,
            'attack_type': self.scenario['attack_type'],
            'intensity': self.scenario['attack_intensity'],
            'pattern': self.scenario['pattern'],
            'survived': reason == 'timeout',
            'reward': self.episode_reward,
            'steps': self.step_count,
            'reward_rate': self.episode_reward / max(self.step_count, 1),
            'det_delay': det_delay, 'false_alarm_rate': fa_rate, 'reason': reason,
        }
        self.current_eval_results.append(result)
        surv = '✅' if result['survived'] else '❌'
        dd = f"{det_delay}" if det_delay >= 0 else 'N/A'
        self.get_logger().info(
            f'  ║ Eval {self.eval_scenario_idx+1}: {surv} {reason} | '
            f'R={self.episode_reward:.1f} | Steps={self.step_count} | DetDelay={dd} | FA={fa_rate:.2f}')

    def _finish_eval_round(self):
        self.eval_mode = False
        results = self.current_eval_results
        survival_rate = np.mean([r['survived'] for r in results])
        mean_rr = np.mean([r['reward_rate'] for r in results])
        det_delays = [r['det_delay'] for r in results if r['det_delay'] >= 0]
        mean_dd = np.mean(det_delays) if det_delays else -1
        mean_fa = np.mean([r['false_alarm_rate'] for r in results])
        eval_summary = {
            'train_episode': self.episode, 'survival_rate': float(survival_rate),
            'mean_reward_rate': float(mean_rr), 'mean_det_delay': float(mean_dd),
            'mean_false_alarm_rate': float(mean_fa), 'per_scenario': results,
        }
        self.eval_history.append(eval_summary)
        dd_str = f"{mean_dd:.1f}" if mean_dd >= 0 else "N/A"
        self.get_logger().info(
            f'  ╠═ Survival: {survival_rate:.0%} | RewardRate: {mean_rr:.3f} | '
            f'DetDelay: {dd_str} | FA: {mean_fa:.3f}\n'
            f'  ╚═══════════════════════════════════════════╝')
        np.savez(os.path.join(self.cfg.outdir, 'eval_history.npz'), eval_history=self.eval_history)

    # ══════════════════════════════════════════════════════════
    #  Main Tick (★ 50Hz 타이머 — 규칙적 제어)
    # ══════════════════════════════════════════════════════════
    def _tick(self):
        # ── Heartbeat (첫 GT 수신 이후에만 작동 = 콜드 로딩 중엔 절대 안 죽임) ──
        if self._first_gt_received and self.flight_state not in ('IDLE', 'HARD_RESET'):
            if not self._check_heartbeat():
                self.get_logger().error('  💀 Heartbeat lost → HARD_RESET')
                self._trigger_hard_reset(); return

        # ── Offboard 유지 (★ 50Hz 규칙 발행 = PX4 안정) ──
        if self.flight_state in ('SOFT_RECOVERY', 'TAKEOFF', 'STABILIZE', 'LEARNING'):
            self._publish_offboard()

        # ── IDLE: sim이 GT를 흘릴 때까지 대기(헤드리스 콜드 로딩 ~수분) → 준비되면 시작 ──
        if self.flight_state == 'IDLE':
            if not self._first_gt_received:
                waited = pytime.time() - self._startup_t
                if int(waited) % 10 == 0 and waited >= 10:
                    self.get_logger().info(f'  … sim 로딩 대기 {int(waited)}s (첫 GT 대기 중)')
                if waited > self.cfg.sim_startup_timeout:
                    self.get_logger().error(
                        f'  sim 기동 {self.cfg.sim_startup_timeout:.0f}s 초과 → HARD_RESET')
                    self._trigger_hard_reset()
                return
            if not self._fmu_ready:
                self._setup_fmu_io()
                self.last_gt_time = pytime.time()
            self._start_new_episode()
            self.flight_state = 'TAKEOFF'
            self.get_logger().info('  → TAKEOFF')

        # ── SOFT_RECOVERY ──
        elif self.flight_state == 'SOFT_RECOVERY':
            self._send_setpoint(0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self.init_counter += 1
            dist = np.linalg.norm(self.cur_pos[:2])
            alt_err = abs(self.cur_pos[2] + self.cfg.flight_altitude)
            if dist < 1.0 and alt_err < 0.5: self.stable_counter += 1
            else: self.stable_counter = 0
            if self.stable_counter >= int(self.cfg.warmup_seconds / self.step_dt):
                self._start_new_episode()
                self.flight_state = 'STABILIZE'; self.stable_counter = 0; self.init_counter = 0
                self.get_logger().info('  → STABILIZE (soft)')
            elif self.init_counter >= int(self.cfg.soft_recovery_timeout / self.step_dt):
                self.get_logger().warn('  [SOFT] 복구 시간 초과 → WARM_RESET 에스컬레이션')
                self._reset_episode_state(); self.home_lat = None
                self.init_counter = 0; self.flight_state = 'WARM_RESET'

        # ── WARM_RESET ──
        elif self.flight_state == 'WARM_RESET':
            self._send_setpoint(0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self.init_counter += 1
            if self.init_counter == 1:
                self._send_sim_reset()
            if self.init_counter >= int(3.0 / self.step_dt):
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._start_new_episode()
                self.flight_state = 'TAKEOFF'

        # ── HARD_RESET ──
        elif self.flight_state == 'HARD_RESET':
            self.init_counter += 1
            if self.init_counter == 1:
                self.get_logger().warn('  [HARD] Restarting simulator... (첫 GT까지 대기)')
                self.sim_mgr.restart(); self.hard_reset_count += 1
                self._first_gt_received = False    # 재로딩 동안 안 죽이게 게이트 리셋
                self._startup_t = pytime.time()
                self.last_gt_time = pytime.time()
            if self._first_gt_received:
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._start_new_episode()
                self.flight_state = 'TAKEOFF'
            elif (pytime.time() - self._startup_t) > self.cfg.sim_startup_timeout:
                self.get_logger().error('  [HARD] 재기동 타임아웃 → 재시도'); self.init_counter = 0

        # ── TAKEOFF ──
        elif self.flight_state == 'TAKEOFF':
            sp = (0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self._send_setpoint(*sp)
            self.init_counter += 1
            ticks_per_sec = int(1.0 / self.step_dt)

            if not self._is_airborne and self.cur_pos[2] < -0.5:
                self._is_airborne = True
                self.get_logger().info(
                    f'  [TAKEOFF] 이륙 감지! alt={-self.cur_pos[2]:.1f}m → Arm 명령 중단')

            if not self._is_airborne:
                if self.init_counter >= ticks_per_sec and self.init_counter % ticks_per_sec == 0:
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    if self.init_counter == ticks_per_sec:
                        self.get_logger().info(
                            f'  [TAKEOFF] Arm 시도 중')

            dist = np.linalg.norm(self.cur_pos - np.array(sp[:3]))
            if dist < 1.0: self.stable_counter += 1
            else: self.stable_counter = 0

            if self.stable_counter >= int(self.cfg.warmup_seconds / self.step_dt):
                self.flight_state = 'STABILIZE'; self.stable_counter = 0
                self.get_logger().info('  → STABILIZE')

            if self.init_counter >= int(60.0 / self.step_dt):
                self.get_logger().warn('  [TAKEOFF] Timeout 60s → WARM_RESET')
                self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
                self.flight_state = 'WARM_RESET'

        # ── STABILIZE ──
        elif self.flight_state == 'STABILIZE':
            sp = (0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self._send_setpoint(*sp)
            self._run_ukf_step()
            self.stable_counter += 1

            warmup_ticks = int(self.cfg.warmup_seconds / self.step_dt)
            attitude_stable = abs(self.cur_euler[0]) < 0.1 and abs(self.cur_euler[1]) < 0.1
            if self.stable_counter >= warmup_ticks and attitude_stable:
                self.window_buffer.clear()
                self.step_count = 0; self.tick_count = 0

                pattern = self.scenario['pattern'] if self.scenario else 'hover'
                if pattern == 'circle':
                    self.theta = 0.0   # 오프셋 원이 원점을 지나므로 0에서 시작 → 드론 위치와 일치
                elif pattern in ('figure8', 'waypoint', 'aggressive'):
                    self.tick_count = 0

                self.flight_state = 'LEARNING'
                self.get_logger().info(
                    f'  → LEARNING (pattern={pattern}, theta={self.theta:.2f})')

        # ── LEARNING (★ 핵심 수정) ──
        elif self.flight_state == 'LEARNING':
            if self.prev_action == 1:
                control_sp = (float(self._hover_pos[0]), float(self._hover_pos[1]),
                              float(self._hover_alt), self._hover_yaw, 0.0, 0.0, 0.0)
                trajectory_sp = control_sp
            else:
                trajectory_sp = self._compute_setpoint()
                control_sp = trajectory_sp
                self.tick_count += 1

            self._send_setpoint(*control_sp)
            self._run_ukf_step()

            if self.gps_updated:
                self.gps_updated = False
                if self.sweep_mode:
                    self._sweep_step_10hz(trajectory_sp)
                else:
                    self._rl_step_10hz(trajectory_sp)

    # ══════════════════════════════════════════════════════════
    #  UKF Step (50Hz)
    # ══════════════════════════════════════════════════════════
    def _run_ukf_step(self):
        gps_ned = [self.obs_gps_pos[1], self.obs_gps_pos[0], -self.obs_gps_pos[2]]
        vel_ned = [self.obs_gps_vel[1], self.obs_gps_vel[0], -self.obs_gps_vel[2]]
        z_9d = np.concatenate([gps_ned, vel_ned, self.cur_gyro])
        u_phys = to_physical_u(np.array([self.cur_thrust]), np.array([self.cur_torque]), self.calib)[0]
        _was_init = not self.is_ukf_initialized
        if not self.is_ukf_initialized:
            self.ukf.x[0:3] = gps_ned; self.ukf.x[3:6] = self.cur_euler
            self.ukf.x[6:9] = vel_ned; self.ukf.x[9:12] = self.cur_gyro
            self.is_ukf_initialized = True
        self.last_res, self.last_Pzz = self.ukf.step(z_9d, u_phys)
        # ── 오프라인 UKF 튜닝용 로깅: [ep, reset, attack, action, z(9), u(4), euler(3), atk_scale, atk_delay] ──
        if self._log_zu:
            try:
                _atk_on = bool(getattr(self, 'attack_active_flag', False))
                _scale = float(self.scenario.get('bias_scale', 0.0)) if (self.scenario and _atk_on) else 0.0
                _delay = float(self.step_count - self._cur_burst_start) if _atk_on else -1.0
                self._zu_rows.append(np.concatenate([
                    [float(self.episode),
                     1.0 if _was_init else 0.0,
                     1.0 if _atk_on else 0.0,
                     float(self.prev_action if self.prev_action is not None else 0.0)],
                    np.asarray(z_9d, dtype=float),
                    np.asarray(u_phys, dtype=float),
                    np.asarray(self.cur_euler, dtype=float),
                    [_scale, _delay]]))    # ★ 공격강도 s(밴드대비) + 공격경과스텝
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════
    #  Async Learning (비동기 — 제어 블로킹 없음)
    # ══════════════════════════════════════════════════════════
    def _async_learn_task(self):
        """백그라운드 스레드에서 실행. theta 접근 시 Lock 사용."""
        try:
            with self._learn_lock:
               loss, dt_ms, z_var = self.agent.learn()
            if loss > 0:
                self.episode_losses.append(loss)
                self.last_learn_dt = dt_ms
                self.last_z_var = z_var  # ★ 추가
                if not hasattr(self, '_ep_learn_dts'):
                    self._ep_learn_dts = []
                self._ep_learn_dts.append(dt_ms)
        except Exception as e:
            self.get_logger().error(f"  [LEARN ERROR] {e}")
        finally:
            self._is_learning_bg = False

    # ══════════════════════════════════════════════════════════
    #  10Hz RL Step
    # ══════════════════════════════════════════════════════════
    def _rl_step_10hz(self, trajectory_sp):
        cfg = self.cfg

        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0, offset=0.5)  # vel 저압축(log0.5)
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)              # gyro log1p 유지
        self._last_nis_raw = (nis_v_raw, nis_g_raw)   # 디버그 로깅용

        if self.step_count < cfg.learning_warmup_steps:
            self.step_count += 1
            return

        act_val = float(self.prev_action if self.prev_action is not None else 0.0)
        self.window_buffer.append([nis_vel, nis_gyr, act_val])

        if len(self.window_buffer) < cfg.window_size:
            self.step_count += 1; return

        state = np.array(self.window_buffer).flatten()
        done, term_reason = self._check_done(trajectory_sp)

        attack_delay = 0
        recovery_delay = 0

        # ── 1. 딜레이 및 FP 카운터 계산 (버스트 기준) ──
        if self.attack_active_flag:
            attack_delay = max(0, self.step_count - self._cur_burst_start)
            self.continuous_fp_count = 0
        else:
            if self._last_burst_end is not None and self.step_count >= self._last_burst_end:
                recovery_delay = max(0, self.step_count - self._last_burst_end)
                self.continuous_fp_count = 0
            else:
                if self.prev_action == 1:
                    self.continuous_fp_count += 1
                else:
                    self.continuous_fp_count = 0

        # ── 2. 퓨어한 보상 계산 ──
        # FP 인자: 공격직후 recovery는 recovery_delay(offset grace); 순수오탐은 연속 hover 카운트(첫스텝 -1 점증).
        fp_rec_arg = (min(recovery_delay, 5) if self._last_burst_end is not None
                      else min(self.continuous_fp_count, 5))
        reward = calculate_reward(
            self.prev_action if self.prev_action is not None else 0,
            self.attack_active_flag,
            min(attack_delay, 5),      # FN: attack_delay (onset grace + 에스컬레이션)
            fp_rec_arg,                # FP: recovery_delay (offset grace) or 큰값(순수오탐)
            rc=self.cfg.reward,
        )
        # 물리적 crash = 결과 기반 종단 페널티 (추종오차 대신 '추락=나쁨' outcome anchor).
        #   강도가 '생존 가능 띠'에 들어와 있을 때만 깨끗한 신호가 됨(즉사 구간이면 노이즈).
        if done and term_reason in PHYSICAL_TERMINALS:
            reward += self.cfg.reward.terminal_penalty
        self.episode_reward += reward

        # ── confusion/지연 메트릭 누적 (prev_action vs 공격상태) ──
        _a = self.prev_action if self.prev_action is not None else 0
        if self.attack_active_flag:
            if _a == 1:
                self._ep_tp += 1
                if self._ep_det_delay is None:
                    self._ep_det_delay = max(0, self.step_count - self._cur_burst_start)
            else:
                self._ep_fn += 1
        else:
            if _a == 1:
                self._ep_fp += 1
            else:
                self._ep_tn += 1

        # ── 3. 논리적 종료 (use_logical_done=True일 때만; 기본 False=물리 crash만) ──
        if cfg.use_logical_done and not done:
            done_steps = cfg.done_steps
            if attack_delay >= done_steps:
                done = True; term_reason = 'detection_failed'
            elif recovery_delay >= done_steps:
                done = True; term_reason = 'recovery_failed'
            elif self.continuous_fp_count >= done_steps:
                done = True; term_reason = 'excessive_fp'

        # ── 부트스트랩용 terminal: 물리적 crash만 True (timeout·논리종료는 truncation) ──
        terminated = term_reason in PHYSICAL_TERMINALS

        # ── Transition 저장 + 비동기 학습 ──
        if self.prev_state is not None and self.prev_action is not None:
            if not self.eval_mode:
                self.agent.push(self.prev_state, self.prev_action, reward, state, terminated)
                if not self._is_learning_bg:
                    self._is_learning_bg = True
                    threading.Thread(target=self._async_learn_task, daemon=True).start()

        if done: self._end_episode(term_reason); return

        # ── Action 선택 (★ Lock 보호) ──
        if self.eval_mode:
            with self._learn_lock:
                action = self.agent.act(state, eps=0.0)
            eps = 0.0
        else:
            eps = self.agent.get_epsilon()
            with self._learn_lock:
                action = self.agent.act(state, eps)

        # ── HOVER 전환 시 위치 고정 (★ 떨림 방지) ──
        if action == 1 and (self.prev_action != 1 or self.prev_action is None):
            self._hover_pos[:] = self.cur_pos[:2]
            self._hover_alt = float(self.cur_pos[2])   # ★ 현재 고도에서 호버 (스냅 과도 제거)
            self._hover_yaw = float(self.cur_euler[2])

        # ── Detection tracking ──
        if action == 1:
            if self.first_hover_step is None:
                self.first_hover_step = self.step_count
            if not self.attack_active_flag:
                self.hover_before_attack_count += 1

        # ── Attack burst on/off (버스트 경계에서 토글) ──
        want_attack = (self.scenario['attack_type'] != 'none') and self._is_attack_step(self.step_count)
        if want_attack and not self.attack_active_flag:
            sc = self.scenario
            self._send_attack_cmd(True, sc['attack_type'], sc.get('attack_intensity', 1.0),
                bias_torque_xy=sc.get('bias_torque_xy', None),
                bias_torque_z=sc.get('bias_torque_z', None),
                bias_thrust_n=sc.get('bias_thrust_n', None))
            self.attack_active_flag = True
            self._cur_burst_start = self.step_count
            # 실제 물리 강도 = intensity × bias (loe_combined: τ=int·bias_torque, thrust=int·bias_thrust)
            _int = sc.get('attack_intensity', 1.0)
            _gx = sc.get('bias_torque_xy', getattr(self.cfg, 'bias_torque_xy', 0.12))
            _gz = sc.get('bias_torque_z',  getattr(self.cfg, 'bias_torque_z', 0.0))
            _gt = sc.get('bias_thrust_n',  getattr(self.cfg, 'bias_thrust_n', 2.0))
            self.get_logger().warn(
                f'  🚨 Attack ON (burst) @ step {self.step_count}: {self.scenario["attack_type"]} '
                f'| int={_int:.2f} → τxy={_int*_gx:+.3f} τz={_int*_gz:+.3f} N·m, '
                f'thrust={_int*_gt:+.2f} N')
        elif (not want_attack) and self.attack_active_flag:
            self._send_attack_cmd(False)
            self.attack_active_flag = False
            self._last_burst_end = self.step_count
            self.get_logger().warn(f'  🟢 Attack OFF (burst) @ step {self.step_count}')

        # ── Debug log ──
        if self.step_count % cfg.log_interval == 0:
            sp = np.array(trajectory_sp[:3])
            gt_ned = np.array([self.gt_pos[1], self.gt_pos[0], -self.gt_pos[2]])
            gt_err = np.linalg.norm(gt_ned[:2] - sp[:2])
            alt = -self.cur_pos[2] if self.cur_pos[2] < 0 else 0.0
            atk = '🔴ATK' if self.attack_active_flag else '⚪NRM'
            if self.attack_active_flag:
                _sc = self.scenario
                _i = _sc.get('attack_intensity', 1.0)
                _txy = _i * _sc.get('bias_torque_xy', getattr(self.cfg, 'bias_torque_xy', 0.12))
                _th  = _i * _sc.get('bias_thrust_n',  getattr(self.cfg, 'bias_thrust_n', 2.0))
                atk = f'🔴ATK(τ{_txy:.2f} T{_th:.1f})'   # 실제 강도: 토크xy[N·m] 추력[N]
            act = 'HOVER' if action == 1 else 'TRACK'
            mode = 'EVAL' if self.eval_mode else 'TRAIN'
            buf = self.agent.buffer.current_size
            cur_loss = self.episode_losses[-1] if self.episode_losses else 0

            self.get_logger().info(
                f'  [{self.step_count:3d}] {mode} {atk} {act} | ε={eps:.3f} | '
                f'NIS v={nis_vel:.3f} g={nis_gyr:.3f} (raw v={nis_v_raw:.2f} g={nis_g_raw:.2f}) | '
                f'R={reward:+.1f} (Σ={self.episode_reward:.1f}) | '
                f'GT={gt_err:.2f}m alt={alt:.1f}m | '
                f'buf={buf} loss={cur_loss:.4f} Zvar={self.last_z_var:.3f} dt={self.last_learn_dt:.0f}ms')

        self.prev_state = state; self.prev_action = action; self.step_count += 1

    # ══════════════════════════════════════════════════════════
    #  학습 에피소드 메트릭 CSV (reward/loss/F1/delay 등) → plot_results.py 용
    # ══════════════════════════════════════════════════════════
    def _write_train_metrics(self, reason, avg_loss, eps):
        import csv, os
        tp, fp, fn, tn = self._ep_tp, self._ep_fp, self._ep_fn, self._ep_tn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        det_delay = self._ep_det_delay if self._ep_det_delay is not None else -1
        crashed = 1 if reason in ('crash_altitude', 'crash_flip', 'crash_drift') else 0
        try:
            td = self.agent.td_kurtosis()
            td_exkurt = float(td[2])
        except Exception:
            td_exkurt = 0.0
        row = {
            'episode': self.episode, 'agent': getattr(self.cfg, 'agent_type', 'rhukf'),
            'reward': round(self.episode_reward, 3), 'loss': round(float(avg_loss), 5),
            'steps': self.step_count, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
            'fp_rate': round(fp_rate, 4), 'det_delay': det_delay, 'crashed': crashed,
            'td_exkurt': round(td_exkurt, 4), 'epsilon': round(float(eps), 4),
        }
        if getattr(self, '_metrics_w', None) is None:
            os.makedirs(self.cfg.outdir, exist_ok=True)
            path = os.path.join(self.cfg.outdir,
                                f'metrics_{getattr(self.cfg, "agent_type", "rhukf")}.csv')
            self._metrics_f = open(path, 'w', newline='')
            self._metrics_w = csv.DictWriter(self._metrics_f, fieldnames=list(row.keys()))
            self._metrics_w.writeheader()
            self.get_logger().info(f'[METRICS] 학습 메트릭 기록 → {path}')
        self._metrics_w.writerow(row)
        self._metrics_f.flush()

    def _flush_zu(self):
        """(z,u) 로그를 zu_log.npz로 누적 저장 (UKF 오프라인 튜닝용). 매 에피소드 덮어씀."""
        if not self._log_zu or not self._zu_rows:
            return
        try:
            import os
            arr = np.asarray(self._zu_rows, dtype=np.float64)
            path = os.path.join(self.cfg.outdir, 'zu_log.npz')
            np.savez(path, data=arr, dt=float(self.step_dt),
                     q_gate=float(self._ukf_q_gate),
                     cols='episode,reset,attack,action,z0_gpsN,z1_gpsE,z2_gpsD,'
                          'z3_velN,z4_velE,z5_velD,z6_gyrx,z7_gyry,z8_gyrz,'
                          'u0_thrust,u1_tx,u2_ty,u3_tz,euler_phi,euler_th,euler_psi,'
                          'atk_scale,atk_delay')
            self.get_logger().info(f'[ZU] (z,u) 로그 저장 → {path}  (rows={len(arr)})')
        except Exception as e:
            self.get_logger().warn(f'[ZU] 저장 실패(무시): {e}')

    # ══════════════════════════════════════════════════════════
    #  Episode End
    # ══════════════════════════════════════════════════════════
    def _end_episode(self, reason):
        self._send_attack_cmd(False); self.attack_active_flag = False
        self._flush_zu()

        if self.sweep_mode:
            self._end_sweep_episode(reason); return

        # ── EVAL: 결과 기록 후, crash 종류에 맞는 리셋으로 라우팅 ──
        #   (이전엔 무조건 SOFT → 추락/뒤집힘 시 복구 불가 → 드론 사라진 채 무한 대기 버그)
        if self.eval_mode:
            self._record_eval_result(reason)
            self.eval_scenario_idx += 1
            if self.eval_scenario_idx >= len(self.cfg.eval_scenarios):
                self._finish_eval_round()
            self._apply_reset(reason)
            return

        self.agent.end_episode(self.episode_reward, self.step_count)
        avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0
        eps = self.agent.get_epsilon(); p_init = self.agent._compute_adaptive_p()
        self._write_train_metrics(reason, avg_loss, eps)
        emojis = {'crash_drift': '⚠️ DRIFT', 'crash_altitude': '💀 CRASH',
                  'crash_flip': '🔥 FLIP', 'timeout': '⏱️ TIMEOUT',
                  'detection_failed': '🙈 MISS', 'recovery_failed': '🔒 STUCK',
                  'excessive_fp': '🤡 PANIC'}
        reset_label = {'crash_flip': 'HARD', 'crash_altitude': 'WARM'}.get(reason, 'SOFT')

        _ld = self._ep_learn_dts
        _sf = getattr(self.cfg, 'sim_speed_factor', 1.0)
        _tdk = getattr(self.agent, 'td_kurtosis', lambda: (0, 0.0, 0.0))()
        _learn_str = (
            f'  │ learn-step: mean={np.mean(_ld):.1f}ms max={np.max(_ld):.1f}ms (n={len(_ld)}) '
            f'| step예산={100.0/max(_sf,0.01):.0f}ms@speed{_sf:.1f}\n'
            if _ld else '')

        self.get_logger().info(
            f'\n  ┌─ Ep {self.episode}: {emojis.get(reason, reason)} → {reset_label} reset\n'
            f'  │ R={self.episode_reward:.1f} Steps={self.step_count} Loss={avg_loss:.4f}\n'
            f'  │ ε={eps:.3f} P={p_init:.5f} | TD|n,μ,exkurt|={_tdk}\n'
            f'{_learn_str}'
            f'  │ Atk: {self.scenario["attack_type"]}(int={self.scenario["attack_intensity"]:.3f}, '
            f'start={self.scenario["attack_start_step"]}) | {self.scenario["pattern"]} | '
            f'{self.scenario.get("disturbance_type","none")}\n  └─{"─"*50}')

        if self.episode % 50 == 0:
            self.agent.save(os.path.join(self.cfg.outdir, f'model_ep{self.episode}.pt'))

        if self.episode % self.cfg.eval_interval == 0:
            self._start_eval_round()

        self._apply_reset(reason)

    # ══════════════════════════════════════════════════════════
    #  α-SWEEP MODE (학습 OFF; 고정정책으로 결과성/탐지가능성 특성화)
    # ══════════════════════════════════════════════════════════
    def _sweep_setup(self):
        """셀 리스트 구성 + CSV 오픈. 셀 = (bias값, policy, pattern).
        baseline 2개(무공격 track/hover) + bias값마다 (track, hover)."""
        import csv
        cfg = self.cfg
        cells = [(0.0, 'track', cfg.sweep_pattern), (0.0, 'hover', 'hover')]
        for a in cfg.sweep_values:
            cells.append((float(a), 'track', cfg.sweep_pattern))
            cells.append((float(a), 'hover', 'hover'))
            # 조건 C: 추적 패턴으로 비행하다 공격 시작 +d 스텝에 호버 전환 (전이 케이스)
            for d in getattr(cfg, 'sweep_hover_delays', ()):
                cells.append((float(a), f'dhover{int(d)}', cfg.sweep_pattern))
        self.sweep_cells = cells
        self.sweep_cell_idx = 0
        self.sweep_ep_in_cell = 0
        self.sweep_value, self.sweep_policy, self.sweep_pattern_cur = cells[0]
        self._sweep_bias = (0.0, 0.0, 0.0)

        os.makedirs(cfg.outdir, exist_ok=True)
        self._sweep_detail_f = open(os.path.join(cfg.outdir, 'sweep_detail.csv'), 'w', newline='')
        self._sweep_detail_w = csv.writer(self._sweep_detail_f)
        self._sweep_detail_w.writerow([
            'cell_idx', 'mode', 'bias', 'tq_xy', 'th_n', 'policy', 'pattern', 'episode', 'step', 'attack_active',
            'nis_v_raw', 'nis_g_raw', 'nis_v_scaled', 'nis_g_scaled',
            'gt_err', 'alt', 'action', 'crash_reason'])
        self._sweep_summary_f = open(os.path.join(cfg.outdir, 'sweep_summary.csv'), 'w', newline='')
        self._sweep_summary_w = csv.writer(self._sweep_summary_f)
        self._sweep_summary_w.writerow([
            'cell_idx', 'mode', 'bias', 'tq_xy', 'th_n', 'policy', 'pattern', 'episode',
            'survived', 'crash_step', 'crash_reason', 'steps'])
        _unit = 'N' if cfg.sweep_attack_mode == 'thrust' else 'Nm'
        _ftr = (f' ft_ratio={cfg.sweep_combined_ft_ratio}'
                if cfg.sweep_attack_mode == 'combined' else '')
        self.get_logger().info(
            f'\n{"#"*60}\n  [SWEEP] {len(cells)} cells × {cfg.sweep_episodes} ep '
            f'| mode={cfg.sweep_attack_mode}{_ftr} bias({_unit})={cfg.sweep_values}'
            f' delays={getattr(cfg, "sweep_hover_delays", ())}\n'
            f'  attack: additive @step{cfg.sweep_attack_start}, '
            f'ramp={cfg.attack_ramp_duration}s | q_gate={self._ukf_q_gate}\n{"#"*60}')

    def _start_sweep_episode(self):
        cell = self.sweep_cells[self.sweep_cell_idx]
        self.sweep_value, self.sweep_policy, self.sweep_pattern_cur = cell
        # 모드+값 → 실제 물리 바이어스 벡터 (이번 셀에서 주입할 값)
        self._sweep_bias = sweep_bias_vector(
            self.cfg.sweep_attack_mode, self.sweep_value,
            self.cfg.sweep_combined_ft_ratio, self.cfg.sweep_torque_yaw_ratio)
        s = self.cfg.sweep_attack_start
        self.scenario = {
            'pattern': self.sweep_pattern_cur, 'attack_type': 'loe_combined',
            'attack_intensity': 1.0, 'attack_start_step': s,
            'attack_end_step': 99999, 'attack_bursts': [(s, 99999)],
            'disturbance_type': getattr(self.cfg, 'sweep_wind_type', 'none'),
            'wind_speed': float(getattr(self.cfg, 'sweep_wind_speed', 0.0)),
        }
        _u = 'N' if self.cfg.sweep_attack_mode == 'thrust' else 'Nm'
        self.get_logger().info(
            f'\n  [SWEEP] cell {self.sweep_cell_idx+1}/{len(self.sweep_cells)} '
            f'{self.cfg.sweep_attack_mode} b={self.sweep_value:.3f}{_u} '
            f'(tq_xy={self._sweep_bias[0]:.3f} th_n={self._sweep_bias[2]:.3f}) '
            f'policy={self.sweep_policy} pat={self.sweep_pattern_cur} '
            f'| ep {self.sweep_ep_in_cell+1}/{self.cfg.sweep_episodes}')
        self._send_scenario_cmd()
        self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
        self.attack_bursts = [(s, 99999)]
        self._cur_burst_start = 0; self._last_burst_end = None
        # hover 정책이면 원점 고정 호버
        if self.sweep_policy == 'hover':
            self._hover_pos[:] = 0.0; self._hover_yaw = 0.0
            self._hover_alt = -abs(self.cfg.flight_altitude)  # hover 셀은 기준고도 유지
            self.prev_action = 1
        else:
            self.prev_action = 0

    def _sweep_step_10hz(self, trajectory_sp):
        """고정정책 1스텝: UKF NIS + 공격토글 + done + CSV. 학습/탐험/push 없음."""
        cfg = self.cfg
        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0, offset=0.5)  # vel 저압축(log0.5)
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)              # gyro log1p 유지

        if self.step_count < cfg.learning_warmup_steps:
            self.step_count += 1
            return

        if self.sweep_policy == 'track':
            action = 0
        elif self.sweep_policy == 'hover':
            action = 1
        else:  # 'dhover{d}': 공격 시작 +d 스텝부터 호버 (b=0 셀은 track과 동일 동작)
            d = int(self.sweep_policy[6:])
            action = 1 if self.step_count >= self.cfg.sweep_attack_start + d else 0

        # 0→1 전환 순간 현재 위치/고도/요를 호버 셋포인트로 캡처 (본 RL 루프와 동일 semantics)
        if action == 1 and self.prev_action == 0:
            self._hover_pos[:] = self.cur_pos[:2]
            self._hover_alt = float(self.cur_pos[2])
            self._hover_yaw = float(self.cur_euler[2])

        done, term_reason = self._check_done(trajectory_sp)

        # ── 공격 토글 (단일 윈도우; b=0이면 무해) ──
        want_attack = self._is_attack_step(self.step_count)
        if want_attack and not self.attack_active_flag:
            # 이번 셀의 물리 바이어스 벡터를 그대로 주입(intensity=1 → ramp로 0→full).
            tq_xy, tq_z, th_n = self._sweep_bias
            self._send_attack_cmd(True, 'loe_combined', 1.0,
                bias_torque_xy=tq_xy, bias_torque_z=tq_z, bias_thrust_n=th_n)
            self.attack_active_flag = True
            self._cur_burst_start = self.step_count
        elif (not want_attack) and self.attack_active_flag:
            self._send_attack_cmd(False)
            self.attack_active_flag = False

        # ── 측정 로깅 ──
        sp = np.array(trajectory_sp[:3])
        gt_ned = np.array([self.gt_pos[1], self.gt_pos[0], -self.gt_pos[2]])
        gt_err = float(np.linalg.norm(gt_ned[:2] - sp[:2]))
        alt = float(-self.cur_pos[2] if self.cur_pos[2] < 0 else 0.0)
        self._sweep_detail_w.writerow([
            self.sweep_cell_idx, self.cfg.sweep_attack_mode, f'{self.sweep_value:.3f}',
            f'{self._sweep_bias[0]:.3f}', f'{self._sweep_bias[2]:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell, self.step_count,
            int(self.attack_active_flag),
            f'{nis_v_raw:.5f}', f'{nis_g_raw:.5f}', f'{nis_vel:.5f}', f'{nis_gyr:.5f}',
            f'{gt_err:.4f}', f'{alt:.4f}', action,
            term_reason if done else ''])

        if self.step_count % cfg.log_interval == 0:
            atk = '🔴ATK' if self.attack_active_flag else '⚪NRM'
            act = 'HOVER' if action == 1 else 'TRACK'
            self.get_logger().info(
                f'  [SWP {self.step_count:3d}] b={self.sweep_value:.3f} {atk} {act} | '
                f'NISraw v={nis_v_raw:.2f} g={nis_g_raw:.2f} | GT={gt_err:.2f}m alt={alt:.1f}m')

        self.prev_action = action
        self.step_count += 1
        if done:
            self._end_episode(term_reason)

    def _end_sweep_episode(self, reason):
        survived = (reason == 'timeout')
        crash_step = -1 if survived else self.step_count
        self._sweep_summary_w.writerow([
            self.sweep_cell_idx, self.cfg.sweep_attack_mode, f'{self.sweep_value:.3f}',
            f'{self._sweep_bias[0]:.3f}', f'{self._sweep_bias[2]:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell,
            int(survived), crash_step, reason, self.step_count])
        self._sweep_detail_f.flush(); self._sweep_summary_f.flush()
        surv = '✅survive' if survived else f'❌{reason}@{crash_step}'
        self.get_logger().info(
            f'  [SWEEP] b={self.sweep_value:.3f} {self.sweep_policy} '
            f'ep{self.sweep_ep_in_cell+1} → {surv}')

        self.sweep_ep_in_cell += 1
        if self.sweep_ep_in_cell >= self.cfg.sweep_episodes:
            self.sweep_ep_in_cell = 0
            self.sweep_cell_idx += 1
            if self.sweep_cell_idx >= len(self.sweep_cells):
                self.get_logger().info(f'\n{"#"*60}\n  [SWEEP] COMPLETE → '
                    f'{self.cfg.outdir}/sweep_detail.csv, sweep_summary.csv\n'
                    f'  분석: python sweep_aggregate.py {self.cfg.outdir}\n{"#"*60}')
                try:
                    self._sweep_detail_f.close(); self._sweep_summary_f.close()
                except Exception:
                    pass
                self.sim_mgr.stop(); raise SystemExit("Sweep complete")
        self._apply_reset(reason)


def _ensure_xrce_agent(cfg):
    """MicroXRCEAgent(PX4↔ROS2 브리지)가 떠 있도록 보장. 이미 실행 중이면 skip.
    isim으로 띄우든 직접 실행하든 /fmu/* 토픽이 보장됨. 실패해도 비치명(경고만)."""
    import shutil
    if not getattr(cfg, 'xrce_autostart', True):
        return
    try:
        r = subprocess.run(['pgrep', '-f', 'MicroXRCEAgent'],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            print('[XRCE] MicroXRCEAgent 이미 실행 중 → skip')
            return
    except Exception:
        pass
    exe = cfg.xrce_agent_cmd.split()[0]
    if shutil.which(exe) is None:
        print(f'[XRCE] ⚠ "{exe}" 를 PATH에서 못 찾음. /fmu 토픽이 안 뜰 수 있음.\n'
              f'        먼저 수동/`isim`으로 켜세요:  {cfg.xrce_agent_cmd}')
        return
    try:
        subprocess.Popen(cfg.xrce_agent_cmd.split(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         preexec_fn=os.setsid)
        print(f'[XRCE] started: {cfg.xrce_agent_cmd}')
        pytime.sleep(2.0)
    except Exception as e:
        print(f'[XRCE] 기동 실패(무시하고 진행): {e}')


def main():
    cfg = Config()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='bias-sweep 모드(학습 OFF)')
    ap.add_argument('--headless', dest='headless', action='store_true', default=None)
    ap.add_argument('--sweep-mode', choices=['combined', 'torque', 'thrust'], default=None,
                    help='sweep 공격 채널(미지정 시 config값)')
    ap.add_argument('--sweep-values', default=None,
                    help='쉼표구분 bias값 (미지정 시 모드별 권장 grid)')
    ap.add_argument('--outdir', default=None, help='결과 폴더(미지정 시 config값)')
    ap.add_argument('--agent', choices=['rhukf', 'adam'], default=None,
                    help='학습 옵티마이저 선택 (rhukf=제안 | adam=Adam+Huber baseline)')
    ap.add_argument('--speed', type=float, default=None,
                    help='sim 속도배율(>1=실시간보다 빠름; lockstep 한계까지. 2~4부터)')
    ap.add_argument('--ramp', type=float, default=None,
                    help='attack_ramp_duration(s) override (미지정 시 config값)')
    ap.add_argument('--episodes', type=int, default=None,
                    help='sweep_episodes(셀당 반복) override (미지정 시 config값)')
    ap.add_argument('--ft-ratio', dest='ft_ratio', type=float, default=None,
                    help='combined 모드 추력/토크비 sweep_combined_ft_ratio override (th_n=ft_ratio·b)')
    ap.add_argument('--hover-delays', dest='hover_delays', default=None,
                    help='쉼표구분 dhover 지연 스텝 목록 override (예: 1,2,3)')
    ap.add_argument('--log-zu', dest='log_zu', action='store_true',
                    help='UKF 오프라인 튜닝용 (z,u) 시계열을 outdir/zu_log.npz로 저장')
    ap.add_argument('--sweep-pattern', dest='sweep_pattern', default=None,
                    help='track/dhover 셀 비행패턴 override (aggressive|circle|figure8|waypoint)')
    ap.add_argument('--sweep-wind-type', dest='sweep_wind_type', default=None,
                    choices=['none', 'wind_constant', 'wind_gust', 'wind_turbulence'],
                    help='sweep 외란 타입 override (기본 none)')
    ap.add_argument('--sweep-wind-speed', dest='sweep_wind_speed', type=float, default=None,
                    help='sweep 바람 속도 m/s override (force≈0.031·v²N; 8≈2N 15≈7N)')
    _args, _ = ap.parse_known_args()

    # 모드별 권장 grid (값 미지정 시) — 각 모드의 '붕괴 경계'를 브래킷
    _grids = {
        'combined': [0.8, 1.0, 1.2, 1.3, 1.5, 1.7],          # Nm 토크 (추력=ft_ratio·b:4~8.5N); 토크를 flip영역까지
        'torque':   [1.0, 1.2, 1.3, 1.4, 1.5, 1.7],          # Nm; 밴드 [1.3,1.5) 정밀화
        'thrust':   [8.0, 12.0, 14.0, 16.0, 20.0, 25.0],     # N; 고도붕괴(~14N=권한포화) 브래킷
    }
    if _args.sweep:
        cfg.sweep_mode = True
    if _args.headless:
        cfg.headless = True
    if _args.sweep_mode:
        cfg.sweep_attack_mode = _args.sweep_mode
        if _args.sweep_values is None:
            cfg.sweep_values = _grids[_args.sweep_mode]
    if _args.sweep_values:
        cfg.sweep_values = [float(x) for x in _args.sweep_values.split(',')]
    if _args.agent:
        cfg.agent_type = _args.agent
        if _args.outdir is None:                 # 미지정 시 에이전트별 폴더로 분리(비교용)
            cfg.outdir = f'results_{_args.agent}'
    if _args.speed is not None:
        cfg.sim_speed_factor = float(_args.speed)
    if _args.ramp is not None:
        cfg.attack_ramp_duration = float(_args.ramp)
    if _args.episodes is not None:
        cfg.sweep_episodes = int(_args.episodes)
    if _args.ft_ratio is not None:
        cfg.sweep_combined_ft_ratio = float(_args.ft_ratio)
    if _args.hover_delays is not None:
        cfg.sweep_hover_delays = tuple(int(x) for x in _args.hover_delays.split(','))
    if getattr(_args, 'log_zu', False):
        cfg.log_zu = True
    if _args.sweep_pattern is not None:
        cfg.sweep_pattern = _args.sweep_pattern
    if _args.sweep_wind_type is not None:
        cfg.sweep_wind_type = _args.sweep_wind_type
    if _args.sweep_wind_speed is not None:
        cfg.sweep_wind_speed = float(_args.sweep_wind_speed)
    if _args.outdir:
        cfg.outdir = _args.outdir
        os.makedirs(cfg.outdir, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*deprecated.*")

    # 정밀도: 전역 FP32 고정 + forward만 스코프 TF32 (use_tf32_forward)
    from rl.network import apply_tf32_config
    _en, _sup = apply_tf32_config(cfg)
    print(f"[TF32] forward TF32 = {'ON' if _en else 'off'} (요청={cfg.use_tf32_forward}, "
          f"GPU지원={'yes' if _sup else 'no'}) | 행렬연산은 FP32 유지")

    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.suppress_errors = True   # 컴파일 실패해도 eager 폴백(런 안 죽음)
        # inductor 컴파일 실패 시 찍히는 WARNING 트레이스백 묵음 (Isaac 번들 토치에서 흔함)
        import logging as _lg
        for _n in ("torch._dynamo", "torch._inductor", "torch._functorch",
                   "torch._dynamo.convert_frame", "torch._inductor.compile_fx"):
            _lg.getLogger(_n).setLevel(_lg.ERROR)
        try:
            torch._logging.set_logs(dynamo=_lg.ERROR, inductor=_lg.ERROR)
        except Exception:
            pass

    import logging
    logging.getLogger('rclpy').setLevel(logging.WARNING)

    _ensure_xrce_agent(cfg)   # PX4 /fmu/* ↔ ROS2 브리지 보장 (이륙/공격주입/센서에 필수)

    rclpy.init()
    node = OnlineRLNode(cfg)

    # ── PX4 배터리 Failsafe는 최초 1회 pxh에서 끄고 저장하면 영구 유지(매 실행 reminder는 노이즈라 제거) ──
    #     param set COM_LOW_BAT_ACT 0
    #     param set COM_DISARM_LAND -1
    #     param save
    #   (미설정이면 에피소드 중 배터리 failsafe로 disarm될 수 있음)

    node.agent.warmup_compile()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as e:
        node.get_logger().info(f'Shutdown: {e}')
    finally:
        node.sim_mgr.stop()
        try:
            from torch._inductor.async_compile import shutdown_compile_workers
            shutdown_compile_workers()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
