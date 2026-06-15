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
from swrl_config import Config, sample_episode_scenario
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
                 log_dir='./results', sim_launcher='~/isaacsim/python.sh'):
        self.sim_script = sim_script
        self.headless = headless
        self.log_dir = log_dir
        self.sim_launcher = os.path.expanduser(sim_launcher)
        self.process = None
        self._log_file = None
        os.makedirs(log_dir, exist_ok=True)

    def start(self):
        launcher_path = os.path.expanduser('~/isaacsim/python.sh')
        cmd = [launcher_path, self.sim_script]
        if self.headless:
            cmd.append('--headless')
        else:
            cmd.append('--no-headless')
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
            log_dir=cfg.outdir, sim_launcher=cfg.sim_launcher)
        self.sim_mgr.start()
        self.get_logger().info('  Waiting 20s for Isaac Sim + PX4 SITL startup...')
        pytime.sleep(20)

        # ── Publishers ──
        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.pub_attack = self.create_publisher(String, '/attack_config', 10)
        self.pub_scenario = self.create_publisher(String, '/scenario_config', 10)
        self.pub_sim_ctrl = self.create_publisher(String, '/sim_control', 10)

        # ── Subscribers ──
        self.create_subscription(SensorGps, '/sim/sensor_gps', self._cb_gps, qos)
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self._cb_sensor, qos)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._cb_odometry, qos)
        self.create_subscription(VehicleThrustSetpoint, '/fmu/out/vehicle_thrust_setpoint', self._cb_thrust, qos)
        self.create_subscription(VehicleTorqueSetpoint, '/fmu/out/vehicle_torque_setpoint', self._cb_torque, qos)
        self.create_subscription(GroundTruthOdometry, '/gt/odometry', self._cb_gt, qos)

        # ── UKF + Agent ──
        self.calib = load_calibration('calibration.json')
        self._ukf_q_gate = getattr(cfg, 'ukf_q_gate_gyro', 0.0)
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
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

    def _send_attack_cmd(self, active, attack_type='none', intensity=0.0):
        msg = String(); msg.data = json.dumps({'active': active, 'type': attack_type,
            'intensity': intensity, 'ramp_duration': self.cfg.attack_ramp_duration})
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
        self.cur_pos[:] = 0; self.cur_vel[:] = 0; self.cur_euler[:] = 0
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
        self.is_ukf_initialized = False; self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)
        self.continuous_fp_count = 0
        self.drift_counter = 0
        self.attack_bursts = []
        self._cur_burst_start = 0
        self._last_burst_end = None

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
        self.sim_mgr.stop(); raise SystemExit("Training complete")

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
        # ── Heartbeat ──
        if self.flight_state not in ('IDLE', 'HARD_RESET'):
            if not self._check_heartbeat():
                self.get_logger().error('  💀 Heartbeat lost → HARD_RESET')
                self._trigger_hard_reset(); return

        # ── Offboard 유지 (★ 50Hz 규칙 발행 = PX4 안정) ──
        if self.flight_state in ('SOFT_RECOVERY', 'TAKEOFF', 'STABILIZE', 'LEARNING'):
            self._publish_offboard()

        # ── IDLE ──
        if self.flight_state == 'IDLE':
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
                self.get_logger().warn('  [HARD] Restarting simulator...')
                self.sim_mgr.restart(); self.hard_reset_count += 1
                self.last_gt_time = pytime.time()
            if self.init_counter >= int(20.0 / self.step_dt):
                if self._check_heartbeat():
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self._start_new_episode()
                    self.flight_state = 'TAKEOFF'
                elif self.init_counter >= int(40.0 / self.step_dt):
                    self.get_logger().error('  [HARD] Retry...'); self.init_counter = 0

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
                              -abs(self.cfg.flight_altitude), self._hover_yaw, 0.0, 0.0, 0.0)
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
        if not self.is_ukf_initialized:
            self.ukf.x[0:3] = gps_ned; self.ukf.x[3:6] = self.cur_euler
            self.ukf.x[6:9] = vel_ned; self.ukf.x[9:12] = self.cur_gyro
            self.is_ukf_initialized = True
        self.last_res, self.last_Pzz = self.ukf.step(z_9d, u_phys)

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
        except Exception as e:
            self.get_logger().error(f"  [LEARN ERROR] {e}")
        finally:
            self._is_learning_bg = False

    # ══════════════════════════════════════════════════════════
    #  10Hz RL Step
    # ══════════════════════════════════════════════════════════
    def _rl_step_10hz(self, trajectory_sp):
        cfg = self.cfg

        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0)
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)
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
        reward = calculate_reward(
            self.prev_action if self.prev_action is not None else 0,
            self.attack_active_flag,
            min(attack_delay, 5),      # Phase0: escalation 캡 (분산 폭주 차단; -1-0.2·5²=-6에서 포화)
            min(recovery_delay, 5),    # Phase0: escalation 캡 (raw delay는 logical_done 판정에 그대로 사용)
            rc=self.cfg.reward,
        )
        # 물리적 crash = 결과 기반 종단 페널티 (추종오차 대신 '추락=나쁨' outcome anchor).
        #   강도가 '생존 가능 띠'에 들어와 있을 때만 깨끗한 신호가 됨(즉사 구간이면 노이즈).
        if done and term_reason in PHYSICAL_TERMINALS:
            reward += self.cfg.reward.terminal_penalty
        self.episode_reward += reward

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
            self._send_attack_cmd(True, self.scenario['attack_type'], self.scenario['attack_intensity'])
            self.attack_active_flag = True
            self._cur_burst_start = self.step_count
            self.get_logger().warn(
                f'  🚨 Attack ON (burst) @ step {self.step_count}: {self.scenario["attack_type"]} '
                f'(int={self.scenario["attack_intensity"]:.3f})')
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
    #  Episode End
    # ══════════════════════════════════════════════════════════
    def _end_episode(self, reason):
        self._send_attack_cmd(False); self.attack_active_flag = False

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
        emojis = {'crash_drift': '⚠️ DRIFT', 'crash_altitude': '💀 CRASH',
                  'crash_flip': '🔥 FLIP', 'timeout': '⏱️ TIMEOUT',
                  'detection_failed': '🙈 MISS', 'recovery_failed': '🔒 STUCK',
                  'excessive_fp': '🤡 PANIC'}
        reset_label = {'crash_flip': 'HARD', 'crash_altitude': 'WARM'}.get(reason, 'SOFT')

        self.get_logger().info(
            f'\n  ┌─ Ep {self.episode}: {emojis.get(reason, reason)} → {reset_label} reset\n'
            f'  │ R={self.episode_reward:.1f} Steps={self.step_count} Loss={avg_loss:.4f}\n'
            f'  │ ε={eps:.3f} P={p_init:.5f}\n'
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
        """셀 리스트 구성 + CSV 오픈. 셀 = (α, policy, pattern).
        baseline 2개(무공격 aggressive/hover) + α마다 (track, hover)."""
        import csv
        cfg = self.cfg
        cells = [(0.0, 'track', cfg.sweep_pattern), (0.0, 'hover', 'hover')]
        for a in cfg.sweep_alphas:
            cells.append((float(a), 'track', cfg.sweep_pattern))
            cells.append((float(a), 'hover', 'hover'))
        self.sweep_cells = cells
        self.sweep_cell_idx = 0
        self.sweep_ep_in_cell = 0
        self.sweep_alpha, self.sweep_policy, self.sweep_pattern_cur = cells[0]

        os.makedirs(cfg.outdir, exist_ok=True)
        self._sweep_detail_f = open(os.path.join(cfg.outdir, 'sweep_detail.csv'), 'w', newline='')
        self._sweep_detail_w = csv.writer(self._sweep_detail_f)
        self._sweep_detail_w.writerow([
            'cell_idx', 'alpha', 'policy', 'pattern', 'episode', 'step', 'attack_active',
            'nis_v_raw', 'nis_g_raw', 'nis_v_scaled', 'nis_g_scaled',
            'gt_err', 'alt', 'action', 'crash_reason'])
        self._sweep_summary_f = open(os.path.join(cfg.outdir, 'sweep_summary.csv'), 'w', newline='')
        self._sweep_summary_w = csv.writer(self._sweep_summary_f)
        self._sweep_summary_w.writerow([
            'cell_idx', 'alpha', 'policy', 'pattern', 'episode',
            'survived', 'crash_step', 'crash_reason', 'steps'])
        self.get_logger().info(
            f'\n{"#"*60}\n  [SWEEP] {len(cells)} cells × {cfg.sweep_episodes} ep '
            f'| α={cfg.sweep_alphas}\n  attack: loe_combined @step{cfg.sweep_attack_start}, '
            f'ramp={cfg.attack_ramp_duration}s | q_gate={self._ukf_q_gate}\n{"#"*60}')

    def _start_sweep_episode(self):
        cell = self.sweep_cells[self.sweep_cell_idx]
        self.sweep_alpha, self.sweep_policy, self.sweep_pattern_cur = cell
        s = self.cfg.sweep_attack_start
        self.scenario = {
            'pattern': self.sweep_pattern_cur, 'attack_type': 'loe_combined',
            'attack_intensity': self.sweep_alpha, 'attack_start_step': s,
            'attack_end_step': 99999, 'attack_bursts': [(s, 99999)],
            'disturbance_type': 'none', 'wind_speed': 0.0,
        }
        self.get_logger().info(
            f'\n  [SWEEP] cell {self.sweep_cell_idx+1}/{len(self.sweep_cells)} '
            f'α={self.sweep_alpha:.2f} policy={self.sweep_policy} pat={self.sweep_pattern_cur} '
            f'| ep {self.sweep_ep_in_cell+1}/{self.cfg.sweep_episodes}')
        self._send_scenario_cmd()
        self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
        self.attack_bursts = [(s, 99999)]
        self._cur_burst_start = 0; self._last_burst_end = None
        # hover 정책이면 원점 고정 호버
        if self.sweep_policy == 'hover':
            self._hover_pos[:] = 0.0; self._hover_yaw = 0.0
            self.prev_action = 1
        else:
            self.prev_action = 0

    def _sweep_step_10hz(self, trajectory_sp):
        """고정정책 1스텝: UKF NIS + 공격토글 + done + CSV. 학습/탐험/push 없음."""
        cfg = self.cfg
        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0)
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)

        if self.step_count < cfg.learning_warmup_steps:
            self.step_count += 1
            return

        action = 0 if self.sweep_policy == 'track' else 1

        done, term_reason = self._check_done(trajectory_sp)

        # ── 공격 토글 (단일 윈도우; α=0이면 무해) ──
        want_attack = self._is_attack_step(self.step_count)
        if want_attack and not self.attack_active_flag:
            self._send_attack_cmd(True, 'loe_combined', self.sweep_alpha)
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
            self.sweep_cell_idx, f'{self.sweep_alpha:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell, self.step_count,
            int(self.attack_active_flag),
            f'{nis_v_raw:.5f}', f'{nis_g_raw:.5f}', f'{nis_vel:.5f}', f'{nis_gyr:.5f}',
            f'{gt_err:.4f}', f'{alt:.4f}', action,
            term_reason if done else ''])

        if self.step_count % cfg.log_interval == 0:
            atk = '🔴ATK' if self.attack_active_flag else '⚪NRM'
            act = 'HOVER' if action == 1 else 'TRACK'
            self.get_logger().info(
                f'  [SWP {self.step_count:3d}] α={self.sweep_alpha:.2f} {atk} {act} | '
                f'NISraw v={nis_v_raw:.2f} g={nis_g_raw:.2f} | GT={gt_err:.2f}m alt={alt:.1f}m')

        self.prev_action = action
        self.step_count += 1
        if done:
            self._end_episode(term_reason)

    def _end_sweep_episode(self, reason):
        survived = (reason == 'timeout')
        crash_step = -1 if survived else self.step_count
        self._sweep_summary_w.writerow([
            self.sweep_cell_idx, f'{self.sweep_alpha:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell,
            int(survived), crash_step, reason, self.step_count])
        self._sweep_detail_f.flush(); self._sweep_summary_f.flush()
        surv = '✅survive' if survived else f'❌{reason}@{crash_step}'
        self.get_logger().info(
            f'  [SWEEP] α={self.sweep_alpha:.2f} {self.sweep_policy} '
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


def main():
    cfg = Config()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='α-sweep 모드(학습 OFF)')
    ap.add_argument('--headless', dest='headless', action='store_true', default=None)
    _args, _ = ap.parse_known_args()
    if _args.sweep:
        cfg.sweep_mode = True
    if _args.headless:
        cfg.headless = True

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
            os.killpg(os.getpgrp(), signal.SIGTERM)
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
