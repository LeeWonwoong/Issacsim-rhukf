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
import random
import signal
import subprocess
import time as pytime
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleAttitudeSetpoint,
    SensorCombined, VehicleOdometry,
    VehicleThrustSetpoint, VehicleTorqueSetpoint, SensorGps
)
try:
    from px4_msgs.msg import ActuatorAttack     # 2026-08-08: allocator 경로 공격 주입(DDS)
except Exception:                               # noqa: BLE001
    ActuatorAttack = None                       # px4_msgs 미재빌드 시 graceful
from nav_msgs.msg import Odometry as GroundTruthOdometry
from std_msgs.msg import String

import torch
from swrl_config import Config, sample_episode_scenario, sweep_bias_vector
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration, to_physical_u
from env.reward import calculate_reward
from rl.agent import OnlineRHUKFAgent


# ★2026-08-23: 결과성=경로이탈(gt_err), 추락은 학습에 안 넣음. 모든 crash(altitude/flip/drift)를
#   truncation 처리(페널티 X, bootstrap 유지) → 추락은 잘린 timeout일 뿐, 학습 타깃 아님.
#   탐지 reward(TP/TN/FP/FN)만 정책을 학습. 공격 중 hover 유도는 TP가 담당(추락 페널티 불요).
# ★2026-09-07 버그수정: 빈 튜플이어서 추락해도 terminated=False = 절벽(에피 절단)이
#   Q 부트스트랩에 반영되지 않던 상태였다. v3.1(ramp 결과성 복원)의 전제라 반드시 채운다.
#   timeout·논리종료는 여기 넣지 않는다(truncation — 부트스트랩 유지).
PHYSICAL_TERMINALS = ('crash_drift', 'crash_altitude', 'crash_flip')

# env OBS_NORM=1 → 관측 압축 clip 3.5 후 /3.5 정규화([0,1]). off면 clip3.0([0,3], 기존).
_OBS_NORM = os.environ.get('OBS_NORM', '') == '1'
_OBS_CLIP = 3.5 if _OBS_NORM else 3.0
_OBS_DIV = 3.5 if _OBS_NORM else 1.0
# env OBS_CLIP=3.5 → 클립 상한만 변경(나눗셈 없음). 미설정시 기존과 완전 동일 (2026-08-29 변형실험 V3)
if os.environ.get('OBS_CLIP'):
    _OBS_CLIP = float(os.environ['OBS_CLIP'])
# ★2026-08-24 POMDP화: 에피별 랜덤 COM 토크바이어스(UKF 미모델링) + GPS vel 노이즈.
#   COM_BIAS_STD=0.10 → 매 에피 τ_com~U(-0.10,0.10)N·m(roll/pitch) → gyro 바닥↑·변동 → 약공격 묻힘(POMDP).
#   = 배터리/페이로드 COM 변동(현실적, ~7mm). VEL_NOISE=0.3 → vel 바닥↑.
_COM_BIAS_STD = float(os.environ.get('COM_BIAS_STD', '0') or 0)
_VEL_NOISE = float(os.environ.get('UKF_VEL_NOISE', '0') or 0)


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
        # ★ 2026-08-17 모델오차 주입: UKF calib 에만 오차(플랜트 run_sim 은 calibration.json 그대로).
        #   UKF≠플랜트 → 기동 중 모델오차 innovation 생성(실기 현실화). 센서노이즈론 안 생기는 aliasing.
        #   UKF_CALIB_ERR="ctq:0.1,cth:0.05,m:0.05" → UKF C_torque×1.1, C_thrust×1.05, mass×1.05.
        _ce = os.environ.get('UKF_CALIB_ERR', '').strip()
        if _ce:
            import copy as _copy
            self.calib = _copy.deepcopy(self.calib)
            _e = dict(kv.split(':') for kv in _ce.split(',') if ':' in kv)
            _ctq = 1.0 + float(_e.get('ctq', 0)); _cth = 1.0 + float(_e.get('cth', 0)); _m = 1.0 + float(_e.get('m', 0))
            _iI = 1.0 + float(_e.get('i', 0))   # ★2026-09-09 관성 오차 (급기동 gyro 예측 어긋남 = aliasing)
            for _k in ('C_torque_x', 'C_torque_y', 'C_torque_z', 'C_torque_xy'):
                if _k in self.calib:
                    self.calib[_k] = float(self.calib[_k]) * _ctq
            if 'C_thrust' in self.calib:
                self.calib['C_thrust'] = float(self.calib['C_thrust']) * _cth
            if 'drone' in self.calib and 'mass' in self.calib['drone']:
                self.calib['drone']['mass'] = float(self.calib['drone']['mass']) * _m
            if _iI != 1.0 and 'drone' in self.calib:
                for _ik in ('Ixx', 'Iyy', 'Izz'):
                    if _ik in self.calib['drone']:
                        self.calib['drone'][_ik] = float(self.calib['drone'][_ik]) * _iI
            self.get_logger().warn(f'[UKF_CALIB_ERR] UKF 모델오차: C_torque×{_ctq:.2f} C_thrust×{_cth:.2f} mass×{_m:.2f} I×{_iI:.2f} (플랜트 불변)')
        # ★ 2026-08-18 진단: UKF_NO_COUPLING=1 → UKF 에서 추력-토크 커플링 보정항 제거(순수 12차원).
        #   플랜트(run_sim COM offset)는 커플링 유지 → UKF 가 PX4 트림토크를 실토크로 오해 = 표준모델 한계 재현.
        if os.environ.get('UKF_NO_COUPLING', '') == '1':
            import copy as _cp2
            self.calib = _cp2.deepcopy(self.calib)
            _rm = self.calib.pop('torque_thrust_coupling', None)
            self.get_logger().warn(f'[UKF_NO_COUPLING] 커플링 보정항 제거(순수 12차원). 제거값={_rm}')
        self._ukf_q_gate = getattr(cfg, 'ukf_q_gate_gyro', 0.0)
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
        # ── UKF 오프라인 튜닝용 (z,u) 로깅 (opt-in; sim 1회만 돌려 수집) ──
        self._log_zu = bool(getattr(cfg, 'log_zu', False))
        self._zu_rows = []
        # ── SysId 재캘리브레이션용 GT 로깅 (opt-in; --log-sysid) ──
        #   zu_log 는 GPS 노이즈가 실린 관측이라 미분 기반 게인적합에 못 쓴다(SNR<1).
        #   여기서는 GT 속도 + IMU(specific force/gyro) + PX4 명령 setpoint 를 50Hz 로 받아
        #   calibrate_sysld.py 가 기대하는 npz 포맷 그대로 저장한다.
        self._log_sysid = bool(getattr(cfg, 'log_sysid', False))
        self._sysid_rows = []
        if getattr(cfg, 'agent_type', 'rhukf') == 'adam':
            from rl.agent_adam import OnlineAdamAgent
            self.agent = OnlineAdamAgent(cfg)
        else:
            self.agent = OnlineRHUKFAgent(cfg)
        # ★2026-09-09 LOAD_MODEL=경로.pt → 학습된 정책 로드 + greedy 추론 모드 (시연/평가)
        _lm = os.environ.get('LOAD_MODEL', '').strip()
        if _lm:
            self.agent.load(_lm)
            try: self.agent.eps = 0.0
            except Exception: pass
            self.get_logger().warn(f'[LOAD_MODEL] {_lm} 로드 — greedy 추론 모드')
        self.window_buffer = collections.deque(maxlen=cfg.window_size)

        # ── Sensor state ──
        self.cur_accel = np.zeros(3); self.cur_gyro = np.zeros(3)
        # ★2026-08-20 센서 σ 정합: gyro 노이즈(실기 실측) + 전역 스케일(스윕용).
        self._sensor_noise_scale = float(os.environ.get('SENSOR_NOISE_SCALE', '1.0'))
        # ── 속도변조 (env SPEED_MOD_AMP>0): 같은 도형을 가감속하며 주행 (2026-08-21 실험용) ──
        #   OFF(기본 0)면 궤적은 현재와 수학적으로 동일. 가감속이 gyro 추정오차를 얼마나 만드는지 측정용.
        self._speed_mod_amp = float(os.environ.get('SPEED_MOD_AMP', '0') or 0.0)
        self._speed_mod_freq = float(os.environ.get('SPEED_MOD_FREQ', '0.5') or 0.5)
        self._alt_mod_amp = float(os.environ.get('ALT_MOD_AMP', '0') or 0.0)    # ★2026-09-08 고도변조(m)
        self._alt_mod_freq = float(os.environ.get('ALT_MOD_FREQ', '0.3') or 0.3)
        #   ★2026-08-27 1.0→0.5 (버그 수정): freq 1.0 은 접선가속 피크 v·amp·2πf = 3.77 m/s² 로
        #   MPC_ACC_HOR 3.0 을 초과 — 추종 불가 명령 → 상시 가속포화 → 사냥(hunting) 진동
        #   (실측: 실제 |v| 중앙 0.43 vs 설계 1.2, 경로이탈 90p 0.65 m). 0.5 는 피크 1.88,
        #   구심 합성 2.74 < 3.0 으로 봉투 안 추종 가능한 최대 가감속. (08-26 의 0.25→1.0 은 과했음)
        self._traj_t = 0.0
        self._gyro_sigma = np.array(getattr(cfg, 'gyro_sensor_sigma', [0.08, 0.07, 0.025]))
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
        # ★ 2026-08-13 궤적 명령 후처리 옵션 (env). 둘 다 sim·실기 동일 적용해야 비교 성립.
        self._yaw_slew_rate = float(os.environ.get('YAW_SLEW_RATE', '1.57') or 0.0)  # rad/s, 0=끔
        self._accel_ff = os.environ.get('ACCEL_FF', '0') not in ('', '0')
        self._yaw_cmd_prev = None; self._vel_cmd_prev = None
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.is_ukf_initialized = False
        self.attack_active_flag = False

        # ── 공격 버스트 상태 ──
        self.attack_bursts = []
        self._cur_burst_start = 0
        self._cur_burst_bias = (0.0, 0.0)
        self._last_burst_end = None
        self.drift_counter = 0

        # ── HOVER 위치 고정 (★ 떨림 방지) ──
        self._hover_pos = np.zeros(2)  # HOVER 전환 시 위치 저장
        self._hover_alt = 0.0          # ★ HOVER 전환 시 고도 저장 (고도 스냅 과도 제거)
        self._hover_yaw = 0.0  # ★ HOVER 전환 시 Yaw 저장용

        # ── Detection tracking ──
        self.first_hover_step = None
        self.hover_before_attack_count = 0
        self._hover_latched = False       # [latch 2026-07-09] 공격 대응 중 최초 hover 고도 고정
        self._ep_relapse = 0              # [진단] 공격중 hover→track 재발(플리커) 횟수
        self._ep_min_alt = 999.0          # [진단] 공격중 최저 고도(m) — 침하 확인용

        # ── Evaluation mode ──
        self.eval_mode = bool(os.environ.get('FORCE_EVAL',''))   # FORCE_EVAL=1: 전 에피 greedy·learn OFF (추론 시연)
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

        # ★ 50Hz(sim) 타이머 — wall 주기를 sim_speed_factor 로 나눠 speed N 에서 50N Hz(wall) 발화.
        #   그래야 Isaac 이 N배로 돌 때 매 틱 sim 이 정확히 step_dt(0.02s) 전진 = speed1 과 동일 native
        #   케이던스(UKF predict dt·gyro 업데이트·GPS 10Hz·RL 10Hz 전부 보존). RTF 가 N 을 지속하는
        #   한 speed 불변. (구: 고정 0.02 wall → speed N 에서 sim 0.02N/틱 = 예측 1/N 지연 버그.)
        _sf = float(getattr(self.cfg, 'sim_speed_factor', 1.0) or 1.0)
        self.timer = self.create_timer(self.step_dt / max(_sf, 1.0), self._tick)
        if _sf > 1.0:
            self.get_logger().info(f'  ⏩ sim-time 타이머: wall {1.0/(self.step_dt/_sf):.0f}Hz '
                                   f'= sim 50Hz @ speed{_sf:.1f} (native 케이던스 보존)')

        self.last_z_var = 0.0  # Z 분산 저장용
        self.last_kgain = 0.0; self.last_pmax = 0.0; self.last_innov = 0.0; self.last_argmax_flip = 0.0; self.last_qmax = 0.0; self.last_nis = 0.0  # Step1 진단

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
        if self._log_sysid and self.gt_pos[2] > 1.0:      # 지상/이륙 과도 제외
            #  IMU 레이트(≈PX4 250Hz)로 기록 — 50Hz GT 콜백에 걸면 자세루프 토크명령이 앨리어싱된다.
            #  vel/euler 는 GT(50Hz, 저주파 성분만 쓰므로 무해), accel/gyro/명령은 PX4 고속.
            self._sysid_rows.append(np.concatenate([
                [msg.timestamp * 1e-6],
                [self.gt_vel[1], self.gt_vel[0], -self.gt_vel[2]],
                self.cur_euler, self.cur_accel, self.cur_gyro,
                self.cur_thrust, self.cur_torque,
                [float(self.episode), float(bool(getattr(self, 'attack_active_flag', False)))],
            ]))

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
        # ★2026-08-27 sim 시간 (run_sim 이 header.stamp 에 sim_time 을 실음)
        self._gt_sim_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
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
        self.pub_att = self.create_publisher(VehicleAttitudeSetpoint, _fmu('/fmu/in/vehicle_attitude_setpoint'), q)
        self.pub_cmd = self.create_publisher(VehicleCommand, _fmu('/fmu/in/vehicle_command'), q)
        # ── 공격 주입(DDS→allocator, 2026-08-08) ── 외부 wrench 폐기, 실기와 동일 경로 ──
        #  online_rl 이 정규화 δ 를 /fmu/in/actuator_attack 으로 발행 → PX4 SITL ControlAllocator
        #  가 c[0] 에 가산 → Pegasus → Isaac. ATTACK_INJECTION.md §6-2. (PX4 ATK_EN=1 필요)
        self.pub_actuator_attack = (
            self.create_publisher(ActuatorAttack, _fmu('/fmu/in/actuator_attack'), q)
            if ActuatorAttack is not None else None)
        if self.pub_actuator_attack is None:
            self.get_logger().warn('  [ATTACK] px4_msgs 에 ActuatorAttack 없음 — DDS 공격 비활성 '
                                   '(px4_msgs 재빌드 필요)')
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
        """Isaac GT 쿼터니언(ENU 관성 / FLU 바디) → **NED/FRD** ZYX 오일러각.

        ⚠ 2026-07-28 버그 수정. 기존에는 ENU/FLU 쿼터니언에 표준 ZYX 공식을 그대로 적용해
        얻은 각(= ENU 기준)을 NED 기준인 양 소비했다. 두 군데서 실제 피해가 났다:
          (1) UKF: 추력벡터를 틀린 수평 방향으로 회전 → 기동 중 vel NIS 오염.
              (실측: 이 관례로는 항력적합이 [0.13,0.03]/corr −0.3, 보정하면 [0.50,0.30]/corr −1.00
               = Pegasus 실제 설정 LinearDrag([0.50,0.30,0.0]) 와 정확히 일치)
          (2) _hover_yaw: ENU yaw 를 PX4 TrajectorySetpoint(NED yaw)로 보내 **호버 전환마다
              약 90° 요 슬루**를 명령. (실측 25회: 전환 후 3s |Δψ| 중앙 84.5°, 전환 전 0.7°)

        변환: R_NED_FRD = T_i · R_ENU_FLU · T_bᵀ,
              T_i = ENU→NED (x↔y, z 반전), T_b = FLU→FRD (y,z 반전).
        """
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])
        T_i = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        T_b = np.diag([1.0, -1.0, -1.0])
        Rn = T_i @ R @ T_b
        return [float(np.arctan2(Rn[2, 1], Rn[2, 2])),
                float(-np.arcsin(np.clip(Rn[2, 0], -1.0, 1.0))),
                float(np.arctan2(Rn[1, 0], Rn[0, 0]))]

    def _send_setpoint(self, x, y, z, yaw, vx=float('nan'), vy=float('nan'), vz=float('nan')):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]

        # ── ★ 2026-08-13 요 슬루레이트 제한 (env YAW_SLEW_RATE, rad/s; 0=끔) ──
        #   waypoint 코너에서 yaw=arctan2(dy,dx) 가 90° **순간 점프**를 명령했다.
        #   기체가 최대 권한으로 쫓아가 |ω| 4.9 rad/s 스파이크 → gyro NIS p95 4.28
        #   (=5× 공격 수준). 물리적으로 불가능한 명령이 만든 가짜 교란이다.
        #   목표 yaw 로 매 스텝 제한된 속도로만 접근시킨다(궤적 위치는 불변).
        yr = getattr(self, '_yaw_slew_rate', 0.0)
        if yr > 0.0 and np.isfinite(yaw):
            if getattr(self, '_yaw_cmd_prev', None) is None:
                self._yaw_cmd_prev = float(yaw)
            err = (float(yaw) - self._yaw_cmd_prev + np.pi) % (2*np.pi) - np.pi
            step = np.clip(err, -yr*self.step_dt, yr*self.step_dt)
            yaw = self._yaw_cmd_prev + step
            self._yaw_cmd_prev = float(yaw)
        msg.yaw = float(yaw)

        msg.velocity = [float(vx), float(vy), float(vz)]

        # ── ★ 2026-08-13 가속도 피드포워드 (env ACCEL_FF=1) ──
        #   PX4 TrajectorySetpoint.acceleration 표준 필드. 곡선 궤적(circle 은 구심가속
        #   Rω²=0.70 m/s²)을 위치오차로 만들지 않고 직접 준다 → 추종지연·정상상태오차↓.
        #   속도의 유한차분으로 근사(저주파 궤적이라 충분). NaN 속도면 미적용.
        acc = [float('nan')] * 3
        if getattr(self, '_accel_ff', False):
            v = np.array([vx, vy, vz], dtype=float)
            vp = getattr(self, '_vel_cmd_prev', None)
            if vp is not None and np.all(np.isfinite(v)) and np.all(np.isfinite(vp)):
                _dt_ff = max(getattr(self, '_dt_sim_last', self.step_dt), 1e-3)
                _a = (v - vp) / _dt_ff
                # ★2026-08-27 클립: 코너/곡률반전의 방향 불연속이 유한차분에서 30+ m/s² 임펄스가 됨
                #   → MPC_ACC_HOR(3.0) 노름 클립. (클립 없이 waypoint gt_err 0.79→1.72 악화 실측)
                _n = float(np.linalg.norm(_a[:2]))
                if _n > 3.0: _a[:2] *= 3.0 / _n
                _a[2] = float(np.clip(_a[2], -2.0, 2.0))
                acc = list(_a)
            self._vel_cmd_prev = v if np.all(np.isfinite(v)) else None
        msg.acceleration = [float(a) for a in acc]

        msg.timestamp = 0
        # ── 디버그: env SETPOINT_LOG=경로 → 보낸 setpoint 전량 기록 (2026-08-27 추종 진단용) ──
        _spl = os.environ.get('SETPOINT_LOG', '')
        if _spl:
            try:
                with open(_spl, 'a') as _f:
                    _f.write(f"{pytime.time():.3f},{x:.3f},{y:.3f},{z:.3f},{yaw:.3f},{vx:.3f},{vy:.3f},{vz:.3f}\n")
            except Exception:
                pass
        self.pub_traj.publish(msg)

    def _vehicle_cmd(self, command, p1, p2=0.0):
        msg = VehicleCommand(); msg.command = command; msg.param1 = float(p1); msg.param2 = float(p2)
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        msg.timestamp = 0
        self.pub_cmd.publish(msg)

    def _publish_offboard(self):
        off = OffboardControlMode()
        if getattr(self, '_att_hold_active', False):
            off.attitude = True            # ★자세-수평 홀드 모드 (HOVER_ATT)
        else:
            off.position = True
        off.timestamp = 0
        self.pub_offboard.publish(off)

    def _send_attitude_level(self):
        """★자세 홀드(방법A): 앵커로 향하는 공격적 반격 틸트를 기하 자세제어로 직접 명령.
           position 모드의 속도캡(2.0)을 우회 → 임무는 2.0 유지하면서 호버만 전권한.
           HOVER_ATT_LEVEL=1 이면 순수 수평(반격 없음, 진단용)."""
        _np = np; _m = math
        anc = getattr(self, '_hover_anchor', None)
        if anc is None:
            anc = _np.array([float(self.cur_pos[0]), float(self.cur_pos[1])])
        # 앵커로 향하는 수평 목표 가속 (PD). NED 수평 (N,E).
        e = _np.array([float(anc[0]) - float(self.cur_pos[0]),
                       float(anc[1]) - float(self.cur_pos[1])])
        v = _np.array([float(self.cur_vel[0]), float(self.cur_vel[1])])
        Kp = float(os.environ.get('HOVER_ATT_KP', '3.0') or 3.0)
        Kd = float(os.environ.get('HOVER_ATT_KD', '2.5') or 2.5)
        a_h = Kp * e - Kd * v                                     # 목표 수평가속 (m/s^2)
        if os.environ.get('HOVER_ATT_LEVEL'):                     # 진단: 반격 없음(순수 수평)
            a_h = _np.zeros(2)
        a_max = float(os.environ.get('HOVER_A_MAX', '12.0') or 12.0)  # 수평가속 상한 (전권한)
        an = float(_np.linalg.norm(a_h))
        if an > a_max: a_h = a_h * (a_max / an)
        g = 9.81
        # 고도 P → 수직 가속 보정
        _kpz = float(os.environ.get('HOVER_ALT_KP', '3.0') or 3.0)
        az = _kpz * float(self._hover_alt - self.cur_pos[2])      # NED: hover_alt(<0) 위, 오차 위로+
        # 추력 벡터 (NED, z 음수=위). t = [a_hN, a_hE, -(g+az)]
        t = _np.array([float(a_h[0]), float(a_h[1]), -(g + az)])
        tn = float(_np.linalg.norm(t))
        if tn < 1e-6: t = _np.array([0.0, 0.0, -g]); tn = g
        body_z = -t / tn                                          # 바디 z(아래축)= 추력 반대
        yaw = float(self._hover_yaw)
        bxc = _np.array([_m.cos(yaw), _m.sin(yaw), 0.0])
        by = _np.cross(body_z, bxc); byn = float(_np.linalg.norm(by))
        if byn < 1e-6: by = _np.array([0.0, 1.0, 0.0]); byn = 1.0
        by = by / byn
        bx = _np.cross(by, body_z)
        R = _np.column_stack([bx, by, body_z])                    # 바디→월드 회전행렬
        # 회전행렬 → 쿼터니언 [w,x,y,z]
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = _m.sqrt(tr + 1.0) * 2; qw = 0.25*S
            qx = (R[2,1]-R[1,2])/S; qy = (R[0,2]-R[2,0])/S; qz = (R[1,0]-R[0,1])/S
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = _m.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; qw=(R[2,1]-R[1,2])/S
            qx=0.25*S; qy=(R[0,1]+R[1,0])/S; qz=(R[0,2]+R[2,0])/S
        elif R[1,1] > R[2,2]:
            S = _m.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; qw=(R[0,2]-R[2,0])/S
            qx=(R[0,1]+R[1,0])/S; qy=0.25*S; qz=(R[1,2]+R[2,1])/S
        else:
            S = _m.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; qw=(R[1,0]-R[0,1])/S
            qx=(R[0,2]+R[2,0])/S; qy=(R[1,2]+R[2,1])/S; qz=0.25*S
        msg = VehicleAttitudeSetpoint()
        msg.q_d = [float(qw), float(qx), float(qy), float(qz)]
        _hov = float(os.environ.get('HOVER_THR', '0.35') or 0.35)
        thr = _hov * tn / g                                       # 추력 크기 ∝ |t|/g (호버=0.35)
        thr = max(0.15, min(0.85, float(thr)))
        msg.thrust_body = [0.0, 0.0, -thr]
        msg.timestamp = 0
        self.pub_att.publish(msg)

    def _get_alloc_B(self):
        """배분행렬 B (모터추력→wrench) 를 1회 로드해 캐시. allocation_B.npz (probe_rotor_geom.py)."""
        if not hasattr(self, '_alloc_B_cache'):
            try:
                import numpy as _np, os as _os
                _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'allocation_B.npz')
                self._alloc_B_cache = _np.load(_p)['B']
            except Exception as _e:
                self.get_logger().warn(f'[MOTOR] allocation_B.npz 로드 실패: {_e}')
                self._alloc_B_cache = None
        return self._alloc_B_cache

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
        self.pub_attack.publish(msg)   # ← 이제 run_sim GT 라벨링 전용 (물리 wrench 안 씀)

        # ── 물리 주입: DDS → allocator (2026-08-08) ──────────────────────────
        #  외부 wrench 폐기. 정규화 δ = 물리 bias / 권한. attack_type 이 축을 정한다.
        #  intensity(ramp 0~1)를 곱해 실제 크기. active=False 면 δ=0.
        if getattr(self, 'pub_actuator_attack', None) is not None:
            aa = ActuatorAttack()
            aa.timestamp = 0
            aa.active = bool(active)
            tqx = tqy = tqz = th = 0.0
            if active:
                s = float(intensity)
                # attack_type 별 축 매핑 (compute_attack_forces 와 동일 논리, 정규화 단위)
                nx = (s * gx) / self.cfg.attack_tq_authority
                ny = (s * gx) / self.cfg.attack_tq_authority
                nz = (s * gz) / self.cfg.attack_yaw_authority
                nt = (s * gt) / self.cfg.attack_th_authority
                if attack_type == 'tilt':
                    # ★ 2026-08-19 틸트 FDI: roll=gx, pitch=gz (둘 다 N·m). yaw·thrust=0.
                    tqx = (s * gx) / self.cfg.attack_tq_authority
                    tqy = (s * gz) / self.cfg.attack_tq_authority
                    tqz = 0.0
                    th = 0.0
                elif attack_type == 'loe_roll':    tqx = nx
                elif attack_type == 'loe_pitch':   tqy = -ny
                elif attack_type == 'loe_yaw':     tqz = nz
                elif attack_type == 'loe_thrust':  th = nt
                elif attack_type == 'loe_combined':
                    tqx, tqy, tqz, th = nx, -ny, nz, nt
                elif attack_type.startswith('motor'):
                    # ★ 2026-08-13 모터 침해 = wrench 커플링 (EADR 등가, ATTACK_COUPLING_PLAN.md).
                    #   attack_type='motorNM...' 의 각 숫자 = 침해 모터(1~4). 여러 개면 B 열 합.
                    #     motor1  단일   / motor12 대각(추력+요) / motor14 인접(추력+롤) / motor13 인접(추력+피치)
                    #   δ_wrench = (Σ_i B[:,i]) · s_mag,  s_mag = 모터당 추력 섭동 [N] = s·gx.
                    #   각 성분을 축별 권한으로 정규화해 c[0] 에 가산.
                    B = self._get_alloc_B()
                    if B is not None:
                        idxs = [max(0, min(3, int(ch) - 1)) for ch in attack_type[5:] if ch.isdigit()]
                        col = sum(B[:, i] for i in idxs) if idxs else B[:, 0]
                        s_mag = s * gx
                        dFz, dtx, dty, dtz = (col * s_mag)
                        tqx = dtx / self.cfg.attack_tq_authority
                        tqy = dty / self.cfg.attack_tq_authority
                        tqz = dtz / self.cfg.attack_yaw_authority
                        th = dFz / self.cfg.attack_th_authority
            aa.torque = [float(tqx), float(tqy), float(tqz)]
            aa.thrust = float(th)
            self.pub_actuator_attack.publish(aa)

    def _send_scenario_cmd(self, dist_override=None, ws_override=None):
        # override 로 바람을 물리적으로 on/off (시간창 게이팅용). 미지정이면 self.scenario 값.
        dist = self.scenario['disturbance_type'] if dist_override is None else dist_override
        ws = self.scenario['wind_speed'] if ws_override is None else ws_override
        msg = String(); msg.data = json.dumps({'disturbance_type': dist, 'wind_speed': float(ws)})
        self.pub_scenario.publish(msg)

    def _send_sim_reset(self):
        msg = String(); msg.data = 'reset'; self.pub_sim_ctrl.publish(msg)


    def _reengage_nearest(self):
        """현재 위치에서 궤적의 최근접점(x,y)과 그 phase 상태를 찾는다 — 재접근용.
           _compute_setpoint 를 phase 후보들로 프로브(상태 저장·복원). 찾은 phase 로 시계를 재동기.
           반환: (x, y, z) 최근접점. 부수효과: _sim_flight_t / _wp_s 를 최근접 phase 로 세팅."""
        import numpy as _np
        # 상태 스냅샷
        snap = (self._sim_flight_t, getattr(self,'_wp_s',0.0), self._traj_t, self.theta,
                self.tick_count, getattr(self,'_dt_sim_last', self.step_dt))
        pat = self.scenario['pattern'] if self.scenario else 'hover'
        # 궤적 1주기에 해당하는 phase 범위
        if pat == 'waypoint':
            kk = getattr(self.cfg,'wp_box_halfwidth',3.2)/5.0
            wps = _np.array([[0.,0.],[5.,0.],[5.,5.],[-5.,5.],[-5.,0.],[0.,0.]])*kk
            L = float(_np.linalg.norm(_np.diff(wps,axis=0),axis=1).sum())
            cands = _np.linspace(0, L, 240); usewp = True
        else:
            R = self.cfg.flight_radius; w = self.cfg.flight_omega
            period = 2*_np.pi/max(w,1e-6)
            if pat == 'scurve':
                Rs = getattr(self.cfg,'scurve_radius',1.1); v = R*w
                period = 2*(_np.pi*Rs*int(getattr(self.cfg,'scurve_arcs',4)))/max(v,1e-6)   # ★09-15: 호 수 하드코딩 4 → 설정값
            elif pat == 'aggressive':
                period = 4.0*float(getattr(self.cfg,'agg_phase_s',5.0))                        # ★09-15: 한 사이클 = 4 phase (구: 원 주기 사용 → 복귀 반원 phase 후보 누락)
            cands = _np.linspace(0, period, 240); usewp = False
        cur = _np.array([self.cur_pos[0], self.cur_pos[1]])
        best=(1e18, None, 0.0)
        self._dt_sim_last = 0.0   # 프로브 중 시계 전진 0
        for c in cands:
            if usewp: self._wp_s = float(c)
            else: self._sim_flight_t = float(c)
            sp = self._compute_setpoint()
            d = (sp[0]-cur[0])**2 + (sp[1]-cur[1])**2
            if d < best[0]: best=(d, (sp[0],sp[1],sp[2]), float(c))
        # 상태 복원 후, 최근접 phase 로 시계 재동기
        (self._sim_flight_t, self._wp_s, self._traj_t, self.theta,
         self.tick_count, self._dt_sim_last) = snap
        if usewp: self._wp_s = best[2]
        else: self._sim_flight_t = best[2]
        return best[1]


    def _nearest_xy(self):
        """재접근 진입 판정용 최근접점 (캐시). 매 틱 1회만 계산."""
        c = getattr(self, '_nearest_cache', None)
        if c is not None and c[0] == self.tick_count:
            return c[1]
        xy = self._reengage_nearest()
        self._nearest_cache = (self.tick_count, xy)
        return xy

    def _check_heartbeat(self):
        return (pytime.time() - self.last_gt_time) < self.heartbeat_timeout

    def _is_attack_step(self, step):
        """현재 step이 어떤 버스트 ON 구간 [s, e)에 속하는지. 버스트는 (s,e) 또는 (s,e,roll,pitch)."""
        for b in self.attack_bursts:
            s, e = b[0], b[1]
            if s <= step < e:
                return True
        return False

    def _current_burst_bias(self, step):
        """step이 속한 tilt 버스트의 (roll_Nm, pitch_Nm). 4-튜플 버스트용. 없으면 (0,0)."""
        for b in self.attack_bursts:
            if len(b) >= 4 and b[0] <= step < b[1]:
                return float(b[2]), float(b[3])
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════
    #  Flight Patterns
    # ══════════════════════════════════════════════════════════
    def _compute_setpoint(self):
        """궤도 setpoint 및 속도(Feedforward) 계산 + theta 전진."""
        alt = -abs(self.cfg.flight_altitude)
        # ★2026-09-08 고도 변조(ALT_MOD_AMP>0): 주행 중 sine 상하 → 수직 추력 변동 = vel 채널 aliasing.
        #   전 패턴 상속(alt 변수). PX4 위치제어가 수직속도 자연생성. 진폭 0.5=고도 2.0~3.0m(지면여유 2.0m).
        _am = getattr(self, '_alt_mod_amp', 0.0)
        if _am > 0.0:
            _tr = getattr(self, '_sim_flight_t', self.tick_count * self.step_dt)
            alt += _am * np.sin(2 * np.pi * getattr(self, '_alt_mod_freq', 0.3) * _tr)
        dt = getattr(self, '_dt_sim_last', self.step_dt) or self.step_dt   # sim-dt (0이면 step_dt — 프로브 안전)
        R = self.cfg.flight_radius; w = self.cfg.flight_omega
        pattern = self.scenario['pattern'] if self.scenario else 'hover'
        # ── 속도변조: 워프시간으로 같은 도형을 가감속. _amp=0 이면 현재와 동일 ──
        _amp = getattr(self, '_speed_mod_amp', 0.0)
        if _amp > 0.0:
            # ★★ 2026-08-26 편향 수정: _mod 를 **실시간**으로 계산한다.
            #   구 구현은 워프시간(self._traj_t)으로 계산했는데, 워프시간은 느린 구간에서
            #   천천히 흐르므로(dt_warp = _mod·dt_real) 실시간 기준 평균배율이 **조화평균**이 된다:
            #     amp 0.3→0.95 / 0.5→0.87 / 0.7→0.71 / 0.9→0.48
            #   = 가감속을 세게 걸수록 평균 속도가 오히려 떨어지는 역설. 실측 |v_xy| 중앙이
            #     설계 1.40 대비 0.27~0.57 로 낮았던 주원인.
            #   실시간 기준이면 sin 의 시간평균이 0 이라 평균배율이 정확히 1.00 이 된다.
            #   (궤적 위상은 여전히 워프시간 누적 → 같은 도형을 가감속하며 주행하는 동작은 유지)
            _treal = getattr(self, '_sim_flight_t', self.tick_count * self.step_dt)   # ★sim 시간축(2026-08-27)
            _mod = max(0.15, 1.0 + _amp*np.sin(2*np.pi*getattr(self, '_speed_mod_freq', 0.5)*_treal))
            t = self._traj_t; _tk = int(round(self._traj_t/dt)); self._traj_t += _mod*dt
        else:
            _mod = 1.0
            t = getattr(self, '_sim_flight_t', self.tick_count * self.step_dt)
            _tk = int(round(t / self.step_dt))

        if pattern == 'hover':
            return (0.0, 0.0, alt, 0.0, 0.0, 0.0, 0.0)

        elif pattern == 'circle':
            th = w * t                            # 워프시간 기반 (등속이면 self.theta 누적과 동일)
            x = R*np.cos(th) - R                  # 중심 (-R,0): th=0에서 원점 통과 (시작 갭 제거)
            y = R*np.sin(th)
            vx = -R*w*np.sin(th)*_mod; vy = R*w*np.cos(th)*_mod
            # yaw wrap (2026-07-29): theta 는 누적각이라 wrap 없이 쓰면 30s(1500 tick)에
            #   yaw=+16.56 rad(+948.9°, 2.64바퀴)까지 커진다. 타 패턴은 arctan2(figure8/waypoint)
            #   또는 [0,pi](aggressive)로 이미 [-pi,pi] 안이라 circle 만 규약이 달랐다.
            #   PX4 가 내부 wrap 하면 수학적으로는 동일하나 미검증 → 여기서 맞춰 보낸다.
            yaw = (th + np.pi/2 + np.pi) % (2*np.pi) - np.pi
            self.theta = th
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'line':
            # ★2026-08-28 직선 왕복 (데모용): +N 로 등속 직진 후 반전. yaw=0 고정(일정 헤딩) →
            #   공격(수직)이 한 월드방향으로 깨끗이 밀어 직선 hijack 이 보인다.
            L = getattr(self.cfg, 'line_half_len', 6.0); Tp = getattr(self.cfg, 'line_period_s', 44.0)  # ★긴 leg(22s)>공격창 → 한방향
            ph = (t / Tp) % 1.0; v0 = 4.0 * L / Tp * _mod
            if ph < 0.5:
                x = -L + 4.0*L*ph; vx = v0
            else:
                x = L - 4.0*L*(ph-0.5); vx = -v0
            y = 0.0; vy = 0.0; yaw = 0.0
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'figure8':
            # ★ 2026-08-07: |v|max = R*w*sqrt(2) 라 그대로 쓰면 실기 제한(2.0)을 넘는다.
            #   반경만 1/sqrt(2) 배해 최대속도를 circle 과 같은 R*w 로 맞춘다.
            R = R * getattr(self.cfg, 'fig8_radius_scale', 0.70710678)
            x = R*np.sin(w*t); y = (R/2)*np.sin(2*w*t)
            vx = R*w*np.cos(w*t)*_mod; vy = R*w*np.cos(2*w*t)*_mod
            yaw = np.arctan2(vy, vx) if (vx!=0 or vy!=0) else 0.0
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'waypoint':
            # ★ 2026-08-13 수정: 구 구현은 **레그당 4.0s 고정**이라 레그 길이가 다르면
            #   명령속도가 레그마다 달랐다 — 10 m 레그가 2.50 m/s 로 상한을 넘겨 클램프됐다
            #   (5/5/10/5/5 m → 1.25/1.25/**2.50**/1.25/1.25).
            #   실기 f5_pattern.py 는 처음부터 **등속**(T_seg = L/wp_speed)이었으므로
            #   sim 만 어긋나 있었다. 실기 쪽에 맞춘다 — 박스도 R/5 로 같이 스케일.
            # ★2026-08-27 박스 크기를 R 에서 분리 (구: kk=R/5 → R 축소가 박스를 물리한계 밑으로 줄임)
            kk = getattr(self.cfg, 'wp_box_halfwidth', 3.2) / 5.0
            wps = np.array([[0.,0.],[5.,0.],[5.,5.],[-5.,5.],[-5.,0.],[0.,0.]]) * kk
            seg_len = np.linalg.norm(np.diff(wps, axis=0), axis=1)
            if getattr(self.cfg, 'wp_corner_decel', False):
                # ★2026-08-27 기하 연동 가감속 (사용자 확정): 코너 감속 + 직선 가속 사다리꼴.
                #   호길이 상태 self._wp_s 를 로컬 속도로 전진 — 사인 SPEED_MOD 는 waypoint 에서 대체.
                vc = float(getattr(self.cfg, 'wp_v_corner', 0.5))
                vcr = float(getattr(self.cfg, 'wp_v_cruise', 1.7))
                ap = float(getattr(self.cfg, 'wp_a_prof', 2.0))
                L_tot = float(seg_len.sum())
                sarc = getattr(self, '_wp_s', 0.0) % L_tot
                # 현재 레그
                idx, acc0 = 0, 0.0
                for idx in range(len(seg_len)):
                    if sarc < acc0 + seg_len[idx]:
                        break
                    acc0 += seg_len[idx]
                d_in = sarc - acc0                    # 레그 시작(코너)부터 거리
                d_out = seg_len[idx] - d_in           # 다음 코너까지 거리
                v_here = min(vcr,
                             np.sqrt(vc*vc + 2.0*ap*max(d_in, 0.0)),
                             np.sqrt(vc*vc + 2.0*ap*max(d_out, 0.0)))
                self._wp_s = getattr(self, '_wp_s', 0.0) + v_here * dt
                f = d_in / seg_len[idx] if seg_len[idx] > 0 else 0.0
                dx = wps[idx+1][0]-wps[idx][0]; dy = wps[idx+1][1]-wps[idx][1]
                ux, uy = dx/seg_len[idx], dy/seg_len[idx]
                x = wps[idx][0] + dx*f; y = wps[idx][1] + dy*f
                vx = v_here*ux; vy = v_here*uy
                yaw = np.arctan2(dy, dx)
                return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)
            v_wp = R * w                                  # (구형 등속 경로 — wp_corner_decel=False)
            T_seg = seg_len / max(v_wp, 1e-6)
            tm = t % float(T_seg.sum())
            idx, acc = 0, 0.0
            for idx in range(len(T_seg)):
                if tm < acc + T_seg[idx]:
                    break
                acc += T_seg[idx]
            f = (tm - acc) / T_seg[idx] if T_seg[idx] > 0 else 0.0
            dx = wps[idx+1][0]-wps[idx][0]; dy = wps[idx+1][1]-wps[idx][1]
            x = wps[idx][0] + dx*f; y = wps[idx][1] + dy*f
            vx = (dx / T_seg[idx])*_mod; vy = (dy / T_seg[idx])*_mod
            yaw = np.arctan2(dy, dx)
            return (float(x), float(y), float(alt), float(yaw), float(vx), float(vy), 0.0)

        elif pattern == 'scurve':
            # ★ 2026-08-27 신설(비활성 — flight_patterns 에 넣어야 사용됨): S-커브 슬라럼.
            #   설계 의도: 실기 안전 봉투(v≤2.0·ACC_HOR 3.0) 안에서 "지속 회전"이 아니라
            #   **곡률 부호 반전(뱅크각 플립)의 밀도**로 토크 과도(du/dt)를 만든다.
            #   반원 4개를 S자로 체인(위/아래 교차) → 전진, 같은 경로 역주행으로 복귀(닫힘).
            #   접선속도 v=Rω(=1.20) 등속. 반전점마다 a_lat 이 +v²/Rs→−v²/Rs 로 플립
            #   (Rs=1.1, v=1.2 → a_lat 1.31, Δa 2.6 < ACC_HOR 3.0. 선회각속도 1.09 rad/s).
            #   ⚠ 실기 f5_pattern.py 에 같은 궤적 이식 후에만 활성화(sim=실기 원칙).
            Rs = getattr(self.cfg, 'scurve_radius', 1.1)
            Na = int(getattr(self.cfg, 'scurve_arcs', 4))          # 편도 반원 수
            v_s = R * w                                            # 접선속도 = 다른 패턴과 동일
            L1 = np.pi * Rs * Na                                   # 편도 호길이
            u_par = (v_s * t) % (2.0 * L1)
            fwd = u_par < L1
            s_arc = u_par if fwd else (2.0 * L1 - u_par)
            i = min(int(s_arc / (np.pi * Rs)), Na - 1)
            ph = s_arc / Rs - i * np.pi                            # 호 내 각 [0, π]
            Cx = (2 * i + 1) * Rs                                  # +x 체인으로 만들고 마지막에 x 미러
            if i % 2 == 0:   # 위쪽 반원 (CW): θ π→0, (2i·Rs,0) → ((2i+2)·Rs,0)
                th_a = np.pi - ph
                x = Cx + Rs * np.cos(th_a); y = Rs * np.sin(th_a)
                dxds = np.sin(th_a); dyds = -np.cos(th_a)
            else:            # 아래쪽 반원 (CCW): θ π→2π
                th_a = np.pi + ph
                x = Cx + Rs * np.cos(th_a); y = Rs * np.sin(th_a)
                dxds = -np.sin(th_a); dyds = np.cos(th_a)
            x = -x; dxds = -dxds                                   # −x 쪽으로 미러(다른 패턴과 공역 통일)
            sgn = 1.0 if fwd else -1.0
            vx = sgn * v_s * dxds * _mod; vy = sgn * v_s * dyds * _mod
            # ★ 3D 성분 (2026-08-27): 호마다 상승/하강 교대 — 곡률 반전점에서 뱅크 플립과
            #   수직속도 극값이 동시 발생 = 자세회전·고도·속도 3축 동시 과도(사용자 정의 급기동).
            #   z = −dz·sin(s/Rs), vz = −dz·(v/Rs)·cos(s/Rs) → vz 피크 dz·v/Rs (dz0.8→0.87 m/s < 2.25 ✓)
            dz_s = getattr(self.cfg, 'scurve_dz', 0.8)
            lam = s_arc / Rs
            z_off = -dz_s * np.sin(lam)
            vz_s = -sgn * dz_s * (v_s / Rs) * np.cos(lam) * _mod
            yaw = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0.0
            return (float(x), float(y), float(alt + z_off), float(yaw), float(vx), float(vy), float(vz_s))

        elif pattern == 'aggressive':
            # ★ 2026-08-07: 속도 FF 추가 (기존엔 이 패턴만 velocity=NaN 이었다).
            #   근거 — PX4 는 위치+속도FF 를 표준으로 설계했다:
            #     PositionControl.cpp:127-135  주어진 속도 SP 에 위치 P항을 더하고,
            #       포화 시 위치보정을 FF 보다 우선한다. 주석이 직접 "feed-forward".
            #     FlightTaskAuto.cpp:183       PX4 자체 임무모드가 스무딩 궤적 → 속도 FF.
            #   위치만 보내면 1차 지연 응답이라 명령 궤적을 그대로 못 그린다
            #   (원 궤적 기준 반경 −12%, 위상 28° 지연 @ MPC_XY_P=0.95, w=0.5).
            #   실기 f5_pattern.py 도 동일하게 FF 를 쓴다 — 양쪽이 같아야 비교가 성립.
            #  ★ 2026-08-07: 반경·고도진폭을 config 로 뺐다(구 하드코딩 4.0/3.0/2.0).
            #    **실기에서 안전하게 날 수 있는 값**을 기준으로 잡고 sim 이 따른다.
            Ra = getattr(self.cfg, 'agg_radius', 2.0)
            dz1 = getattr(self.cfg, 'agg_dz1', 1.0)
            dz2 = getattr(self.cfg, 'agg_dz2', 0.7)
            Tp = getattr(self.cfg, 'agg_phase_s', 5.0)
            # ★★ 2026-09-15 수정(사용자 확인): 위상을 궤적 시간 t 에서 직접 계산한다.
            #   구 구현 spp=int(Tp/dt)(dt=매 틱 측정 sim dt) 와 _tk=round(t/0.02)(고정 0.02 s) 의 단위가 섞여,
            #   dt 가 한 틱만 흔들려도 phase=(_tk//spp)%4 가 다른 칸으로 튀었다(cert_WIND 무공격 실측:
            #   에피당 설정점 점프 5–11.5회·최대 4.5 m, 추종오차 최대 4.4 m, 위상 길이 52스텝≠56).
            #   t 는 다른 패턴과 같은 궤적 시간(SPEED_MOD 켜면 워프시간)이라 위상 길이가 정확히 Tp 가 된다.
            T = Tp                          # 한 phase 길이 [s]
            phase = int(t // T) % 4
            f = (t % T) / T
            k = (np.pi/T)*_mod              # 속도변조: aggressive 속도 FF가 전부 k를 쓰므로 여기서 스케일
            # ★★ 2026-08-13 수정: phase 경계 **setpoint 불연속** 제거.
            #   구 구현은 ph0=(0,0) / ph1 은 (Ra,0)→(−Ra,0) / ph2=(**+**Ra,0) 이라
            #   경계마다 2.24 m·4.48 m 씩 순간이동을 명령했다(1스텝에 22~45 m/s 요구).
            #   기체가 최대속도로 쫓아가며 속도가 2.34 m/s 로 튀고 추종오차가 4.9 m 까지 뛰었다.
            #   → 반원을 **원점 기준**으로 옮겨(circle 이 x=Rcosθ−R 로 하는 것과 같은 방식)
            #     ph0(0,0) → ph1 (0,0)→(−2Ra,0) → ph2(−2Ra,0) → ph3 (−2Ra,0)→(0,0) 로 닫는다.
            #   반원 자체 속도는 그대로 Ra·π/T (기본 1.41 m/s) — 기동 강도는 안 줄었다.
            #   ⚠ 실기 f5_pattern.py 에도 **같은 수정**이 들어가야 궤적 비교가 성립한다.
            if phase == 0:      # 제자리 상승·하강 (반원 시작점 = 원점)
                return (0.0, 0.0, alt-dz1*np.sin(np.pi*f), 0.0,
                        0.0, 0.0, -dz1*k*np.cos(np.pi*f))
            elif phase == 1:    # 반원 전진 (a: 0→pi) — 원점에서 출발
                a = np.pi*f
                return (Ra*np.cos(a)-Ra, Ra*np.sin(a), alt, a,
                        -Ra*k*np.sin(a), Ra*k*np.cos(a), 0.0)
            elif phase == 2:    # 반대편(−2Ra, 0)에서 상승·하강
                return (-2.0*Ra, 0.0, alt+dz2*np.sin(np.pi*f), np.pi,
                        0.0, 0.0, dz2*k*np.cos(np.pi*f))
            else:               # 반원 복귀 (a: pi→0) — 원점으로 닫는다
                a = np.pi*(1-f)
                return (Ra*np.cos(a)-Ra, Ra*np.sin(a), alt, a,
                        Ra*k*np.sin(a), -Ra*k*np.cos(a), 0.0)

        return (0.0, 0.0, alt, 0.0, 0.0, 0.0, 0.0)

    # ══════════════════════════════════════════════════════════
    #  Episode State Reset
    # ══════════════════════════════════════════════════════════
    def _reset_episode_state(self):
        self.step_count = 0; self.tick_count = 0; self.stable_counter = 0; self.theta = 0.0
        self._traj_t = 0.0   # 속도변조 워프시간 리셋
        self._sim_flight_t = 0.0; self._prev_gt_sim_time = None   # ★sim 시간축 리셋(2026-08-27)
        self._wp_s = 0.0   # waypoint 호길이 상태 리셋 (코너감속 프로파일)
        self._hover_anchor = None   # 누수홀드 앵커 리셋
        self._last_traj_sp = None   # 재접근 동결점 리셋
        self._reengaging = False; self._nearest_cache = None; self._did_hover = False
        self._yaw_cmd_prev = None; self._vel_cmd_prev = None   # 궤적 후처리 상태 리셋(2026-08-13)
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.episode_losses = []; self.attack_active_flag = False
        self.window_buffer.clear(); self.gps_updated = False
        self.first_hover_step = None; self.hover_before_attack_count = 0
        self._hover_latched = False; self._ep_relapse = 0; self._ep_min_alt = 999.0; self._pprev_action = None
        self._ep_max_roll = 0.0; self._ep_max_pitch = 0.0   # 마감분석: 에피소드 최대 자세이탈
        self._is_airborne = False; self._hover_pos[:] = 0;  self._hover_yaw = 0.0; self._fs_switch(False)
        self._dwell_left = 0   # ★약속 hover 리셋
        self._hover_alt = -abs(self.cfg.flight_altitude)  # ★ 기본값(미전환 상태용)
        self.cur_pos[:] = 0; self.cur_vel[:] = 0; self.cur_euler[:] = 0
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib, q_gate=self._ukf_q_gate)
        # ★ POMDP: 에피별 COM 토크바이어스(roll/pitch, UKF가 예측에 쓰는 u에 가산=미모델링 COM오프셋)
        if _COM_BIAS_STD > 0:
            self._com_bias = np.array([random.uniform(-_COM_BIAS_STD, _COM_BIAS_STD),
                                       random.uniform(-_COM_BIAS_STD, _COM_BIAS_STD)])
        else:
            self._com_bias = np.zeros(2)
        self.is_ukf_initialized = False; self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)
        self.continuous_fp_count = 0
        self.drift_counter = 0; self._last_atk_step = None   # 공격 조건부 지오펜스(ATK_COND_DONE) 에피소드 리셋
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
        # ★FROZEN-ENV v2 (2026-09-02): 강풍 윈도우 — wind_window 있으면 무풍 시작,
        #   _rl_step_10hz 가 창 경계에서 토글. 약풍(윈도우 없음)은 즉시 ON(기존 동작).
        self._wind_win = self.scenario.get('wind_window')
        self._wind_win_on = False
        if self._wind_win:
            self._send_scenario_cmd(dist_override='none', ws_override=0.0)
        else:
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
        self._cur_burst_bias = (0.0, 0.0)
        self._last_burst_end = None

    def _check_done(self, trajectory_sp):
        dist = math.hypot(self.cur_pos[0]-trajectory_sp[0], self.cur_pos[1]-trajectory_sp[1])
        _maxerr = float(os.environ.get('DRIFT_MAX_ERR', '0') or 0) or self.cfg.max_error  # 실험용 완화
        # drift는 순간 스파이크가 아니라 지속 이탈일 때만 종료 (transient 보호)
        if dist >= _maxerr:
            self.drift_counter += 1
        else:
            self.drift_counter = 0
        if self.attack_active_flag: self._last_atk_step = self.step_count
        if self.drift_counter >= self.cfg.drift_patience:
            # ★2026-09-14 공격 조건부 지오펜스(env ATK_COND_DONE=K 스텝, 기본 0=무조건): 결과성 = "하이재킹 미대응의 결과" 이므로
            #   최근 K 스텝 안에 공격이 없었던 이탈(바람 단독, cert_WIND ws10 1/20 실측)은 종료하지 않는다 — 절단은 벌점 없어도
            #   남은 TN 보상이 사라져 결정과 무관한 암묵적 벌점이 되기 때문. surrogate 절벽(치명 에피 전용)과도 정합.
            _K = int(os.environ.get('ATK_COND_DONE', '0') or 0)
            if _K > 0 and (getattr(self, '_last_atk_step', None) is None or self.step_count - self._last_atk_step > _K):
                self._wind_drift_events = getattr(self, '_wind_drift_events', 0) + 1
                if self._wind_drift_events == 1 or self._wind_drift_events % 50 == 0:
                    self.get_logger().warn(f'  [GEOFENCE] 공격 없는 이탈 {self._wind_drift_events}회 — ATK_COND_DONE={_K} 로 종료 안 함 (step {self.step_count})')
                self.drift_counter = 0
                return False, ''
            return True, 'crash_drift'
        if self.cur_pos[2] > self.cfg.min_altitude: return True, 'crash_altitude'
        # ★2026-09-11 사용자 확정: 종료 = 지면 충돌(고도) ∨ 비행공간 이탈(drift) 뿐. 각도 규칙(60°)은 기본 OFF (FLIP_TERMINAL=1 로만 복원).
        if os.environ.get('FLIP_TERMINAL', '0') not in ('', '0') and (abs(self.cur_euler[0]) > 1.05 or abs(self.cur_euler[1]) > 1.05): return True, 'crash_flip'
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
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etc', 'analysis', 'plot_results.py')
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
        if reason == 'crash_flip' or (reason == 'crash_altitude' and (abs(self.cur_euler[0]) > 1.05 or abs(self.cur_euler[1]) > 1.05)):
            self._trigger_hard_reset()   # ★2026-09-11 각도 규칙 제거 후: 뒤집힌 채 지면에 닿은 추락은 WARM 재이륙 불가 → 바로 HARD (리셋 라우팅만, 종료 규칙 아님)
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
        self.eval_mode = bool(os.environ.get("FORCE_EVAL", ""))
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
        #   이번 틱 자세-수평 홀드 여부를 offboard 모드 발행 전에 확정(모드↔셋포인트 일치).
        self._att_hold_active = bool(self.flight_state == 'LEARNING'
                                     and getattr(self, 'prev_action', 0) == 1
                                     and os.environ.get('HOVER_ATT'))
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
                self._takeoff_fail_cnt = 0   # 이륙 성공 → 연속실패 카운터 리셋
                self.get_logger().info('  → STABILIZE')

            # ★2026-09-06 60→150s: HARD 리셋 직후 "첫 GT→즉시 TAKEOFF" 인데 PX4 SITL 이
            #   부팅→arm가능까지 ~56 sim초를 먹어, 이륙이 60s 타임아웃 직전에 시작되고
            #   타임아웃이 끊는 경합이 매 HARD 마다 재현 → 무한루프(v3_band 5h 소모 사고).
            if self.init_counter >= int(150.0 / self.step_dt):
                # ★2026-08-27 에스컬레이션: WARM 3연속 실패 → HARD(sim 재기동).
                #   08-25 밤샘에서 HARD 후 arm 이 영영 안 되는데 WARM 만 95회 반복하며
                #   1h39m 을 무학습으로 소모한 버그의 수정. (results_tune_rhukf_t02u4 ep152-246)
                self._takeoff_fail_cnt = getattr(self, '_takeoff_fail_cnt', 0) + 1
                if self._takeoff_fail_cnt >= 3:
                    self.get_logger().error(
                        f'  [TAKEOFF] Timeout 150s ×{self._takeoff_fail_cnt}연속 → HARD_RESET 에스컬레이션')
                    self._takeoff_fail_cnt = 0
                    self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
                    self.flight_state = 'HARD_RESET'
                else:
                    self.get_logger().warn(
                        f'  [TAKEOFF] Timeout 150s → WARM_RESET ({self._takeoff_fail_cnt}/3)')
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
                self.step_count = 0; self.tick_count = 0; self._traj_t = 0.0

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
                # ★2026-08-27 누수 홀드(leaky hold) — env HOVER_EMAX[m] 로 3모드:
                #     0      = 구 soft-hold (매 스텝 현재위치 = 복원력 0 → δ0.8 온셋분출 42m 활강)
                #     e>0    = 앵커가 현재위치에서 최대 e 까지만 뒤처짐 → 상시 e·P(≈e×0.95 m/s) 복원력,
                #              포화 회피(구 사수-flip 진단은 구플랜트·구공격 산물 — 재검 대상)
                #     매우 큼 = 사실상 좌표 사수
                #   고도·yaw latch 유지, 속도목표 0. (soft hold 2026-07-23 주석은 이력으로 대체)
                _emax = float(os.environ.get('HOVER_EMAX', '1e9') or 1e9)   # ★기본=강제 홀딩(사수). 07-23 flip 트레이드 부재 재검 완료(roll 47.6<60)
                if _emax > 0.0:
                    _anc = getattr(self, '_hover_anchor', None)
                    if _anc is None:
                        _anc = np.array([float(self.cur_pos[0]), float(self.cur_pos[1])])
                    _err = np.array([float(self.cur_pos[0]), float(self.cur_pos[1])]) - _anc
                    _d = float(np.linalg.norm(_err))
                    if _d > _emax:
                        _anc = _anc + _err * (1.0 - _emax / _d)   # 앵커가 e_max 밖으로는 끌려옴
                    self._hover_anchor = _anc
                    _hg = float(os.environ.get('HOVER_HARD_GAIN', '0') or 0)  # ★사수모드: 앵커 너머 가상타깃+속도FF
                    if _hg > 0.0:
                        _e = _anc - np.array([float(self.cur_pos[0]), float(self.cur_pos[1])])  # 앵커로의 오차
                        _tgt = _anc + _hg * _e                                    # 앵커 너머로 목표 밀기(공격적 P)
                        _vff = np.clip(2.0 * _e, -6.0, 6.0)                       # 앵커 향 속도 피드포워드
                        control_sp = (float(_tgt[0]), float(_tgt[1]),
                                      float(self._hover_alt), self._hover_yaw, float(_vff[0]), float(_vff[1]), 0.0)
                    else:
                        control_sp = (float(_anc[0]), float(_anc[1]),
                                      float(self._hover_alt), self._hover_yaw, 0.0, 0.0, 0.0)
                else:
                    control_sp = (float(self.cur_pos[0]), float(self.cur_pos[1]),
                                  float(self._hover_alt), self._hover_yaw, 0.0, 0.0, 0.0)
                # ★판정 분리(2026-08-27): drift 판정용 trajectory_sp 는 현재 위치 —
                #   호버 중 drift 판정 비활성(구 soft-hold 의 의미론 복원). 명령(control_sp)은 앵커.
                #   (이거 없으면 사수 홀드가 밀리는 과도 중에 crash_drift 로 잘림 — δ0.8 실측 +2.5s 절단)
                # ★2026-09-12 종료 규칙 대칭(사용자 확정): hover 중에도 앵커 기준 10 m·1 s 지오펜스 적용 (HOVER_DRIFT_SYM=0 이면 구 면제)
                if os.environ.get('HOVER_DRIFT_SYM', '1') not in ('', '0') and _emax > 0.0:
                    trajectory_sp = (float(self._hover_anchor[0]), float(self._hover_anchor[1]),
                                     float(self._hover_alt), self._hover_yaw, 0.0, 0.0, 0.0)
                else:
                    trajectory_sp = (float(self.cur_pos[0]), float(self.cur_pos[1]),
                                     float(self._hover_alt), self._hover_yaw, 0.0, 0.0, 0.0)
                self._did_hover = True   # ★재접근은 '호버 후 복귀'에서만 — 무대응 track 은 재접근 안함(hijack 노출)
            elif (not os.environ.get('NAIVE_TRACK')) and getattr(self, '_did_hover', False) and (getattr(self, '_reengaging', False) or (
                 getattr(self, '_last_traj_sp', None) is not None and
                 (lambda _n: math.hypot(self.cur_pos[0]-_n[0], self.cur_pos[1]-_n[1])
                     > float(os.environ.get('REENGAGE_R', '2.0') or 2.0))(self._nearest_xy()))):
                # ★2026-08-27 재접근 — "강제 홀딩 후 원궤적 복귀 기동". 목표 = 궤적 **최근접점**
                #   (동결점 아님 → 지나온 경로 역주행 방지). 도달(REENGAGE_R 안)하면 시계가 그
                #   phase 로 재동기된 채 정상 track 으로 넘어가 전진 재개. 판정은 유예(현재위치).
                _tgt = self._reengage_nearest()   # 최근접점 + 시계 재동기(부수효과)
                self._reengaging = True
                _tx, _ty = float(_tgt[0]), float(_tgt[1])
                _dx, _dy = _tx - float(self.cur_pos[0]), _ty - float(self.cur_pos[1])
                _dd = math.hypot(_dx, _dy)
                if _dd <= float(os.environ.get('REENGAGE_R', '2.0') or 2.0):
                    self._reengaging = False; self._did_hover = False   # 도달 → 순수 track 재개
                _vap = min(self.cfg.flight_radius * self.cfg.flight_omega * 1.4,
                           math.sqrt(max(2.0 * 2.0 * (_dd - 0.5), 0.01)))
                _ux, _uy = (_dx/_dd, _dy/_dd) if _dd > 1e-6 else (0.0, 0.0)
                control_sp = (_tx, _ty, float(_tgt[2]),
                              math.atan2(_dy, _dx), _vap*_ux, _vap*_uy, 0.0)
                # 판정 유예: 재접근 중 trajectory_sp=현재위치 (hover 와 동일 의미론).
                #   안전망: 원점 기준 60 m 초과는 진짜 통제상실로 간주(기존 drift 목적 유지).
                if math.hypot(self.cur_pos[0], self.cur_pos[1]) > 60.0:
                    trajectory_sp = control_sp   # → gt_err 커져 drift 판정 작동(안전망)
                else:
                    trajectory_sp = (float(self.cur_pos[0]), float(self.cur_pos[1]),
                                     float(self._last_traj_sp[2]), 0.0, 0.0, 0.0, 0.0)
            else:
                # ★2026-08-27 궤적 전진을 sim-dt 로 (wall 50Hz 타이머 × sim speed N 불일치 수정).
                #   구 구현: 매 틱 고정 0.02s 전진 → speed10·RTF10 이면 sim 기준 1/10 속도로
                #   위치목표가 기어가고 FF(1.2~1.8)와 싸워 사냥진동 발생 (실측: 명령 위치
                #   진행속도 중앙 0.00, 실속도 0.7 진동. 과거 캡처 전부 RTF 의존 오염).
                _gs = getattr(self, '_gt_sim_time', None)
                _gp = getattr(self, '_prev_gt_sim_time', None)
                if _gs is not None and _gp is not None and 0.0 < (_gs - _gp) <= 1.0:
                    self._dt_sim_last = _gs - _gp
                else:
                    self._dt_sim_last = self.step_dt   # 폴백(스탬프 이상·첫 틱)
                self._prev_gt_sim_time = _gs
                self._sim_flight_t = getattr(self, '_sim_flight_t', 0.0) + self._dt_sim_last
                trajectory_sp = self._compute_setpoint()
                control_sp = trajectory_sp
                self._last_traj_sp = trajectory_sp   # 재접근용 동결점 (hover/재접근 동안 유지)
                self.tick_count += 1

            if getattr(self, '_att_hold_active', False):
                self._send_attitude_level()   # ★자세-수평 홀드 (position 대신 attitude offboard)
            else:
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
        # ★2026-08-20 gyro 센서 σ 정합: PX4 SensorCombined(clean)에 실기 σ 주입.
        #   실기 실측 gyro σ ≈ [0.08,0.07,0.025] rad/s. GPS는 run_sim에서 처리(0.9×).
        #   SENSOR_NOISE_SCALE 로 clean(0)/real(1)/stress(2·3) 스윕. gyro 는 공격 채널이라 핵심.
        _gyro = self.cur_gyro
        if self._sensor_noise_scale > 0.0:
            _gyro = _gyro + np.random.normal(0.0, self._gyro_sigma) * self._sensor_noise_scale
        z_9d = np.concatenate([gps_ned, vel_ned, _gyro])
        u_phys = to_physical_u(np.array([self.cur_thrust]), np.array([self.cur_torque]), self.calib)[0]
        _was_init = not self.is_ukf_initialized
        if not self.is_ukf_initialized:
            self.ukf.x[0:3] = gps_ned; self.ukf.x[3:6] = self.cur_euler
            self.ukf.x[6:9] = vel_ned; self.ukf.x[9:12] = self.cur_gyro
            self.is_ukf_initialized = True
        # 멀티레이트(2026-08-05): predict+gyro=50Hz(루프), GPS update=10Hz(센서 실제 레이트).
        #   gps_updated 는 이 함수 직후 _rl_step_10hz/_sweep_step_10hz 게이트에서 소비되므로
        #   여기서 읽는 값이 곧 "이 스텝에 신선한 GPS 가 들어왔는가" 다.
        # ★ POMDP: UKF에만 COM 바이어스(u에 가산) + vel 센서노이즈(z에 가산). 플랜트는 불변(공격 GT 깨끗).
        if _COM_BIAS_STD > 0 or _VEL_NOISE > 0:
            u_ukf = u_phys.copy(); z_ukf = z_9d.copy()
            u_ukf[1] += self._com_bias[0]; u_ukf[2] += self._com_bias[1]
            if _VEL_NOISE > 0:
                z_ukf[3:6] += np.random.normal(0.0, _VEL_NOISE, 3)
        else:
            u_ukf, z_ukf = u_phys, z_9d
        self.last_res, self.last_Pzz = self.ukf.step(
            z_ukf, u_ukf, gps_fresh=(_was_init or self.gps_updated))
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
                    [_scale, _delay,
                     float(['circle','figure8','waypoint','aggressive'].index(self.scenario['pattern'])
                           if self.scenario and self.scenario.get('pattern') in ['circle','figure8','waypoint','aggressive'] else -1),
                     float(self.scenario.get('wind_speed', 0.0) if self.scenario else 0.0)],
                    [float(self.gt_pos[1]), float(self.gt_pos[0]), float(-self.gt_pos[2])]]))  # +pattern_idx +wind +GT_NED(N,E,U)
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
                # ── Step1 레짐진단: rhukf agent 계측 미러링 (adam엔 속성 없음 → getattr 0) ──
                self.last_kgain = float(getattr(self.agent, '_last_kgain', 0.0))
                self.last_pmax = float(getattr(self.agent, '_last_pmax', 0.0))
                self.last_innov = float(getattr(self.agent, '_last_innov', 0.0))
                self.last_argmax_flip = float(getattr(self.agent, '_last_argmax_flip', 0.0))
                self.last_qmax = float(getattr(self.agent, '_last_qmax', 0.0))
                self.last_nis = float(getattr(self.agent, '_last_nis', 0.0))
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

        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0, clip=_OBS_CLIP)  # OBS_NORM시 clip3.5
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0, clip=_OBS_CLIP)
        nis_p_raw, nis_p = compute_nis_scaled(self.last_res[0:3], self.last_Pzz[0:3, 0:3], 3.0)    # pos NIS(로깅 전용, 정책 입력 아님)
        nis_vel /= _OBS_DIV; nis_gyr /= _OBS_DIV   # OBS_NORM=1 → /3.5 정규화([0,1]), off면 무변화
        self._last_nis_raw = (nis_v_raw, nis_g_raw)   # 디버그 로깅용

        if self.step_count < cfg.learning_warmup_steps:
            self.step_count += 1
            return

        # ★FROZEN-ENV v2: 강풍 윈도우 토글 (학습 경로). wind_window=(s,e) → s에서 ON, e에서 OFF.
        _ww = getattr(self, '_wind_win', None)
        if _ww:
            if (not self._wind_win_on) and self.step_count >= _ww[0] and self.step_count < _ww[1]:
                self._send_scenario_cmd(); self._wind_win_on = True
                self.get_logger().info(f'  🌬 강풍 윈도우 ON @ step {self.step_count} (~{_ww[1]})')
            elif self._wind_win_on and self.step_count >= _ww[1]:
                self._send_scenario_cmd(dist_override='none', ws_override=0.0); self._wind_win_on = False
                self.get_logger().info(f'  🌬 강풍 윈도우 OFF @ step {self.step_count}')

        act_val = float(self.prev_action if self.prev_action is not None else 0.0)
        if getattr(cfg, '_gyro_only', False):   # GYRO_ONLY ablation: vel 채널 제거
            self.window_buffer.append([nis_gyr, act_val])
        else:
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
            relapse=bool(self.attack_active_flag and self.prev_action == 0
                         and getattr(self, '_pprev_action', None) == 1),   # ★2026-09-10 재발 벌점(RELAPSE_PEN)
        )
        # ★2026-09-07 terminal_penalty 삭제(사용자 확정): 절벽 = 보상이 아니라 에피 절단.
        #   CartPole 동형 — 추락하면 TN +0.5 흐름이 끊기는 것 자체가 절벽(γ0.85 가치 3.3 + 직전 FN 사슬).
        #   보상은 TP/TN/FP/FN 4개뿐. 절단 반영은 아래 terminated(PHYSICAL_TERMINALS)가 담당.
        # ── LL원리 보상 스케일링: 타깃 T_Var를 필터 R에 정합(loss 수렴 목적). push·로그 모두 스케일 단위 ──
        #   ※ F1/delay/crash 등 탐지지표는 reward 무관이라 실단위 유지. reward_sum만 스케일됨.
        reward *= getattr(cfg, 'reward_scale', 1.0)
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
        # ★2026-09-12 추락 벌점 A/B (env TERMINAL_PEN, 기본 0 = 절단만). 사용자 요청: SWIRL/Adam × {0, −5} 비교.
        _tp = float(getattr(cfg.reward, 'terminal_penalty', 0.0) or 0.0)
        if terminated and _tp:
            _tp *= getattr(cfg, 'reward_scale', 1.0); reward += _tp; self.episode_reward += _tp
        # ★2026-09-11 추락 벌점 없음(사용자 확정): 절벽 = terminated 부트스트랩 절단만. (잠시 넣었던 track-only −5 규칙 제거)

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
        # ★2026-09-13 약속 hover (env HOVER_DWELL=D, 기본 0=끔): hover 선언 후 D−1 스텝은 선택 없이 hover 유지(환경 동역학 = failsafe 래치).
        #   저장되는 transition 의 행동은 실행된 행동(강제 hover 포함). Hammar&Stadler 2022 다중정지의 '정지=지속되는 방어행동' 과 동일 구조.
        _dw = int(os.environ.get('HOVER_DWELL', '0') or 0)
        if _dw > 0:
            if getattr(self, '_dwell_left', 0) > 0:
                action = 1; self._dwell_left -= 1
            elif action == 1 and self.prev_action != 1:
                self._dwell_left = _dw - 1

        # ── HOVER 전환 시 위치 고정 (★ 떨림 방지) ──
        #   [latch 2026-07-09] 공격 대응 중에는 최초 진입 고도를 latch — 재진입 시 재캡처 금지.
        #   (재캡처=플리커마다 더 낮은 고도 재저장=계단식 침하→crash. 스크립트 latched dhover와 동일 동작)
        #   평시(공격무·에피소드시작)엔 정상 재캡처. latch는 공격 OFF에서 해제(아래 elif).
        if action == 1 and (self.prev_action != 1 or self.prev_action is None):
            # 위치·yaw는 항상 재캡처 (수평 드리프트 보정 정상)
            self._hover_pos[:] = self.cur_pos[:2]
            self._hover_yaw = float(self.cur_euler[2])
            # 고도만 latch: 공격 대응 중이면 재캡처 금지 (계단식 침하 방지)
            if self.attack_active_flag and self._hover_latched:
                pass   # _hover_alt 그대로 유지
            else:
                self._hover_alt = float(self.cur_pos[2])   # 최초 진입 고도에서 호버
                if self.attack_active_flag:
                    self._hover_latched = True             # 공격 중 최초 진입 → 이후 고정
            self._fs_switch(True)
        if action == 0 and self.prev_action == 1:
            self._fs_switch(False)

        # ── Detection tracking ──
        if action == 1:
            if self.first_hover_step is None:
                self.first_hover_step = self.step_count
            if not self.attack_active_flag:
                self.hover_before_attack_count += 1

        # ── [진단 2026-07-09] 공격중 relapse(hover→track 재발) + 최저고도 ──
        if self.attack_active_flag:
            if self.prev_action == 1 and action == 0:
                self._ep_relapse += 1
            self._ep_min_alt = min(self._ep_min_alt, -float(self.cur_pos[2]))

        # ── Attack burst on/off (버스트 경계에서 토글) ──
        want_attack = (self.scenario['attack_type'] != 'none') and self._is_attack_step(self.step_count)
        if want_attack and not self.attack_active_flag:
            sc = self.scenario
            if sc['attack_type'] == 'tilt':
                # ★ 2026-08-19 틸트 FDI: 이 버스트의 (roll_Nm, pitch_Nm)를 조회해 주입.
                _gx, _gz = self._current_burst_bias(self.step_count)
                _gt = 0.0; _int = 1.0
                self._cur_burst_bias = (_gx, _gz)
                self._send_attack_cmd(True, 'tilt', 1.0,
                    bias_torque_xy=_gx, bias_torque_z=_gz, bias_thrust_n=0.0)
            else:
                _int = sc.get('attack_intensity', 1.0)
                _gx = sc.get('bias_torque_xy', getattr(self.cfg, 'bias_torque_xy', 0.12))
                _gz = sc.get('bias_torque_z',  getattr(self.cfg, 'bias_torque_z', 0.0))
                _gt = sc.get('bias_thrust_n',  getattr(self.cfg, 'bias_thrust_n', 2.0))
                self._send_attack_cmd(True, sc['attack_type'], _int,
                    bias_torque_xy=sc.get('bias_torque_xy', None),
                    bias_torque_z=sc.get('bias_torque_z', None),
                    bias_thrust_n=sc.get('bias_thrust_n', None))
            self.attack_active_flag = True
            self._cur_burst_start = self.step_count
            self.get_logger().warn(
                f'  🚨 Attack ON (burst) @ step {self.step_count}: {self.scenario["attack_type"]} '
                f'| int={_int:.2f} → τx={_int*_gx:+.3f} τy/z={_int*_gz:+.3f} N·m, '
                f'thrust={_int*_gt:+.2f} N')
        elif want_attack and self.attack_active_flag and os.environ.get('HIJACK_TARGET'):
            # ★2026-08-27 B 유인공격(closed-loop): burst 시간구조는 A와 동일, 방향만 매 스텝
            #   해커 목표점 방향으로 갱신 재발행. HIJACK_TARGET="N,E" (NED 수평 목표) + HIJACK_DELTA.
            try:
                _tn, _te = [float(v) for v in os.environ['HIJACK_TARGET'].split(',')]
                _dn = _tn - float(self.gt_pos[1]); _de = _te - float(self.gt_pos[0])  # gt_pos=(E,N,U)?→NED 변환은 아래 일관
                _dd = math.hypot(_dn, _de)
                if _dd > 1e-3:
                    _dl = float(os.environ.get('HIJACK_DELTA', '0.5')) * self.cfg.attack_tq_authority_nm
                    _gx = _dl * (_dn/_dd); _gz = _dl * (_de/_dd)   # roll/pitch = 목표방향 성분
                    self._cur_burst_bias = (_gx, _gz)
                    self._send_attack_cmd(True, 'tilt', 1.0, bias_torque_xy=_gx, bias_torque_z=_gz, bias_thrust_n=0.0)
            except Exception:
                pass
        elif (not want_attack) and self.attack_active_flag:
            self._send_attack_cmd(False)
            self.attack_active_flag = False
            self._last_burst_end = self.step_count
            self._hover_latched = False   # [latch] 공격 OFF → 해제(다음 대응은 새 고도 latch)
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
                if _sc.get('attack_type') == 'tilt':
                    _rx, _py = self._cur_burst_bias
                    atk = f'🔴ATK(τx{_rx:+.2f} τy{_py:+.2f})'   # 틸트 토크[N·m]
                else:
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
                f'buf={buf} loss={cur_loss:.4f} Zvar={self.last_z_var:.3f} dt={self.last_learn_dt:.0f}ms | '
                f'Kg={self.last_kgain:.3f} Pmax={self.last_pmax:.2f} Qmax={self.last_qmax:.1f} innov={self.last_innov:.3f} flip={self.last_argmax_flip:.3f} NIS={self.last_nis:.2f}')

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
            'relapse': self._ep_relapse,   # [진단] 공격중 hover→track 재발 횟수
            'min_alt': round(self._ep_min_alt, 2) if self._ep_min_alt < 999 else -1,  # 공격중 최저고도(m)
            'bias_scale': round(float(self.scenario.get('bias_scale', 0.0)), 4) if self.scenario else 0.0,  # s별 delay 분석용
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
                          'atk_scale,atk_delay,pattern_idx,wind_speed')
            self.get_logger().info(f'[ZU] (z,u) 로그 저장 → {path}  (rows={len(arr)})')
        except Exception as e:
            self.get_logger().warn(f'[ZU] 저장 실패(무시): {e}')

    def _flush_sysid(self):
        """GT 기반 SysId 로그를 calibrate_sysld.py 포맷(npz)으로 저장. 매 에피소드 덮어씀."""
        if not self._log_sysid or not self._sysid_rows:
            return
        try:
            import os
            arr = np.asarray(self._sysid_rows, dtype=np.float64)
            path = os.path.join(self.cfg.outdir, 'sysid_log.npz')
            dts = np.diff(arr[:, 0])
            dt = float(np.median(dts[(dts > 0) & (dts < 0.1)])) if len(dts) > 10 else 0.004
            np.savez(path, dt=dt, t=arr[:, 0],
                     velocity=arr[:, 1:4], euler=arr[:, 4:7],
                     accelerometer=arr[:, 7:10], gyro=arr[:, 10:13],
                     thrust=arr[:, 13:16], torque=arr[:, 16:19],
                     episode=arr[:, 19], attack=arr[:, 20])
            self.get_logger().info(
                f'[SYSID] GT 로그 저장 → {path}  (rows={len(arr)}, dt={dt*1000:.2f}ms '
                f'= {1/dt:.0f}Hz)')
        except Exception as e:
            self.get_logger().warn(f'[SYSID] 저장 실패(무시): {e}')

    # ══════════════════════════════════════════════════════════
    #  Episode End
    # ══════════════════════════════════════════════════════════
    def _end_episode(self, reason):
        self._send_attack_cmd(False); self.attack_active_flag = False
        self._flush_zu()
        self._flush_sysid()

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
        cap = getattr(cfg, 'capture_mode', '') or ''
        if cap == 'deadline':
            # ── 마감 재측정: pattern × bias × delay_condition, disturbance=none ──
            #   패턴 고정·지연만 스윕(패턴/지연 교란 제거). delay=dhover{d}, no_response=track(대조군).
            cells = []; extra = []
            for pat in cfg.deadline_patterns:
                for b in cfg.deadline_biases:
                    for d in cfg.deadline_delays:
                        cells.append((float(b), f'dhover{int(d)}', pat)); extra.append(('none', 0.0))
                    cells.append((float(b), 'track', pat)); extra.append(('none', 0.0))   # no_response 대조군
            self.sweep_cell_extra = extra
            self.get_logger().info(
                f'\n{"="*60}\n  [CAPTURE:deadline] {len(cells)} cells '
                f'({len(cfg.deadline_patterns)}pat × {len(cfg.deadline_biases)}bias × '
                f'{len(cfg.deadline_delays)+1}delay[{",".join(map(str,cfg.deadline_delays))},track]) '
                f'× {cfg.sweep_episodes}ep | attack_start={cfg.sweep_attack_start} | disturbance=none\n{"="*60}')
        elif cap in ('normal', 'hijack'):
            # ── NIS 기준선 캡처 격자: (pattern × disturbance × wind × bias) ──
            biases = [0.0] if cap == 'normal' else list(cfg.capture_biases_hijack)
            cells = []; extra = []   # extra[i] = (disturbance_type, wind_speed)
            # track 셀: 패턴 비행 중 하이재킹 → crash 나면 나는 대로 기록(현실 조건)
            #   env CAPTURE_POLICIES="track,whover2" 지정 시 정책 확장(지정 시 자동 hover 셀 생략)
            _cap_pols = [x for x in os.environ.get('CAPTURE_POLICIES', 'track').split(',') if x]
            for pat in cfg.capture_patterns:
                for (dist, ws) in cfg.capture_disturbances:
                    for b in biases:
                        for _pol in _cap_pols:
                            cells.append((float(b), _pol, pat)); extra.append((dist, float(ws)))
            n_track = len(cells)
            if os.environ.get('CAPTURE_POLICIES'):
                cap = 'normal'   # 아래 자동 hover 셀 생략 (커스텀 정책 모드)
            # hover 셀(hijack만): 강제 원점호버 → 온셋 후 생존(basin)해 under-attack NIS 장기관측.
            #   pattern은 무관(action=1이면 _compute_setpoint 미호출·trajectory_sp=hover라 drift 오탐 없음) → dist×bias만 순회(중복 제거).
            if cap == 'hijack':
                for (dist, ws) in cfg.capture_disturbances:
                    for b in biases:
                        cells.append((float(b), 'hover', 'hover')); extra.append((dist, float(ws)))
            self.sweep_cell_extra = extra
            self.get_logger().info(
                f'\n{"="*60}\n  [CAPTURE:{cap}] {len(cells)} cells = {n_track} track'
                f'({len(cfg.capture_patterns)}pat×{len(cfg.capture_disturbances)}dist×{len(biases)}bias)'
                f' + {len(cells)-n_track} hover({len(cfg.capture_disturbances)}dist×{len(biases)}bias) '
                f'× {cfg.sweep_episodes}ep | attack_start={cfg.sweep_attack_start}\n{"="*60}')
        else:
            cells = [(0.0, 'track', cfg.sweep_pattern), (0.0, 'hover', 'hover')]
            for a in cfg.sweep_values:
                cells.append((float(a), 'track', cfg.sweep_pattern))
                cells.append((float(a), 'hover', 'hover'))
                # 조건 C: 추적 패턴으로 비행하다 공격 시작 +d 스텝에 호버 전환 (전이 케이스)
                for d in getattr(cfg, 'sweep_hover_delays', ()):
                    cells.append((float(a), f'dhover{int(d)}', cfg.sweep_pattern))
            self.sweep_cell_extra = None
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
            'nis_p_raw', 'nis_p_scaled',
            'gt_err', 'alt', 'action', 'crash_reason',
            'disturbance_type', 'wind_speed', 'roll', 'pitch', 'yaw_rate', 'u_norm',
            # 2026-08-13 추가: 한 주행 timestep plot 용 (속도·각속도 크기)
            'speed_xy', 'speed_z', 'omega_norm',
            # 2026-08-17 추가: raw innovation(잔차) 크기 — NIS(R로 정규화됨)와 분리해 보기 위함
            'res_v', 'res_g',
            # ★2026-09-08 추가: 3D 궤적 플롯용 절대 위치 (NED x/y). alt 는 이미 있음(-z).
            'pos_x', 'pos_y',
            # ★2026-09-10 추가: 궤적 기준점(setpoint) — reference vs 실제 추종 검증용 (ref_alt = -z)
            'ref_x', 'ref_y', 'ref_alt'])
        self._sweep_summary_f = open(os.path.join(cfg.outdir, 'sweep_summary.csv'), 'w', newline='')
        self._sweep_summary_w = csv.writer(self._sweep_summary_f)
        self._sweep_summary_w.writerow([
            'cell_idx', 'mode', 'bias', 'tq_xy', 'th_n', 'policy', 'pattern', 'episode',
            'survived', 'crash_step', 'crash_reason', 'steps', 'min_alt',
            'max_roll', 'max_pitch', 'disturbance_type', 'wind_speed'])
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
        # capture 모드면 셀별 외란, 아니면 sweep 전역 외란
        _extra = getattr(self, 'sweep_cell_extra', None)
        if _extra is not None:
            _dist, _ws = _extra[self.sweep_cell_idx]
        else:
            _dist = getattr(self.cfg, 'sweep_wind_type', 'none')
            _ws = float(getattr(self.cfg, 'sweep_wind_speed', 0.0))
        # ★ 2026-08-17 시간창: 공격/바람을 에피소드 내 창으로 (겹침구간 생성).
        #   SWEEP_ATK_START/END = 공격 창(스텝),  WIND_START/END = 바람 창(스텝).
        #   WIND_START≥0 이면 바람도 창(시작 clean → WIND_START on → WIND_END off).
        s = int(os.environ.get('SWEEP_ATK_START', s))
        self._atk_end = int(os.environ.get('SWEEP_ATK_END', '99999'))
        self._wind_start = int(os.environ.get('WIND_START', '-1'))
        self._wind_end = int(os.environ.get('WIND_END', '99999'))
        self.scenario = {
            'pattern': self.sweep_pattern_cur,
            'attack_type': os.environ.get('SWEEP_ATTACK_TYPE', 'loe_combined'),  # 2026-08-13 roll 전용 스윕용
            'attack_intensity': 1.0, 'attack_start_step': s,
            'attack_end_step': self._atk_end, 'attack_bursts': [(s, self._atk_end)],
            'disturbance_type': _dist,     # 로깅용 셀 배정값(창 무관 상수)
            'wind_speed': float(_ws),
        }
        _u = 'N' if self.cfg.sweep_attack_mode == 'thrust' else 'Nm'
        self.get_logger().info(
            f'\n  [SWEEP] cell {self.sweep_cell_idx+1}/{len(self.sweep_cells)} '
            f'{self.cfg.sweep_attack_mode} b={self.sweep_value:.3f}{_u} '
            f'(tq_xy={self._sweep_bias[0]:.3f} th_n={self._sweep_bias[2]:.3f}) '
            f'policy={self.sweep_policy} pat={self.sweep_pattern_cur} '
            f'| ep {self.sweep_ep_in_cell+1}/{self.cfg.sweep_episodes}')
        # 바람 창 모드면 시작은 clean(none) — WIND_START 에서 켠다. 아니면 셀 바람을 바로 켬(기존).
        if self._wind_start >= 0:
            self._send_scenario_cmd(dist_override='none', ws_override=0.0)
        else:
            self._send_scenario_cmd()
        self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
        # ★ 2026-08-14: 공격 지시함수 γ_k (FDI 비교용). peak ‖a‖ 동일, 타이밍만 다름.
        #   SWEEP_BURST 미설정        → 상수(단일 창 [s,∞)), γ_k=1 지속
        #   SWEEP_BURST="on,off"      → 고정 주기 펄스 (ON on / OFF off 스텝 반복)
        #   SWEEP_BURST="rand:a,b,c,d"→ 무작위 지시함수(표준 FDI): ON~U(a,b), OFF~U(c,d) 스텝
        _burst_env = os.environ.get('SWEEP_BURST', '').strip()
        _atk_hi = min(self._atk_end, self.cfg.episode_max_steps)   # 공격 창 상한
        # ★2026-09-07 SWEEP_ATK_WINDOWS="s1-e1,s2-e2,…" : 명시적 다중 공격창 (시나리오 타임라인 그림용).
        _win_env = os.environ.get('SWEEP_ATK_WINDOWS', '').strip()
        if _win_env:
            self.attack_bursts = [(int(w.split('-')[0]), int(w.split('-')[1])) for w in _win_env.split(',')]
        elif _burst_env.startswith('rand:'):
            _a, _b, _c, _d = (int(x) for x in _burst_env[5:].split(','))
            bursts = []; t = s
            while t < _atk_hi:
                _on = random.randint(_a, _b)
                bursts.append((t, min(t + _on, _atk_hi)))
                t += _on + random.randint(_c, _d)
            self.attack_bursts = bursts
        elif _burst_env:
            _on, _off = (int(x) for x in _burst_env.split(','))
            bursts = []; t = s
            while t < _atk_hi:
                bursts.append((t, min(t + _on, _atk_hi))); t += _on + _off
            self.attack_bursts = bursts
        else:
            self.attack_bursts = [(s, self._atk_end)]
        self._cur_burst_start = 0; self._last_burst_end = None
        # hover 정책이면 원점 고정 호버
        if self.sweep_policy == 'hover':
            self._hover_pos[:] = 0.0; self._hover_yaw = 0.0
            self._hover_alt = -abs(self.cfg.flight_altitude)  # hover 셀은 기준고도 유지
            self.prev_action = 1
        else:
            self.prev_action = 0

    def _fs_switch(self, on):
        """★2026-09-11 failsafe 파라미터 스위치 (env FAILSAFE_PARAMS 비면 no-op). hover 진입=engage, 복귀/리셋=release."""
        fs = getattr(self, '_fs', None)
        if fs is None:
            from env.failsafe_params import FailsafeParams
            fs = self._fs = FailsafeParams(log=lambda m: self.get_logger().info(m))
        if os.environ.get('FAILSAFE_ALWAYS', '') not in ('', '0'): on = True   # D 실험: 평시에도 failsafe 파라미터 상시 적용
        (fs.engage if on else fs.release)()

    def _sweep_step_10hz(self, trajectory_sp):
        """고정정책 1스텝: UKF NIS + 공격토글 + done + CSV. 학습/탐험/push 없음."""
        cfg = self.cfg
        nis_v_raw, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0)  # 통일압축 offset=1.0 (2026-07-22)
        nis_g_raw, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)  # 통일압축 offset=1.0
        nis_p_raw, nis_p = compute_nis_scaled(self.last_res[0:3], self.last_Pzz[0:3, 0:3], 3.0)    # pos NIS(로깅 전용, 정책 입력 아님)

        if self.step_count < cfg.learning_warmup_steps:
            self.step_count += 1
            return

        if self.sweep_policy == 'model':
            # ★2026-09-15 학습 정책 greedy 롤아웃(LOAD_MODEL): RL 루프(_rl_step_10hz)와 같은 관측 [vel, gyro, 이전행동]×W·클립/정규화·약속 hover(HOVER_DWELL)
            _nv = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0, clip=_OBS_CLIP)[1] / _OBS_DIV
            _ng = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0, clip=_OBS_CLIP)[1] / _OBS_DIV
            _pa = float(self.prev_action if self.prev_action is not None else 0.0)
            if self.step_count == cfg.learning_warmup_steps: self._dwell_left = 0
            self.window_buffer.append([_ng, _pa] if getattr(cfg, '_gyro_only', False) else [_nv, _ng, _pa])
            if len(self.window_buffer) < cfg.window_size:
                action = 0
            else:
                with self._learn_lock:
                    action = int(self.agent.act(np.array(self.window_buffer).flatten(), eps=0.0))
            _dw = int(os.environ.get('HOVER_DWELL', '0') or 0)
            if _dw > 0:
                if getattr(self, '_dwell_left', 0) > 0:
                    action = 1; self._dwell_left -= 1
                elif action == 1 and self.prev_action != 1:
                    self._dwell_left = _dw - 1
        elif self.sweep_policy == 'track':
            action = 0
        elif self.sweep_policy == 'hover':
            action = 1
        elif self.sweep_policy.startswith('whover'):
            # ★2026-08-27 'whover{d}': 공격창 호버 후 복귀 — onset+d 부터 공격창 종료+20스텝(2s)
            #   까지만 호버, 이후 track 재개. (dhover 는 래치형이라 복귀 안 함 — 궤적 플롯용 신설)
            d = int(self.sweep_policy[6:] or 2)
            _as = int(self.scenario.get('attack_start_step', self.cfg.sweep_attack_start)) if self.scenario else self.cfg.sweep_attack_start
            _lo = _as + d
            _hi = min(self._atk_end, self.cfg.episode_max_steps) + 20
            action = 1 if (_lo <= self.step_count < _hi) else 0
        else:  # 'dhover{d}': 공격 시작 +d 스텝부터 호버 (b=0 셀은 track과 동일 동작)
            d = int(self.sweep_policy[6:])
            # ★2026-09-07 버그수정: cfg.sweep_attack_start(기본 30)가 아니라 시나리오 실제 온셋 기준.
            #   v3_band 스윕에서 온셋 180인데 step 33부터 호버 = 사실상 '처음부터 호버'를 측정했다.
            _as = int(self.scenario.get('attack_start_step', self.cfg.sweep_attack_start)) if self.scenario else self.cfg.sweep_attack_start
            action = 1 if self.step_count >= _as + d else 0

        # 0→1 전환 순간 현재 위치/고도/요를 호버 셋포인트로 캡처 (본 RL 루프와 동일 semantics)
        if action == 1 and self.prev_action == 0:
            self._hover_pos[:] = self.cur_pos[:2]
            self._hover_alt = float(self.cur_pos[2])
            self._hover_yaw = float(self.cur_euler[2])
            self._hover_anchor = np.array(self.cur_pos[:2], dtype=float)   # 누수홀드 앵커 초기화
            self._fs_switch(True)
        elif action == 0 and self.prev_action == 1:
            self._fs_switch(False)

        done, term_reason = self._check_done(trajectory_sp)

        # ── 바람 시간창 토글 (WIND_START 에서 켜고 WIND_END 에서 끔) ──
        #   공격과 부분 겹침 구조: clean → 바람만 → 겹침 → 공격만 → 복귀.
        if getattr(self, '_wind_start', -1) >= 0:
            if self.step_count == self._wind_start:
                self._send_scenario_cmd()                          # 셀 바람 on
            elif self.step_count == self._wind_end:
                self._send_scenario_cmd(dist_override='none', ws_override=0.0)   # off

        # ── 공격 토글 (단일 윈도우; b=0이면 무해) ──
        want_attack = self._is_attack_step(self.step_count)
        if want_attack and not self.attack_active_flag:
            # 이번 셀의 물리 바이어스 벡터를 그대로 주입(intensity=1 → ramp로 0→full).
            #   ★ 2026-08-13: attack_type 은 scenario 에서(=SWEEP_ATTACK_TYPE env, 1463행).
            #   loe_roll → 순수 roll(tqx만), loe_combined → roll+pitch+yaw. 하드코딩 제거.
            tq_xy, tq_z, th_n = self._sweep_bias
            # ★2026-09-07 ramp 온셋 버그수정: 토글-온 최초 전송이 intensity=1.0 고정이라
            #   ramp>0 인데도 첫 스텝만 풀크기 킥이 나갔다(rampb2 trace 온셋 스파이크 실측).
            _int0 = 1.0
            if self.cfg.attack_ramp_duration > 0:
                from swrl_config import compute_attack_ramp as _car0
                _int0 = _car0(0.1, 1.0, self.cfg.attack_ramp_duration)
            self._send_attack_cmd(True, self.scenario.get('attack_type', 'loe_combined'), _int0,
                bias_torque_xy=tq_xy, bias_torque_z=tq_z, bias_thrust_n=th_n)
            self.attack_active_flag = True
            self._cur_burst_start = self.step_count
        elif want_attack and self.attack_active_flag and self.cfg.attack_ramp_duration > 0 \
                and (self.step_count - self._cur_burst_start) * 0.1 <= self.cfg.attack_ramp_duration + 0.15:
            # ★2026-09-07 ramp 물리 주입: allocator 통일(08-08) 후 물리 경로는 토글 시 1회 상수
            #   전송뿐이라 ramp 가 사실상 죽어 있었다(compute_attack_ramp 는 run_sim 라벨링 전용).
            #   ramp>0 이면 상승 구간 동안 매 스텝 intensity 를 재계산해 재전송한다.
            from swrl_config import compute_attack_ramp as _car
            _int = _car((self.step_count - self._cur_burst_start) * 0.1, 1.0, self.cfg.attack_ramp_duration)
            tq_xy, tq_z, th_n = self._sweep_bias
            self._send_attack_cmd(True, self.scenario.get('attack_type', 'loe_combined'), _int,
                bias_torque_xy=tq_xy, bias_torque_z=tq_z, bias_thrust_n=th_n)
        elif want_attack and self.attack_active_flag and os.environ.get('WORLD_ATK_DIR'):
            # ★2026-08-28 월드 고정방향 hijack: 매 스텝 yaw 보상해 body roll/pitch 재계산 →
            #   드론 자세와 무관하게 한 월드 방향으로 민다(깔끔한 단일방향 이탈 데모용).
            import math as _m
            _pw = _m.radians(float(os.environ['WORLD_ATK_DIR']))
            _yaw = float(self.cur_euler[2]); _mag = float(self._sweep_bias[0])
            _bd = _pw - _yaw
            _rt = _mag * _m.cos(_bd); _pt = _mag * _m.sin(_bd)
            self._send_attack_cmd(True, 'tilt', 1.0, bias_torque_xy=_rt, bias_torque_z=_pt, bias_thrust_n=0.0)
        elif (not want_attack) and self.attack_active_flag:
            self._send_attack_cmd(False)
            self.attack_active_flag = False

        # ── 측정 로깅 ──
        sp = np.array(trajectory_sp[:3])
        gt_ned = np.array([self.gt_pos[1], self.gt_pos[0], -self.gt_pos[2]])
        gt_err = float(np.linalg.norm(gt_ned[:2] - sp[:2]))
        alt = float(-self.cur_pos[2] if self.cur_pos[2] < 0 else 0.0)
        # [min_alt 2026-07-10] 공격 중 최저고도(침하 정도) — 스크립트 latch도 침하하나 확인용
        if self.attack_active_flag:
            self._ep_min_alt = min(self._ep_min_alt, alt)
        self._ep_max_roll = max(self._ep_max_roll, abs(float(self.cur_euler[0])))
        self._ep_max_pitch = max(self._ep_max_pitch, abs(float(self.cur_euler[1])))
        # capture용 추가 상태(관측 불변, 로깅만): roll/pitch/yaw_rate/u_norm/외란
        _dist = self.scenario.get('disturbance_type', 'none') if self.scenario else 'none'
        _ws = float(self.scenario.get('wind_speed', 0.0)) if self.scenario else 0.0
        try:
            _u = to_physical_u(np.array([self.cur_thrust]), np.array([self.cur_torque]), self.calib)[0]
            _u_norm = float(np.linalg.norm(_u))
        except Exception:
            _u_norm = float('nan')
        self._sweep_detail_w.writerow([
            self.sweep_cell_idx, self.cfg.sweep_attack_mode, f'{self.sweep_value:.3f}',
            f'{self._sweep_bias[0]:.3f}', f'{self._sweep_bias[2]:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell, self.step_count,
            int(self.attack_active_flag),
            f'{nis_v_raw:.5f}', f'{nis_g_raw:.5f}', f'{nis_vel:.5f}', f'{nis_gyr:.5f}',
            f'{nis_p_raw:.5f}', f'{nis_p:.5f}',
            f'{gt_err:.4f}', f'{alt:.4f}', action,
            term_reason if done else '',
            _dist, f'{_ws:.1f}',
            f'{self.cur_euler[0]:.5f}', f'{self.cur_euler[1]:.5f}', f'{self.cur_gyro[2]:.5f}', f'{_u_norm:.5f}',
            f'{float(np.hypot(self.cur_vel[0], self.cur_vel[1])):.4f}',
            f'{float(self.cur_vel[2]):.4f}',
            f'{float(np.linalg.norm(self.cur_gyro)):.5f}',
            # raw innovation 크기(정규화 전): vel=res[3:6], gyro=res[6:9]
            f'{float(np.linalg.norm(self.last_res[3:6])):.5f}',
            f'{float(np.linalg.norm(self.last_res[6:9])):.5f}',
            f'{float(self.cur_pos[0]):.4f}', f'{float(self.cur_pos[1]):.4f}',
            # ★2026-09-10 궤적 기준점 (reference vs 실제 추종 검증). ref_alt=-z
            *(lambda _sp: [f'{_sp[0]:.4f}', f'{_sp[1]:.4f}', f'{-_sp[2]:.4f}'])(
                getattr(self, '_last_traj_sp', None) or (float('nan'),) * 3)])

        if self.step_count % cfg.log_interval == 0:
            atk = '🔴ATK' if self.attack_active_flag else '⚪NRM'
            act = 'HOVER' if action == 1 else 'TRACK'
            self.get_logger().info(
                f'  [SWP {self.step_count:3d}] b={self.sweep_value:.3f} {atk} {act} | '
                f'NISraw v={nis_v_raw:.2f} g={nis_g_raw:.2f} | GT={gt_err:.2f}m alt={alt:.1f}m')

        self._pprev_action = self.prev_action; self.prev_action = action
        self.step_count += 1
        if done:
            self._end_episode(term_reason)

    def _end_sweep_episode(self, reason):
        survived = (reason == 'timeout')
        crash_step = -1 if survived else self.step_count
        _min_alt = round(self._ep_min_alt, 3) if self._ep_min_alt < 999 else -1.0
        _sd = self.scenario.get('disturbance_type', 'none') if self.scenario else 'none'
        _sw = float(self.scenario.get('wind_speed', 0.0)) if self.scenario else 0.0
        self._sweep_summary_w.writerow([
            self.sweep_cell_idx, self.cfg.sweep_attack_mode, f'{self.sweep_value:.3f}',
            f'{self._sweep_bias[0]:.3f}', f'{self._sweep_bias[2]:.3f}', self.sweep_policy,
            self.sweep_pattern_cur, self.sweep_ep_in_cell,
            int(survived), crash_step, reason, self.step_count, _min_alt,
            round(self._ep_max_roll, 4), round(self._ep_max_pitch, 4), _sd, f'{_sw:.1f}'])
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
    # ★ 2026-08-17 진단용: SPEED_SCALE 로 궤적 속도 배율(circle/fig8 = radius×omega). run_sim MPC 캡도 같은 env로 상향.
    # ★2026-09-11 TRAJ_SCALE: 궤적 속도만 배율 (PX4 캡은 실기값 2.0 그대로 — SPEED_SCALE 과 달리 제어기는 안 건드림).
    #   circle/figure8/scurve: ω×s · aggressive: Tp/s · waypoint: v_cruise·v_corner ×s (a_prof 유지)
    _ts = float(os.environ.get('TRAJ_SCALE', '1.0'))
    if _ts != 1.0:
        cfg.flight_omega *= _ts; cfg.agg_phase_s /= _ts
        cfg.wp_v_cruise *= _ts; cfg.wp_v_corner *= _ts
        print(f'[TRAJ_SCALE] ×{_ts}: ω {cfg.flight_omega:.3f} (접선 {cfg.flight_radius*cfg.flight_omega:.2f} m/s) · agg Tp {cfg.agg_phase_s:.2f} '
              f'(반원 {cfg.agg_radius*3.14159/cfg.agg_phase_s:.2f} m/s) · wp {cfg.wp_v_cruise:.2f}/{cfg.wp_v_corner:.2f} m/s')
    _ss = float(os.environ.get('SPEED_SCALE', '1.0'))
    if _ss != 1.0:
        cfg.flight_omega *= _ss
        cfg.agg_phase_s /= _ss   # ★2026-09-02 aggressive 도 동일 배율(반원속도 Ra·π/Tp ∝ 1/Tp, 선회각속도 ∝ ss)
        print(f'[SPEED_SCALE] flight_omega ×{_ss} → {cfg.flight_omega:.3f} (circle/fig8 속도 {cfg.flight_radius*cfg.flight_omega:.2f} m/s) '
              f'| agg_phase_s → {cfg.agg_phase_s:.2f} (반원속도 {cfg.agg_radius*3.14159/cfg.agg_phase_s:.2f} m/s)')

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
    ap.add_argument('--max-ep', dest='max_ep', type=int, default=None,
                    help='학습 총 에피소드 상한(max_episodes) override (미지정 시 config값=200)')
    ap.add_argument('--reward-scale', dest='reward_scale', type=float, default=None,
                    help='보상 스케일 c override (LL원리: 타깃 T_Var→필터R 정합). 미지정 시 config값')
    ap.add_argument('--gamma', dest='gamma', type=float, default=None,
                    help='할인율 gamma override (loss개형/부트스트랩 분산 튜닝)')
    ap.add_argument('--p-delta', dest='p_delta', type=float, default=None,
                    help='p_delta_init override (K_gain 조절: ↑p_Δ→↑K)')
    ap.add_argument('--r-init', dest='r_init_cli', type=float, default=None,
                    help='r_init override (K_gain 조절: ↑r→↓K)')
    ap.add_argument('--huber-c', dest='huber_c_cli', type=float, default=None,
                    help='huber_c override (잔차스케일에 맞춤; reward_scale 바뀌면 함께 조정)')
    ap.add_argument('--ft-ratio', dest='ft_ratio', type=float, default=None,
                    help='combined 모드 추력/토크비 sweep_combined_ft_ratio override (th_n=ft_ratio·b)')
    ap.add_argument('--hover-delays', dest='hover_delays', default=None,
                    help='쉼표구분 dhover 지연 스텝 목록 override (예: 1,2,3)')
    ap.add_argument('--log-zu', dest='log_zu', action='store_true',
                    help='UKF 오프라인 튜닝용 (z,u) 시계열을 outdir/zu_log.npz로 저장')
    ap.add_argument('--log-sysid', dest='log_sysid', action='store_true',
                    help='재캘리브레이션용 GT 시계열(속도/오일러/IMU/명령)을 outdir/sysid_log.npz로 저장')
    ap.add_argument('--capture-mode', dest='capture_mode', default=None,
                    choices=['normal', 'hijack', 'deadline'], help='캡처 격자 (normal|hijack|deadline)')
    ap.add_argument('--capture-disturbances', dest='capture_disturbances', default=None,
                    help='capture 외란 override (쉼표; 예: none:0,wind_turbulence:2,wind_turbulence:4)')
    ap.add_argument('--capture-biases', dest='capture_biases', default=None,
                    help='capture hijack bias override (쉼표; 예: 0.0,2.62)')
    ap.add_argument('--capture-patterns', dest='capture_patterns', default=None,
                    help='capture 패턴 override (쉼표; 예: hover,aggressive)')
    ap.add_argument('--sweep-attack-start', dest='sweep_attack_start', type=int, default=None,
                    help='sweep/capture 공격 ON 스텝 override (하이재킹은 100+ 권장)')
    ap.add_argument('--deadline-patterns', dest='deadline_patterns', default=None,
                    help='deadline 격자 패턴 override (쉼표구분; 예: hover,waypoint)')
    ap.add_argument('--deadline-biases', dest='deadline_biases', default=None,
                    help='deadline 격자 bias override (쉼표구분; 예: 1.37,1.40)')
    ap.add_argument('--deadline-delays', dest='deadline_delays', default=None,
                    help='deadline 격자 dhover 지연 override (쉼표구분; 예: 0,3,5). track 자동추가')
    ap.add_argument('--torque-yaw-ratio', dest='torque_yaw_ratio', type=float, default=None,
                    help='sweep torque 모드 yaw/roll 비 override. loe_yaw 스윕=1.0, 순수 tilt=0.0')
    ap.add_argument('--sweep-pattern', dest='sweep_pattern', default=None,
                    help='track/dhover 셀 비행패턴 override (aggressive|circle|figure8|waypoint)')
    ap.add_argument('--sweep-wind-type', dest='sweep_wind_type', default=None,
                    choices=['none', 'wind_constant', 'wind_gust', 'wind_turbulence'],
                    help='sweep 외란 타입 override (기본 none)')
    ap.add_argument('--sweep-wind-speed', dest='sweep_wind_speed', type=float, default=None,
                    help='sweep 바람 속도 m/s override (force≈0.031·v²N; 8≈2N 15≈7N)')
    ap.add_argument('--seed', type=int, default=None,
                    help='재현성 seed (random/numpy/torch 동시 시드) — baseline seed 반복용')
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
    if _args.max_ep is not None:
        cfg.max_episodes = int(_args.max_ep)
    if _args.reward_scale is not None:
        cfg.reward_scale = float(_args.reward_scale)
    if _args.gamma is not None:
        cfg.gamma = float(_args.gamma)
    if _args.p_delta is not None:
        cfg.p_delta_init = float(_args.p_delta)
    if _args.r_init_cli is not None:
        cfg.r_init = float(_args.r_init_cli); cfg.r_end = float(_args.r_init_cli)
    if _args.huber_c_cli is not None:
        cfg.huber_c = float(_args.huber_c_cli)
    if _args.ft_ratio is not None:
        cfg.sweep_combined_ft_ratio = float(_args.ft_ratio)
    if _args.hover_delays is not None:
        cfg.sweep_hover_delays = tuple(int(x) for x in _args.hover_delays.split(','))
    if getattr(_args, 'log_zu', False):
        cfg.log_zu = True
    if getattr(_args, 'log_sysid', False):
        cfg.log_sysid = True
    if _args.capture_mode is not None:
        cfg.capture_mode = _args.capture_mode
    if _args.sweep_attack_start is not None:
        cfg.sweep_attack_start = int(_args.sweep_attack_start)
    if _args.deadline_patterns is not None:
        cfg.deadline_patterns = [p.strip() for p in _args.deadline_patterns.split(',') if p.strip()]
    if _args.deadline_biases is not None:
        cfg.deadline_biases = [float(b) for b in _args.deadline_biases.split(',') if b.strip()]
    if _args.deadline_delays is not None:
        cfg.deadline_delays = tuple(int(d) for d in _args.deadline_delays.split(',') if d.strip())
    if getattr(_args, 'capture_disturbances', None) is not None:
        cfg.capture_disturbances = [(s.split(':')[0], float(s.split(':')[1]))
                                    for s in _args.capture_disturbances.split(',') if ':' in s]
    if getattr(_args, 'capture_biases', None) is not None:
        cfg.capture_biases_hijack = [float(b) for b in _args.capture_biases.split(',') if b.strip()]
    if getattr(_args, 'capture_patterns', None) is not None:
        cfg.capture_patterns = [p.strip() for p in _args.capture_patterns.split(',') if p.strip()]
    if getattr(_args, 'torque_yaw_ratio', None) is not None:
        cfg.sweep_torque_yaw_ratio = float(_args.torque_yaw_ratio)
    if _args.sweep_pattern is not None:
        cfg.sweep_pattern = _args.sweep_pattern
    if _args.sweep_wind_type is not None:
        cfg.sweep_wind_type = _args.sweep_wind_type
    if _args.sweep_wind_speed is not None:
        cfg.sweep_wind_speed = float(_args.sweep_wind_speed)
    if _args.outdir:
        cfg.outdir = _args.outdir
        os.makedirs(cfg.outdir, exist_ok=True)

    # ── 재현성 seed 적용 (baseline seed 반복 / RHUKF 공정비교). 미지정 시 cfg.seed(42) ──
    if _args.seed is not None:
        cfg.seed = int(_args.seed)
    import random as _random
    _random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    print(f"[SEED] {cfg.seed} 적용 (random/numpy/torch{'/cuda' if torch.cuda.is_available() else ''})")

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
