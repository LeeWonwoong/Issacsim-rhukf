#!/usr/bin/env python3
"""f13_policy.py — 실기·SITL 공용 **실시간 탐지 정책** 오프보드 노드 (2026-09-21).

  F5 패턴 비행(f5_pattern.F5Pattern) 위에 sim 학습 루프와 같은 관측·행동을 얹는다:
    shadow UKF(50 Hz, GPS 멀티레이트, Isaac 오버레이 UKF 노브 G2+ 동일) → 원시 NIS(vel, gyro)
    → env/observation 규약(log1p√·clip·÷div·4스텝 창·직전 행동) → numpy 정책(export_policy.py npz) → track/hover
    hover = sim failsafe 와 동일: 현재 위치·고도·요 캡처 → position 설정점 유지 + FailsafeParams(PX4 rate-loop 파라미터) engage,
            track 복귀 시 release + 패턴 설정점 재개(패턴 시각은 계속 흐른다 = sim 과 동일)
    공격(옵션 --attack dds): env/attack.sample_attack(v4 클래스) 프로파일을 /fmu/in/actuator_attack 으로 발행 (tilt: [δcosφ, δsinφ, 0])

  안전: 조종사 인계(오프보드 스위치 OFF → WAIT)·지오펜스는 offboard_common 그대로. --shadow 면 결정만 기록하고 hover 를 걸지 않는다.
        --attack-max 로 δ 상한(기본 0.4). 강공격(0.7+)은 현장에서 띄우지 않는다.

  사용
    SITL 리허설 :  python3 f13_policy.py --model models/swirl_v2_s42.npz --pattern circle --attack dds --fs-url udpin:0.0.0.0:14540
    실기(젯슨)  :  python3 f13_policy.py --model models/swirl_v4_s42.npz --pattern circle --attack dds --fs-url /dev/ttyACM0
    RC/외부 공격 :  ... --attack rc   (실기: ATK_EN=1·ATK_AUX_SW/TQ 로 조종기 노브가 주입 / SITL: 다른 창에서 python3 etc/analysis/rc_attack_trigger.py --stdin)
    shadow 만   :  ... --shadow
  로그: <outdir>/f13_policy_<pattern>_<stamp>.csv (10 Hz: NIS·관측·Q·행동·δ·위치)
"""
import argparse, csv, json, math, os, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import SensorCombined, SensorGps, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleLocalPosition, VehicleCommand, ManualControlSetpoint
try:
    from px4_msgs.msg import ActuatorAttack
except Exception:
    ActuatorAttack = None

import yaml
from env.knobs import set_knobs, knob
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration, to_physical_u
from env.observation import ObsSpec, ObsBuilder
from env.attack import AttackConfig, ProfileClass, sample_attack
from env.failsafe_params import FailsafeParams
from policy_np import NumpyPolicy
from nis_from_ulog import quat_to_euler_ned
from offboard_common import run, add_postflight_args, PUB_HZ
from f5_pattern import F5Pattern, build as build_pattern, main as _f5_main  # noqa: F401

R_EARTH = 6371000.0
UKF_DT = 0.02


class PolicyPattern(F5Pattern):
    SEQ_NAME = 'f13_policy'
    CHECK_TYPE = 'pattern'

    def __init__(self, bench=False, outdir='field_logs', *args, **kw):
        # ── 정책·관측·UKF 설정은 F5Pattern 생성(=Node 생성) 전에 준비 ──
        self.A = PolicyPattern.ARGS
        self.pol = NumpyPolicy(self.A.model)
        ob = self.pol.obs
        self.obs = ObsBuilder(ObsSpec(compress=ob['compress'], clip=float(ob['clip']), div=float(ob['div']), window=int(ob['window']), features=list(ob['features'])))
        # UKF 노브: Isaac 오버레이의 UKF_* 를 그대로 (빠지면 sim 과 다른 관측이 된다 — 09-21 실측)
        Y = yaml.safe_load(open(self.A.knobs)); K = ((Y.get('env') or {}).get('isaac') or {}).get('knobs') or {}
        kn = {k: v for k, v in K.items() if k.startswith('UKF_') or k in ('NIS_CLIP',)}
        if self.A.failsafe_params: kn['FAILSAFE_PARAMS'] = self.A.failsafe_params
        elif 'FAILSAFE_PARAMS' in K: kn['FAILSAFE_PARAMS'] = K['FAILSAFE_PARAMS']
        if self.A.fs_url: kn['FAILSAFE_MAV_URL'] = self.A.fs_url
        set_knobs(kn); self._knobs = kn
        if self.A.match_train:   # ★ 학습 루프(Isaac 오버레이)와 같은 주행: 속도변조 amp·가속 FF 를 노브에서 (기본 SPEED_MOD_AMP 0·ACCEL_FF 1)
            self.A.speed_mod = float(K.get('SPEED_MOD_AMP', 0.0) or 0.0); self.A.no_speed_mod = (self.A.speed_mod <= 0.0)
            self.A.no_ff = (str(K.get('ACCEL_FF', 1)) in ('0', ''))
        self.calib = load_calibration(self.A.calib) if self.A.calib else load_calibration()
        super().__init__(bench, outdir, *args, **kw)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)
        R = self._make_resolver(None)
        self.create_subscription(SensorCombined, R('/fmu/out/sensor_combined'), self._cb_imu, qos)
        self.create_subscription(SensorGps, R('/fmu/out/vehicle_gps_position'), self._cb_gps, qos)
        self.create_subscription(VehicleThrustSetpoint, R('/fmu/out/vehicle_thrust_setpoint'), self._cb_th, qos)
        self.create_subscription(VehicleTorqueSetpoint, R('/fmu/out/vehicle_torque_setpoint'), self._cb_tq, qos)
        self.pub_attack = self.create_publisher(ActuatorAttack, R('/fmu/in/actuator_attack'), qos) if (ActuatorAttack is not None and self.A.attack == 'dds') else None
        # --attack rc : 노드는 공격을 쏘지 않는다. RC aux(실기 ATK_AUX_*) 또는 외부 DDS 트리거(rc_attack_trigger.py --stdin)가 넣는 δ 를 **기록만** 한다.
        self._aux = (0.0, 0.0); self._ext_delta = 0.0
        if self.A.attack == 'rc':
            self.create_subscription(ManualControlSetpoint, R('/fmu/out/manual_control_setpoint'), self._cb_manual, qos)
            if ActuatorAttack is not None:
                self.create_subscription(ActuatorAttack, R('/fmu/in/actuator_attack'), self._cb_ext_attack, qos)

        # 최신 샘플
        self._gyro = None; self._gps = None; self._gps_seq = 0; self._gps_used = -1; self._th = None; self._tq = None
        self._ref = None                      # (lat, lon, alt) 로컬 NED 원점 — vehicle_local_position ref_*
        self._lock = threading.Lock()
        # UKF·정책 상태
        self.ukf = None; self.ukf_n = 0; self.fresh_n = 0
        self.prev_action = 0; self.action = 0; self.q = np.zeros(2); self.last_nis = (0.0, 0.0); self.last_obs = (0.0, 0.0)
        self.hovering = False; self.anchor = None; self.n_decl = 0; self.n_hover_steps = 0
        self.fs = FailsafeParams(log=lambda m: self.get_logger().info(m)) if not self.A.shadow else None
        # 공격 계획 (오프보드 진입 시 샘플)
        self.plan = None; self.atk_k = -1; self.atk_delta = 0.0; self.atk_dir = 0.0
        self._rng = np.random.default_rng(self.A.attack_seed)
        self.atk_cfg = self._attack_cfg() if self.A.attack == 'dds' else None
        # 정책 로그
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self._pcsv_path = os.path.join(outdir, f'f13_policy_{self.pattern}_{stamp}.csv')
        self._pcsv = open(self._pcsv_path, 'w', newline=''); self._pw = csv.writer(self._pcsv)
        self._pw.writerow(['t_wall', 't_seq', 'k', 'state', 'nis_v_raw', 'nis_g_raw', 'obs_v', 'obs_g', 'q_track', 'q_hover', 'action', 'hovering',
                           'atk_active', 'delta', 'dir', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'gps_hz', 'ukf_ms', 'sp_x', 'sp_y', 'sp_z', 'sp_vx', 'sp_vy', 'stage',
                           *[f'o{i}' for i in range(self.obs.spec.dim)]])   # ★09-22 정책 입력 벡터(창 4×[vel,gyro,prev_action], 오래된 프레임부터)
        self._ukf_ms = 0.0
        self.create_timer(UKF_DT, self._ukf_tick)
        self.get_logger().info(
            f"\n  ★ 정책 노드: 모델 {self.A.model} ({self.pol.meta.get('agent')} seed {self.pol.meta.get('seed')}, {self.pol.act_name}, θ {self.pol.theta.size})\n"
            f"    관측 {ob}\n    UKF 노브 {kn}\n    hover 실행 {'OFF (shadow: 결정만 기록)' if self.A.shadow else 'ON (position 홀드 + failsafe 파라미터)'}\n"
            f"    공격 {self.A.attack} (δ 상한 {self.A.attack_max}, seed {self.A.attack_seed})\n    정책 로그 {self._pcsv_path}\n")

    # ── 공격 설정: v4 무대와 같은 클래스·사건 수·간격 (cfgload 로 상속 체인 전체 병합) ──
    def _attack_cfg(self):
        """YAML extends 체인을 직접 병합해 AttackConfig 를 만든다 (cfgload 는 swrl_config→torch 를 끌어와 젯슨에서 못 쓴다)."""
        def _load(path):
            Y = yaml.safe_load(open(path)) or {}
            base = {}
            if 'extends' in Y:
                base = _load(os.path.join(os.path.dirname(path), Y['extends']))
            return _merge(base, Y)
        def _merge(a, b):
            out = dict(a)
            for k, v in b.items():
                out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
            return out
        at = dict((_load(self.A.attack_cfg).get('scenario') or {}).get('attack') or {})
        cls = [ProfileClass(name=c['name'], weight=float(c['weight']), delta=tuple(c['delta']), kind=c['kind'], dur=tuple(c['dur']),
                            grow=tuple(c.get('grow', (1.0, 1.0))), grow_steps=c.get('grow_steps', 10)) for c in at.pop('classes', [])]
        for k in ('gap', 'events', 'delta'):
            if k in at and isinstance(at[k], list): at[k] = tuple(at[k])
        cfg = AttackConfig(classes=cls, **at)
        if self.A.attack_p is not None: cfg.p_attack = float(self.A.attack_p)   # 현장은 매 비행 공격(기본 1.0)
        return cfg

    # ── 콜백 ──
    def _cb_lp(self, m):
        super()._cb_lp(m)
        if hasattr(m, 'ref_lat') and math.isfinite(m.ref_lat) and abs(m.ref_lat) > 1e-6:
            self._ref = (float(m.ref_lat), float(m.ref_lon), float(m.ref_alt))

    def _cb_imu(self, m):
        with self._lock: self._gyro = np.array(m.gyro_rad, dtype=float)

    def _cb_gps(self, m):
        lat = getattr(m, 'latitude_deg', None)
        if lat is None: lat, lon, alt = m.lat * 1e-7, m.lon * 1e-7, m.alt * 1e-3
        else: lon, alt = m.longitude_deg, m.altitude_msl_m
        with self._lock:
            self._gps = (float(lat), float(lon), float(alt), float(m.vel_n_m_s), float(m.vel_e_m_s), float(m.vel_d_m_s)); self._gps_seq += 1

    def _cb_manual(self, m):
        self._aux = (float(m.aux1), float(m.aux2))

    def _cb_ext_attack(self, m):
        self._ext_delta = float(math.hypot(m.torque[0], m.torque[1])) if m.active else 0.0

    def _cb_th(self, m):
        with self._lock: self._th = np.array(m.xyz, dtype=float)

    def _cb_tq(self, m):
        with self._lock: self._tq = np.array(m.xyz, dtype=float)

    # ── 50 Hz shadow UKF ──
    def _ukf_tick(self):
        with self._lock:
            g, gps, seq, th, tq, ref = self._gyro, self._gps, self._gps_seq, self._th, self._tq, self._ref
        if self.A.fake_gps:                      # ★ 벤치 전용(09-22): GPS 픽스/EKF 원점 없이도 돌게 — 정지 기체 = 위치·속도 0 (vel NIS 는 0 근방, gyro 만 유효)
            self._fake_seq = getattr(self, '_fake_seq', 0) + 1; seq = self._fake_seq
            gps = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); ref = (0.0, 0.0, 0.0)
        if g is None or gps is None or th is None or tq is None or ref is None:
            self.warn_once('ukf_wait', f'  UKF 입력 대기: gyro {g is not None} gps {gps is not None} thrust {th is not None} torque {tq is not None} ref {ref is not None}'
                           + ('  ← gps/ref 가 False 면 GPS 픽스·EKF 원점 없음(실외 지상 또는 --fake-gps 벤치), thrust/torque 가 False 면 arm 필요' ))
            self._wait_n = getattr(self, '_wait_n', 0) + 1
            if self._wait_n % 250 == 0: self.get_logger().warn(f'  UKF 입력 대기 계속({self._wait_n // 50}s): gyro {g is not None} gps {gps is not None} thrust {th is not None} torque {tq is not None} ref {ref is not None}')
            return
        t0 = time.time()
        lat, lon, alt, vn, ve, vd = gps
        px = math.radians(lat - ref[0]) * R_EARTH; py = math.radians(lon - ref[1]) * R_EARTH * math.cos(math.radians(ref[0])); pz = -(alt - ref[2])
        z = np.array([px, py, pz, vn, ve, vd, g[0], g[1], g[2]])
        u = to_physical_u(th.reshape(1, 3), tq.reshape(1, 3), self.calib)[0]
        now = time.time(); new_sample = seq != self._gps_used
        fresh = new_sample and (now - getattr(self, '_last_fresh_t', 0.0) >= (1.0 / float(self.A.gps_hz)) - 0.6 * UKF_DT)   # ★ 결정·GPS 갱신 10 Hz 게이트(학습과 동일). SITL GPS 100 Hz 대비
        if fresh: self._gps_used = seq; self._last_fresh_t = now
        if self.ukf is None:
            self.ukf = DynamicsUKF(dt=UKF_DT, calib=self.calib)
            self.ukf.x[0:3] = z[0:3]
            if self.att_q is not None: self.ukf.x[3:6] = quat_to_euler_ned(np.array([self.att_q], dtype=float))[0]
            self.ukf.x[6:9] = z[3:6]; self.ukf.x[9:12] = z[6:9]
            self.get_logger().info(f'  UKF 초기화: pos {z[0:3].round(2)} vel {z[3:6].round(2)}')
        res, Pzz = self.ukf.step(z, u, gps_fresh=bool(fresh)); self.ukf_n += 1
        if fresh:
            self.fresh_n += 1
            nv, _ = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3.0); ng, _ = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)
            self.last_nis = (float(nv), float(ng))
            st = self.obs.push(nv, ng, int(self.prev_action)); self.last_obs = self.obs.last_scaled
            self.last_st = st
            if st is not None:
                self.q = self.pol.q(st); a = int(np.argmax(self.q))
            else:
                a = 0
            # 정책 행동은 ENGAGED·패턴 구간에서만 유효. 그 밖(대기·settle·완료)은 track 고정.
            self.action = a if (self.state == 'ENGAGED' and self._in_pattern()) else 0
            self.prev_action = self.action
            self._attack_step()
            self._policy_log()
        self._ukf_ms = (time.time() - t0) * 1000.0

    def _in_pattern(self):
        if self.t_engage is None: return False
        te = time.time() - self.t_engage - self.settle
        return 0.0 <= te < self.T_run

    # ── 공격 발행 (10 Hz, GPS fresh 스텝) ──
    def _attack_step(self):
        if self.A.attack == 'rc':
            a1, a2 = self._aux
            if a1 > float(self.A.atk_sw_thr):   # 게이트(VRA=aux1) on
                d_rc = float(self.A.atk_tq_max) if int(self.A.atk_aux_tq) == 0 else float(self.A.atk_tq_max) * min(max(a2, 0.0), 1.0)   # ATK_AUX_TQ 0 = 고정 크기(08-10 실기), 2 = VRB 노브 비율
            else:
                d_rc = 0.0
            self.atk_delta = max(d_rc, self._ext_delta); return
        if self.pub_attack is None: return
        active = False; d = 0.0
        if self.state == 'ENGAGED' and self._in_pattern() and self.plan is not None:
            self.atk_k += 1; k = min(self.atk_k, self.plan.n - 1)
            active = bool(self.plan.active[k]); d = min(float(self.plan.delta[k]), float(self.A.attack_max)) if active else 0.0
        elif self.plan is None and self.state == 'ENGAGED' and self._in_pattern():
            n = int(self.T_run * 10) + 10
            if self.A.attack_fixed:
                d0, on_s, dur = [float(v) for v in self.A.attack_fixed.split(',')]; k0 = int(on_s * 10); dur = int(dur)
                from env.attack import AttackPlan
                plan = AttackPlan(n=n, active=np.zeros(n, bool), delta=np.zeros(n), bstart=np.full(n, -1), cls='fixed_persist', direction=0.0, events=[])
                prof = [d0 * f for f in self.atk_cfg.prefix] + [d0] * max(0, dur - len(self.atk_cfg.prefix))
                e = min(n, k0 + len(prof)); plan.delta[k0:e] = prof[:e - k0]; plan.active[k0:e] = True
                plan.events.append(dict(start=k0, end=e, cls='fixed_persist', d0=d0, g=1.0, grow_steps=0)); plan.direction = float(self.A.attack_dir_deg) * math.pi / 180.0
                self.plan = plan
            else:
                self.plan = sample_attack(self._rng, self.atk_cfg, n, with_direction=True)
            self.atk_k = -1
            self.atk_dir = float(getattr(self.plan, 'direction', 0.0) or 0.0)
            self.get_logger().info(f"  공격 계획: 사건 {len(self.plan.events)} " + ' '.join(f"{e['cls']}@{e['start']}(d0 {e['d0']:.2f}, {e['end']-e['start']}스텝)" for e in self.plan.events) + f"  방향 {math.degrees(self.atk_dir):.0f}°")
        self.atk_delta = d
        aa = ActuatorAttack(); aa.timestamp = 0; aa.active = bool(active and d > 0)
        aa.torque = [float(d * math.cos(self.atk_dir)), float(d * math.sin(self.atk_dir)), 0.0]; aa.thrust = 0.0
        self.pub_attack.publish(aa)

    def _policy_log(self):
        lp = self.lp; t_seq = 0.0 if self.t_engage is None else time.time() - self.t_engage
        st = getattr(self, 'last_st', None)   # 정책 입력 12-D (창 미충전이면 빈칸)
        st_cols = [''] * self.obs.spec.dim if st is None else [f'{v:.4f}' for v in st]
        self._pw.writerow([f'{time.time():.3f}', f'{t_seq:.2f}', self.atk_k, self.state, f'{self.last_nis[0]:.4f}', f'{self.last_nis[1]:.4f}',
                           f'{self.last_obs[0]:.4f}', f'{self.last_obs[1]:.4f}', f'{self.q[0]:.3f}', f'{self.q[1]:.3f}', self.action, int(self.hovering),
                           int(self.atk_delta > 0), f'{self.atk_delta:.3f}', f'{self.atk_dir:.3f}',
                           *(f'{v:.3f}' for v in ((lp.x, lp.y, lp.z, lp.vx, lp.vy, lp.vz) if lp is not None else (0,) * 6)),
                           f'{self.fresh_n / max(1e-6, time.time() - self._t_start):.1f}', f'{self._ukf_ms:.1f}',
                           *(f'{v:.3f}' for v in getattr(self, '_last_sp', (float('nan'),) * 5)), self.stage, *st_cols])

    # ── 패턴 스텝: 정책 hover 삽입 ──
    def step(self, t):
        if getattr(self, '_engage_id', None) != self.t_engage:      # 새 ENGAGED 진입: 궤적·정책·공격 상태 리셋
            self._engage_id = self.t_engage
            if self.gen is not None: self.gen.reset()
            self.obs.reset(); self.prev_action = 0; self.action = 0; self.hovering = False; self._reengaging = False
            self.plan = None; self.atk_k = -1; self.atk_delta = 0.0; self._hover_k = 0; self._hover_until = -1
            self.get_logger().info('  진입 리셋: 궤적 위상·관측 창·공격 계획 초기화')
        te = t - self.settle
        in_pat = 0.0 <= te < self.T_run
        # ★ 실행 최소 유지(--min-hover-steps): 정책 원결정은 그대로 기록하고, 실행되는 hover 만 N 스텝 이상 유지한다.
        #   (sim 은 dwell 0 이라 10 Hz 로 hover↔track 이 깜빡일 수 있다 — 실기에서 failsafe 파라미터를 10 Hz 로 썼다 지웠다 하지 않기 위한 현장 규칙)
        if self.action == 1 and in_pat: self._hover_until = getattr(self, '_hover_k', 0) + int(self.A.min_hover_steps)
        self._hover_k = getattr(self, '_hover_k', 0) + 1
        want_hover = ((self.action == 1) or (self.hovering and self._hover_k <= getattr(self, '_hover_until', -1))) and in_pat and not self.A.shadow
        if want_hover:
            if not self.hovering:
                self.hovering = True; self.n_decl += 1
                self.anchor = (self.lp.x, self.lp.y, self.lp.z, self.heading())
                if self.fs is not None: self.fs.engage()
                self.get_logger().info(f'  ▶ HOVER (선언 #{self.n_decl}) t={te:.1f}s q={self.q.round(2)} nis g={self.last_nis[1]:.1f} δ={self.atk_delta:.2f}')
            if self.gen is not None: self.gen.step(1.0 / PUB_HZ)     # 패턴 시각은 계속 흐른다(sim 동일)
            self.n_hover_steps += 1
            x, y, z, yaw = self.anchor; self.send_position_velocity(x, y, z, yaw, 0.0, 0.0, 0.0)   # sim: 앵커 사수·속도목표 0
            self.set_stage(f'HOVER {te:.0f}/{self.T_run:.0f}s (#{self.n_decl})')
            return False
        if self.hovering:
            self.hovering = False
            if self.fs is not None: self.fs.release()
            self.get_logger().info(f'  ◀ TRACK 복귀 t={te:.1f}s (hover {self.n_hover_steps} 스텝 누적)')
            if self.gen is not None and in_pat:
                tgt, d = self._reengage_nearest()
                if d > float(self.A.reengage_r):
                    self._reengaging = True; self._reengage_tgt = tgt
                    self.get_logger().info(f'  ↪ 재접근: 최근접점까지 {d:.1f} m (> {self.A.reengage_r} m) — 시계 재동기')
        if getattr(self, '_reengaging', False) and in_pat:
            tx, ty, tz = self._reengage_tgt; dx, dy = tx - self.lp.x, ty - self.lp.y; dd = math.hypot(dx, dy)
            if dd <= float(self.A.reengage_r):
                self._reengaging = False; self.get_logger().info(f'  ↩ 재접근 완료 ({dd:.1f} m) → track 재개')
            else:
                v_ap = max(0.1, min(self.R * self.w * 1.4, math.sqrt(max(2.0 * 2.0 * (dd - 0.5), 0.01))))   # sim 과 동일
                ux, uy = dx / dd, dy / dd
                if self.gen is not None: self.gen.step(1.0 / PUB_HZ)
                self.send_position_velocity(tx, ty, tz, math.atan2(dy, dx), v_ap * ux, v_ap * uy, 0.0)
                self.set_stage(f'REENGAGE {dd:.1f}m')
                return False
        done = super().step(t)
        if done and float(self.A.sitl_auto or 0.0) > 0 and not getattr(self, '_auto_landed', False):
            self._auto_landed = True
            import signal
            self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0)
            self.get_logger().info('  [SITL-AUTO] 착륙 명령 → 12 s 뒤 자동 종료')
            self.create_timer(12.0, lambda: os.kill(os.getpid(), signal.SIGINT))
        if done and self.plan is not None and self.pub_attack is not None:
            aa = ActuatorAttack(); aa.timestamp = 0; aa.active = False; aa.torque = [0.0, 0.0, 0.0]; aa.thrust = 0.0; self.pub_attack.publish(aa)
        return done

    # ── 마지막 발행 설정점 보관(추종오차 분석용 로그) ──
    def send_position(self, x, y, z, yaw):
        self._last_sp = (float(x), float(y), float(z), float('nan'), float('nan'))
        return super().send_position(x, y, z, yaw)

    def send_position_velocity(self, x, y, z, yaw, vx, vy, vz):
        self._last_sp = (float(x), float(y), float(z), float(vx), float(vy))
        return super().send_position_velocity(x, y, z, yaw, vx, vy, vz)

    # ── hover 뒤 재접근: 궤적 최근접점 + 시계 재동기 (sim online_rl_main._reengage_nearest 와 동일 논리) ──
    def _reengage_nearest(self):
        g = self.gen; cur = (self.lp.x, self.lp.y); dt = 1.0 / PUB_HZ
        snap = (g.t_real, g.t_warp, g.wp_s); best = (1e18, None, 0.0)
        if self.pattern == 'waypoint':
            cands = [i * g.L_tot / 240.0 for i in range(240)]
        else:
            if self.pattern in ('circle', 'figure8'): period = 2.0 * math.pi / g.w
            elif self.pattern == 'scurve': period = 2.0 * math.pi * g.Rs * g.Na / (g.R * g.w)
            elif self.pattern == 'aggressive': period = 4.0 * g.Tp
            else: period = g.lap_time()
            cands = [i * period / 240.0 for i in range(240)]
        for c in cands:
            if self.pattern == 'waypoint': g.wp_s = c; sp = g._sp(0.0, 1.0, 0.0)
            else: sp = g._sp(c, 1.0, dt)
            x, y, *_ = self._to_ned(sp[0], sp[1], 0.0, 0.0, 0.0)
            d2 = (x - cur[0]) ** 2 + (y - cur[1]) ** 2
            if d2 < best[0]: best = (d2, (x, y, self.origin[2] + sp[2]), c)
        g.t_real, g.t_warp, g.wp_s = snap
        if self.pattern == 'waypoint': g.wp_s = best[2]
        else: g.t_real = best[2]; g.t_warp = best[2]
        return best[1], math.sqrt(best[0])

    # ── SITL 무인 리허설용 자동 arm·오프보드·이륙 (--sitl-auto Z). 실기에서는 쓰지 않는다 ──
    def _vehicle_cmd(self, command, p1, p2=0.0):
        m = VehicleCommand(); m.command = int(command); m.param1 = float(p1); m.param2 = float(p2)
        m.target_system = 1; m.target_component = 1; m.source_system = 1; m.source_component = 1; m.from_external = True; m.timestamp = 0
        self.pub_cmd.publish(m)

    def hold_here(self):
        Z = float(self.A.sitl_auto or 0.0)
        if Z > 0 and self.state == 'WAIT' and self.lp is not None and not getattr(self, '_auto_landed', False):
            if getattr(self, '_auto_xy', None) is None:
                self._auto_xy = (self.lp.x, self.lp.y, self.heading()); self._auto_t0 = time.time(); self._auto_last = 0.0
                self.get_logger().info(f'  [SITL-AUTO] 이륙 설정점 ({self._auto_xy[0]:+.2f},{self._auto_xy[1]:+.2f},{-Z:.1f}) 스트림 → 2 s 뒤 OFFBOARD+ARM')
            self.stream_armed = True
            x0, y0, yaw0 = self._auto_xy
            self.send_position(x0, y0, -Z, yaw0)
            if time.time() - self._auto_t0 > 2.0 and time.time() - self._auto_last > 1.0 and not (self.offboard and self.armed):
                self._auto_last = time.time()
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            return
        super().hold_here()

    def _preflight_ok(self):
        Z = float(self.A.sitl_auto or 0.0)
        if getattr(self, '_auto_landed', False): return False                 # SITL 자동: 착륙 뒤 재진입 금지
        if Z > 0 and self.lp is not None and self.lp.z > -(Z - 0.4):
            self.warn_once('auto_climb', f'  [SITL-AUTO] 고도 도달 대기 (z {self.lp.z:+.2f} → {-Z:.1f})')
            return False
        return super()._preflight_ok()

    def destroy_node(self):
        try:
            if self.hovering and self.fs is not None: self.fs.release()
            if self.pub_attack is not None:
                aa = ActuatorAttack(); aa.timestamp = 0; aa.active = False; aa.torque = [0.0, 0.0, 0.0]; aa.thrust = 0.0; self.pub_attack.publish(aa)
            self._pcsv.close(); print(f'정책 로그 저장: {self._pcsv_path}  (UKF 스텝 {self.ukf_n}, GPS fresh {self.fresh_n}, 선언 {self.n_decl}, hover {self.n_hover_steps} 스텝)')
        except Exception:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='export_policy.py 가 만든 npz')
    ap.add_argument('--knobs', default=os.path.join(HERE, '..', 'configs', 'overlays', 'isaac.yaml'), help='UKF 노브 출처(Isaac 오버레이)')
    ap.add_argument('--calib', default=None, help='calibration.json (기본 자동 탐색)')
    ap.add_argument('--shadow', action='store_true', help='정책 결정만 기록, hover 를 걸지 않음')
    ap.add_argument('--attack', choices=['none', 'dds', 'rc'], default='none', help='dds: 노드가 v4 프로파일 발행 / rc: RC aux·외부 트리거가 넣는 δ 를 기록만')
    ap.add_argument('--atk-tq-max', dest='atk_tq_max', type=float, default=0.4, help='rc 모드 δ 기록용 ATK_TQ_MAX (PX4 파라미터와 같게)')
    ap.add_argument('--atk-sw-thr', dest='atk_sw_thr', type=float, default=0.9, help='rc 모드 aux1 스위치 임계 (ATK_SW_THR)')
    ap.add_argument('--atk-aux-tq', dest='atk_aux_tq', type=int, default=0, help='rc 모드 크기 노브: 0 = 없음(스위치 on = ATK_TQ_MAX 고정, 08-10 실기 설정) / 2 = aux2(VRB) 비율 × ATK_TQ_MAX. PX4 ATK_AUX_TQ 와 같게')
    ap.add_argument('--attack-cfg', dest='attack_cfg', default=os.path.join(HERE, '..', 'configs', 'newenv_v4.yaml'))
    ap.add_argument('--attack-max', dest='attack_max', type=float, default=0.4, help='δ 상한 (현장 안전: 강공격 0.7+ 금지)')
    ap.add_argument('--attack-seed', dest='attack_seed', type=int, default=0)
    ap.add_argument('--attack-fixed', dest='attack_fixed', default=None, help='고정 1사건 "d0,onset_s,dur_steps" (예 0.5,20,40): prefix 램프 뒤 d0 유지. 현장·리허설 검증용')
    ap.add_argument('--attack-dir-deg', dest='attack_dir_deg', type=float, default=0.0, help='고정 사건 토크 방향 [deg] (0=roll)')
    ap.add_argument('--attack-p', dest='attack_p', type=float, default=1.0, help='공격 에피 확률 (sim 0.5, 현장 기본 1.0)')
    ap.add_argument('--match-train', dest='match_train', type=int, default=1, help='1: Isaac 오버레이 노브(SPEED_MOD_AMP·ACCEL_FF)로 주행 정합 (학습과 동일)')
    ap.add_argument('--reengage-r', dest='reengage_r', type=float, default=2.0, help='hover 뒤 재접근 반경 [m] (sim REENGAGE_R)')
    ap.add_argument('--min-hover-steps', dest='min_hover_steps', type=int, default=5, help='실행 hover 최소 유지 스텝(10 Hz). 정책 원결정 로그와 별개. 0=sim 과 동일(즉시 복귀 가능)')
    ap.add_argument('--gps-hz', dest='gps_hz', type=float, default=10.0, help='정책 결정·GPS 갱신 게이트 [Hz] (학습 10)')
    ap.add_argument('--sitl-auto', dest='sitl_auto', type=float, default=0.0, help='SITL 전용: 자동 arm·OFFBOARD·이륙 고도[m] (0=끔, 실기 금지)')
    ap.add_argument('--fs-url', dest='fs_url', default=None, help='FailsafeParams MAVLink URL (SITL udpin:0.0.0.0:14540 / 실기 /dev/ttyACM0)')
    ap.add_argument('--failsafe-params', dest='failsafe_params', default=None, help='비우면 Isaac 오버레이 FAILSAFE_PARAMS 그대로; "" 면 비활성')
    # F5 인자 재사용
    ap.add_argument('--pattern', default='circle', choices=['waypoint', 'circle', 'figure8', 'yaw_spin', 'accel_line', 'aggressive', 'scurve'])
    ap.add_argument('--speed-mod', dest='speed_mod', type=float, default=0.5); ap.add_argument('--no-speed-mod', dest='no_speed_mod', action='store_true')
    ap.add_argument('--no-ff', dest='no_ff', action='store_true'); ap.add_argument('--yaw-amp', dest='yaw_amp', type=float, default=90.0); ap.add_argument('--yaw-T', dest='yaw_T', type=float, default=8.0)
    ap.add_argument('--acc-v', dest='acc_v', type=float, default=1.75); ap.add_argument('--acc-T', dest='acc_T', type=float, default=10.0)
    ap.add_argument('--agg-R', dest='agg_R', type=float, default=2.24); ap.add_argument('--agg-phase', dest='agg_phase', type=float, default=5.6)
    ap.add_argument('--R', type=float, default=3.4); ap.add_argument('--omega', type=float, default=0.38); ap.add_argument('--laps', type=float, default=2.0)
    ap.add_argument('--settle', type=float, default=4.0); ap.add_argument('--wp-speed', dest='wp_speed', type=float, default=None)
    ap.add_argument('--need-alt', dest='need_alt', type=float, default=1.0); ap.add_argument('--max-radius', dest='max_radius', type=float, default=0.0)
    ap.add_argument('--bench', action='store_true'); ap.add_argument('--bench-thrust', dest='bench_thrust', type=float, default=0.10)
    ap.add_argument('--fake-gps', dest='fake_gps', action='store_true', help='벤치 전용: GPS 픽스/EKF 원점 없이 위치·속도 0 으로 UKF 구동(실내). 실비행 금지')
    ap.add_argument('--outdir', default='field_logs')
    add_postflight_args(ap)
    a = ap.parse_args()
    PolicyPattern.ARGS = a
    import f5_pattern as _f5
    _orig = _f5.F5Pattern

    def _build(bench, outdir):
        _f5.F5Pattern = PolicyPattern           # build() 가 이 클래스로 생성하도록
        try:
            node = build_pattern(bench, outdir, a)
        finally:
            _f5.F5Pattern = _orig
        return node
    run(_build, bench=a.bench, outdir=a.outdir, args=a)


if __name__ == '__main__':
    main()
