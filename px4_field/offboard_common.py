"""offboard_common.py — 실기 데이터 수집용 오프보드 노드 공통 골격 (2026-08-03)

설계 원칙
  1. 발행 루프는 상태와 무관하게 **항상 10Hz** 로 돈다.
     PX4 는 setpoint 스트림이 이미 오고 있어야 오프보드 진입을 수락하고,
     0.5초만 끊겨도 페일세이프로 빠진다.
  2. 진입 전 기본 setpoint = **현재 위치 유지**.
     하드코딩 좌표를 쓰면 스위치를 켜는 순간 기체가 그 좌표로 날아간다.
  3. 오프보드 진입 순간 origin(x,y,z) 과 yaw0 을 스냅샷 → 이후 전부 이 기준.
     비행장 방향·기체 놓은 방향과 무관해진다.
  4. nav_state 가 OFFBOARD 를 벗어나면 즉시 SAFE. 조종사 인계가 항상 최우선.
  5. 이륙/착륙은 **사람이** 한다. 스크립트는 떠 있는 기체를 잠깐 넘겨받을 뿐이다.

px4_msgs 필드 (설치본에서 확인, 2026-08-03)
  OffboardControlMode      position/velocity/acceleration/attitude/body_rate/
                           thrust_and_torque/direct_actuator (bool), timestamp
  TrajectorySetpoint       position[3], velocity[3], acceleration[3], jerk[3],
                           yaw, yawspeed, timestamp     ※ NaN = 제어 안 함
  VehicleAttitudeSetpoint  q_d[4], thrust_body[3], yaw_sp_move_rate, timestamp
  VehicleStatus            nav_state (NAVIGATION_STATE_OFFBOARD=14),
                           arming_state (ARMING_STATE_ARMED=2)
  VehicleLocalPosition     x,y,z,vx,vy,vz,heading, xy_valid/z_valid/v_xy_valid/v_z_valid
"""
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleAttitudeSetpoint,
    VehicleCommand, VehicleStatus, VehicleLocalPosition, VehicleAttitude,
)

NAN = float('nan')
PUB_HZ = 20.0          # 발행 주기 (PX4 요구 최소 2Hz, 여유있게)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_to_quat(yaw, roll=0.0, pitch=0.0):
    """ZYX 오일러 → 쿼터니언 [w,x,y,z] (NED/FRD)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy]


class OffboardSequenceNode(Node):
    """시퀀스 스크립트가 상속해서 on_engaged()/step() 만 구현하면 된다."""

    SEQ_NAME = 'base'
    NEED_ALT = 3.0             # 진입 최소 고도 [m] (bench 모드면 무시)
    MAX_RADIUS = 25.0          # origin 기준 수평 이탈 한계 [m]
    MAX_ALT_DEV = 15.0         # origin 기준 고도 이탈 한계 [m]

    BENCH_THRUST = 0.10        # 프로펠러 뺀 지상검증 시 추력. 모터가 저속으로만 돈다

    def __init__(self, bench=False, outdir='field_logs'):
        super().__init__(f'offb_{self.SEQ_NAME}')
        self.bench = bench
        self.bench_thrust = self.BENCH_THRUST
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_att = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._cb_status, qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self._cb_lp, qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._cb_att, qos)

        # 상태
        self.nav_state = -1
        self.arming = -1
        self.lp = None
        self.att_q = None
        self.state = 'WAIT'            # WAIT → ENGAGED → DONE
        self.origin = None             # (x, y, z)
        self.yaw0 = 0.0
        self.t_engage = None
        self.stage = 'idle'
        self._last_stage = None
        self._warned = set()

        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(outdir, f'{self.SEQ_NAME}_{stamp}.csv')
        self._csv = open(self._csv_path, 'w', newline='')
        self._w = csv.writer(self._csv)
        self._w.writerow(['t_wall', 't_seq', 'state', 'stage', 'nav_state', 'armed',
                          'x', 'y', 'z', 'vx', 'vy', 'vz', 'heading'])

        self.create_timer(1.0 / PUB_HZ, self._tick)
        self.get_logger().info(
            f"\n{'='*64}\n  {self.SEQ_NAME.upper()}  {'[BENCH 모드 — 사전조건 완화]' if bench else ''}\n"
            f"  setpoint 스트림 {PUB_HZ:.0f}Hz 발행 시작.\n"
            f"  → 수동으로 이륙시킨 뒤 **오프보드 스위치를 켜세요**.\n"
            f"  마커 로그: {self._csv_path}\n{'='*64}")

    # ── 콜백 ─────────────────────────────────────────────
    def _cb_status(self, m):
        prev = self.nav_state
        self.nav_state, self.arming = m.nav_state, m.arming_state
        if prev != self.nav_state:
            self.get_logger().info(f"  nav_state {prev} → {self.nav_state}"
                                   f"{'  (OFFBOARD)' if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD else ''}")

    def _cb_lp(self, m):
        self.lp = m

    def _cb_att(self, m):
        self.att_q = m.q

    # ── 유틸 ─────────────────────────────────────────────
    @property
    def offboard(self):
        return self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    @property
    def armed(self):
        return self.arming == VehicleStatus.ARMING_STATE_ARMED

    def heading(self):
        if self.lp is not None and math.isfinite(self.lp.heading):
            return float(self.lp.heading)
        return 0.0

    def warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(msg)

    def set_stage(self, name):
        if name != self._last_stage:
            t = 0.0 if self.t_engage is None else time.time() - self.t_engage
            self.get_logger().info(f"  [{t:6.1f}s] {name}")
            self._last_stage = name
        self.stage = name

    def body_dir(self, forward=True):
        """yaw0 기준 바디 전진(+x) / 좌우(+y) 방향의 NED 단위벡터."""
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return (c, s) if forward else (-s, c)

    # ── 발행 ─────────────────────────────────────────────
    def _ocm(self, position=False, velocity=False, attitude=False):
        m = OffboardControlMode()
        m.position, m.velocity, m.attitude = position, velocity, attitude
        m.acceleration = m.body_rate = m.thrust_and_torque = m.direct_actuator = False
        m.timestamp = 0
        self.pub_ocm.publish(m)

    def send_position(self, x, y, z, yaw):
        self._ocm(position=True)
        m = TrajectorySetpoint()
        m.position = [float(x), float(y), float(z)]
        m.velocity = [NAN, NAN, NAN]
        m.acceleration = [NAN, NAN, NAN]
        m.yaw = float(wrap_pi(yaw))
        m.yawspeed = NAN
        m.timestamp = 0
        self.pub_traj.publish(m)

    def send_velocity(self, vx, vy, vz, yaw):
        if self.bench:                       # 실내 = 위치/속도 추정 무효 → 자세로 대체
            self.send_attitude(0.0, 0.0, yaw, self.bench_thrust)
            return
        self._ocm(velocity=True)
        m = TrajectorySetpoint()
        m.position = [NAN, NAN, NAN]
        m.velocity = [float(vx), float(vy), float(vz)]
        m.acceleration = [NAN, NAN, NAN]
        m.yaw = float(wrap_pi(yaw))
        m.yawspeed = NAN
        m.timestamp = 0
        self.pub_traj.publish(m)

    def send_attitude(self, roll, pitch, yaw, thrust_norm):
        """thrust_norm 은 0~1 (아래로 미는 크기). FRD 라 thrust_body[2] = -thrust_norm."""
        self._ocm(attitude=True)
        m = VehicleAttitudeSetpoint()
        m.q_d = [float(v) for v in yaw_to_quat(wrap_pi(yaw), roll, pitch)]
        m.thrust_body = [0.0, 0.0, -float(thrust_norm)]
        m.yaw_sp_move_rate = 0.0
        m.timestamp = 0
        self.pub_att.publish(m)

    def hold_here(self):
        """현재 위치·방향 유지. 진입 전 기본 상태이자 SAFE 상태.

        ★ bench 모드는 **자세 setpoint** 로 대체한다.
          실내는 GPS 가 없어 위치 추정이 무효인데, position 타입 setpoint 를 흘리면
          PX4 가 오프보드 진입 자체를 거부한다. 자세 타입은 위치 추정이 필요 없다.
        """
        if self.bench:
            self.send_attitude(0.0, 0.0, self.heading(), self.bench_thrust)
            return
        if self.lp is None:
            self._ocm(position=True)
            return
        self.send_position(self.lp.x, self.lp.y, self.lp.z, self.heading())

    def hold_origin(self, dz=0.0):
        if self.bench:
            self.send_attitude(0.0, 0.0, self.yaw0, self.bench_thrust)
            return
        x, y, z = self.origin
        self.send_position(x, y, z + dz, self.yaw0)

    # ── 메인 루프 ────────────────────────────────────────
    def _tick(self):
        if self.lp is None:
            self.warn_once('nolp', '  vehicle_local_position 수신 대기 중...')
            self._ocm(position=True)
            return

        if self.state == 'WAIT':
            self.hold_here()
            if self.offboard:
                if not self._preflight_ok():
                    return
                self.origin = (self.lp.x, self.lp.y, self.lp.z)
                self.yaw0 = self.heading()
                self.t_engage = time.time()
                self.state = 'ENGAGED'
                self.get_logger().info(
                    f"\n  ★ OFFBOARD 진입 — 시퀀스 시작\n"
                    f"    origin = ({self.origin[0]:+.2f}, {self.origin[1]:+.2f}, {self.origin[2]:+.2f}) NED\n"
                    f"    yaw0   = {math.degrees(self.yaw0):+.1f}°  (이 방향 기준으로 궤적을 만듭니다)\n")

        elif self.state == 'ENGAGED':
            if not self.offboard:
                self.get_logger().warn('  오프보드 이탈 감지 → SAFE. 조종사가 인계했습니다.')
                self.state = 'WAIT'
                self._last_stage = None
                self.hold_here()
                self._log_row()
                return
            if not self._bounds_ok():
                self.set_stage('ABORT_BOUNDS')
                self.hold_origin()
                self._log_row()
                return
            t = time.time() - self.t_engage
            if self.step(t):
                self.state = 'DONE'
                self.get_logger().info(
                    f"\n  ✅ 시퀀스 완료 ({t:.1f}s). 위치 유지 중.\n"
                    f"     조종사: Position 으로 복귀 → 착륙 → disarm\n"
                    f"     그 다음 Ctrl-C 로 노드 종료\n")

        else:                                   # DONE
            self.hold_origin() if self.origin else self.hold_here()
            if not self.offboard:
                self.state = 'WAIT'
                self._last_stage = None

        self._log_row()

    def _preflight_ok(self):
        if self.bench:
            self.warn_once('bench', '  [BENCH] 고도·유효성 사전조건을 건너뜁니다')
            return True
        alt = -self.lp.z
        if not (self.lp.xy_valid and self.lp.z_valid):
            self.warn_once('valid', f'  ✗ 위치 추정 무효 (xy_valid={self.lp.xy_valid}, '
                                    f'z_valid={self.lp.z_valid}) — 시퀀스 시작 거부')
            return False
        if alt < self.NEED_ALT:
            self.warn_once('alt', f'  ✗ 고도 부족 ({alt:.1f}m < {self.NEED_ALT}m) — 더 올린 뒤 다시 켜세요')
            return False
        return True

    def _bounds_ok(self):
        if self.bench or self.origin is None:
            return True
        dx, dy = self.lp.x - self.origin[0], self.lp.y - self.origin[1]
        if math.hypot(dx, dy) > self.MAX_RADIUS:
            self.get_logger().error(f'  ✗ 수평 이탈 {math.hypot(dx,dy):.1f}m > {self.MAX_RADIUS}m — 중단')
            return False
        if abs(self.lp.z - self.origin[2]) > self.MAX_ALT_DEV:
            self.get_logger().error(f'  ✗ 고도 이탈 {abs(self.lp.z-self.origin[2]):.1f}m — 중단')
            return False
        return True

    def _log_row(self):
        t_seq = 0.0 if self.t_engage is None else time.time() - self.t_engage
        lp = self.lp
        self._w.writerow([f'{time.time():.3f}', f'{t_seq:.3f}', self.state, self.stage,
                          self.nav_state, int(self.armed),
                          f'{lp.x:.3f}', f'{lp.y:.3f}', f'{lp.z:.3f}',
                          f'{lp.vx:.3f}', f'{lp.vy:.3f}', f'{lp.vz:.3f}',
                          f'{self.heading():.4f}'])

    # ── 시퀀스가 구현할 것 ────────────────────────────────
    def step(self, t):
        """t = 진입 후 경과시간[s]. 완료면 True 반환."""
        raise NotImplementedError

    def destroy_node(self):
        try:
            self._csv.close()
            print(f"\n마커 로그 저장: {self._csv_path}")
        except Exception:
            pass
        super().destroy_node()


def run(node_cls, bench=False, outdir='field_logs'):
    rclpy.init()
    node = node_cls(bench=bench, outdir=outdir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n중단 (Ctrl-C)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
