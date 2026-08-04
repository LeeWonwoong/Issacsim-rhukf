"""offboard_common.py — 실기 데이터 수집용 오프보드 노드 공통 골격 (2026-08-03)

설계 원칙
  1. ★★ setpoint 는 **오프보드일 때만** 발행한다. 이게 가장 중요하다.
     /fmu/in/trajectory_setpoint 는 PX4 내부의 uORB `trajectory_setpoint` 가 되는데,
     **조종기 수동비행의 FlightTask 도 같은 토픽에 발행**한다.
     오프보드가 아닐 때 우리가 계속 쓰면 두 발행자가 번갈아 덮어써서
     위치 컨트롤러가 조종기 명령과 우리 명령을 교대로 따른다
     → 호버가 돌고 떨리고 조종기가 안 먹는다 (2026-08-03 현장 실측).

     OffboardControlMode 는 전용 토픽이라 충돌이 없으므로 항상 발행해도 된다.
     다만 PX4 가 오프보드 진입을 수락하려면 setpoint 도 와 있어야 하므로,
     **조종사가 준비됐을 때 Enter 로 스트림을 연다**(ARM_STREAM).
  2. 스트림을 열면 기본 setpoint = **현재 위치 유지**.
     하드코딩 좌표를 쓰면 스위치를 켜는 순간 기체가 그 좌표로 날아간다.
  3. 오프보드 진입 순간 origin(x,y,z) 과 yaw0 을 스냅샷 → 이후 전부 이 기준.
     비행장 방향·기체 놓은 방향과 무관해진다.
  4. nav_state 가 OFFBOARD 를 벗어나면 즉시 SAFE. 조종사 인계가 항상 최우선.
  5. 이륙/착륙은 **사람이** 한다. 스크립트는 떠 있는 기체를 잠깐 넘겨받을 뿐이다.
  6. ★ 스크립트는 **수동비행이 안정된 뒤에** 실행한다.
     이륙 전부터 켜두면 수동비행 내내 setpoint 가 흘러 불필요한 부하·위험이 된다.
     PX4 는 스트림이 몇 초만 먼저 와 있으면 오프보드를 수락한다.

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
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleAttitudeSetpoint,
    VehicleCommand, VehicleStatus, VehicleLocalPosition, VehicleAttitude,
    VehicleOdometry,
)


class Pose:
    """vehicle_local_position 또는 vehicle_odometry 중 오는 쪽으로 채우는 상태.

    dds_topics.yaml 에 vehicle_local_position 이 등재돼 있지 않은 설치본이 흔하다.
    그 경우 vehicle_odometry 로 대체한다(position/velocity/q 를 모두 담고 있다).
    """
    __slots__ = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'heading', 'src', 'valid')

    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.heading = 0.0
        self.src = None
        self.valid = False

    @staticmethod
    def yaw_from_q(q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

NAN = float('nan')
PUB_HZ = 10.0          # 발행 주기.
#  PX4 요구 최소는 2Hz. 10Hz 면 충분하고, 20Hz 는 uXRCE-DDS 링크와 FC CPU 에
#  불필요한 부하를 준다(제어 루프 타이밍에 영향을 줄 수 있음).


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
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._cb_att, qos)
        # 위치/속도 소스 이중화 — 설치본에 따라 둘 중 하나만 노출돼 있다
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self._cb_lp, qos)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._cb_odom, qos)

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
        self.stream_armed = False      # Enter 로 여는 수동 개방(폴백)
        self.user_intent = -1          # 조종사가 고른 모드 (nav_state_user_intention)
        self._intent_seen = False
        # ★ OffboardControlMode 에 선언하는 제어 타입. 오프보드 수락 여부를 결정한다.
        #   실내(위치 추정 무효)에서 position 을 선언하면 PX4 가 모드를 거부한다.
        self._ocm_kind = 'attitude' if bench else 'position'

        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(outdir, f'{self.SEQ_NAME}_{stamp}.csv')
        self._csv = open(self._csv_path, 'w', newline='')
        self._w = csv.writer(self._csv)
        self._w.writerow(['t_wall', 't_seq', 'state', 'stage', 'nav_state', 'armed',
                          'x', 'y', 'z', 'vx', 'vy', 'vz', 'heading'])

        self.create_timer(1.0 / PUB_HZ, self._tick)
        threading.Thread(target=self._stdin_waiter, daemon=True).start()
        self.get_logger().info(
            f"\n{'='*68}\n  {self.SEQ_NAME.upper()}  {'[BENCH — 실내/프로펠러 제거]' if bench else ''}\n"
            f"  OffboardControlMode({self._ocm_kind}) 만 {PUB_HZ:.0f}Hz 로 발행 중.\n"
            f"  setpoint 는 아직 **발행하지 않습니다** (조종기 수동비행과 충돌하므로).\n"
            f"  PX4 는 OffboardControlMode 만으로 오프보드를 수락하므로 이대로 충분합니다.\n"
            f"\n"
            f"  순서:  조종기로 이륙·안정 → **오프보드 스위치 ON** → 자동으로 스트림 개방\n"
            f"        (Enter 는 폴백일 뿐입니다. 평소엔 누르지 마세요 — 수동비행 중\n"
            f"         setpoint 가 흘러 조종기와 충돌합니다.)\n"
            f"\n"
            f"  감시할 토픽: ros2 topic hz "
            f"{'/fmu/in/vehicle_attitude_setpoint' if bench else '/fmu/in/trajectory_setpoint'}\n"
            f"  마커 로그: {self._csv_path}\n{'='*68}")

    # ── 콜백 ─────────────────────────────────────────────
    def _cb_status(self, m):
        prev = self.nav_state
        self.nav_state, self.arming = m.nav_state, m.arming_state
        # ★ 조종사가 고른 모드. 오프보드 스위치를 켜면 setpoint 가 아직 없어서
        #   nav_state 는 안 바뀌지만 이 값은 즉시 OFFBOARD 가 된다.
        #   이걸 보고 스트림을 열면 조종사는 스위치만 켜면 되고,
        #   수동비행 중에는 setpoint 가 나가지 않아 충돌도 없다.
        self.user_intent = getattr(m, 'nav_state_user_intention', -1)
        if (not self._intent_seen) and self.user_intent == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self._intent_seen = True
            self.get_logger().info('  ▶ 조종사가 오프보드를 선택함 → setpoint 스트림 개방')
        if prev != self.nav_state:
            self.get_logger().info(f"  nav_state {prev} → {self.nav_state}"
                                   f"{'  (OFFBOARD)' if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD else ''}")

    def _cb_lp(self, m):
        p = self.lp or Pose()
        p.x, p.y, p.z = float(m.x), float(m.y), float(m.z)
        p.vx, p.vy, p.vz = float(m.vx), float(m.vy), float(m.vz)
        if math.isfinite(m.heading):
            p.heading = float(m.heading)
        p.valid = bool(m.xy_valid and m.z_valid)
        p.src = 'local_position'
        self._first_src('local_position')
        self.lp = p

    def _cb_odom(self, m):
        # local_position 이 이미 오고 있으면 그쪽을 우선한다
        if self.lp is not None and self.lp.src == 'local_position':
            return
        p = self.lp or Pose()
        pos, vel = m.position, m.velocity
        if all(math.isfinite(v) for v in pos):
            p.x, p.y, p.z = float(pos[0]), float(pos[1]), float(pos[2])
            p.valid = True
        else:
            p.valid = False
        if all(math.isfinite(v) for v in vel):
            p.vx, p.vy, p.vz = float(vel[0]), float(vel[1]), float(vel[2])
        if math.isfinite(m.q[0]):
            p.heading = Pose.yaw_from_q(m.q)
        p.src = 'odometry'
        self._first_src('odometry')
        self.lp = p

    def _first_src(self, name):
        if getattr(self, '_src_announced', None) != name:
            self._src_announced = name
            self.get_logger().info(f'  위치/속도 소스 = /fmu/out/vehicle_{name}')

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

    def _may_stream(self):
        """setpoint 를 발행해도 되는가.

        셋 중 하나면 발행한다:
          1) 이미 오프보드              (nav_state == OFFBOARD)
          2) ★ 조종사가 오프보드를 선택 (nav_state_user_intention == OFFBOARD)
             스위치를 켜는 순간 이 값이 바뀐다. setpoint 가 아직 없어
             nav_state 는 안 바뀌지만, 우리가 이걸 보고 스트림을 열어주면
             PX4 가 곧바로 오프보드로 진입한다.
          3) Enter 로 수동 개방 (폴백 — 위 필드가 없는 펌웨어용)

        그 외(수동비행 중)에는 발행하지 않는다.
        /fmu/in/trajectory_setpoint 는 조종기 FlightTask 와 같은 uORB 토픽이라
        동시에 쓰면 위치 컨트롤러가 둘을 교대로 따른다.
        """
        return (self.offboard
                or self.user_intent == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                or self.stream_armed)

    def _stdin_waiter(self):
        try:
            input()
        except Exception:
            return
        self.stream_armed = True
        self.get_logger().info(
            "\n  ▶ setpoint 스트림 열림. 3초 안에 오프보드 스위치를 켜세요.\n")

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
        if position:
            self._ocm_kind = 'position'
        elif velocity:
            self._ocm_kind = 'velocity'
        elif attitude:
            self._ocm_kind = 'attitude'

    def _keepalive_ocm(self):
        """setpoint 를 아직 안 보낼 때도 OffboardControlMode 는 계속 흘려야 한다.

        PX4 는 **OffboardControlMode 만** 보고 오프보드 가능 여부를 판정한다
        (setpoint 메시지는 모드 진입 조건이 아니다).
        `HealthAndArmingChecks/checks/offboardCheck.cpp`:

            offboard_available = (어떤 타입이든 선언됨) && (COM_OF_LOSS_T=1s 이내 수신)
            if (position && local_position_invalid)  → 거부 "Offboard requires local position"
            if (velocity && local_velocity_invalid)  → 거부 "Offboard requires local velocity"
            if (attitude && attitude_invalid)        → 거부 "Offboard requires attitude estimate"

        ★ 그래서 **선언 타입을 실제로 보낼 타입과 맞춰야** 한다.
          실내 지상검증(bench)에서 position 을 선언하면 위치 추정이 무효라
          오프보드 스위치가 아예 먹지 않는다. 이때 bench 는 attitude 를 선언한다.
        """
        self._ocm(**{self._ocm_kind: True})

    def send_position(self, x, y, z, yaw):
        if not self._may_stream():
            return
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
        if not self._may_stream():
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
        if not self._may_stream():
            return
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
        self._keepalive_ocm()           # 전용 토픽이라 충돌 없음 — 항상 발행
        # ★ bench 는 위치 추정이 필요 없다(자세 명령). lp 를 기다리면 실내에서
        #   영원히 아무것도 발행하지 못해 오프보드 진입 자체가 막힌다.
        if self.lp is None and not self.bench:
            self.warn_once('nolp', '  위치/속도 수신 대기 중... '
                                   '(vehicle_local_position 또는 vehicle_odometry)')
            return
        if not self._may_stream():
            self.warn_once('nostream',
                           '  대기 중 — 오프보드 스위치를 켜면 자동으로 스트림이 열립니다 '
                           '(안 되면 Enter)')
            # ★ 여기서 return 하면 안 된다.
            #   발행은 send_* 안의 가드가 이미 막고 있고, 상태기계는 계속 돌아야 한다.
            #   (조종사가 인계했을 때 ENGAGED → WAIT 복귀가 여기서 일어나므로,
            #    return 하면 상태가 ENGAGED 에 갇혀 재진입 시 옛 origin 을 쓴다)

        if self.state == 'WAIT':
            self.hold_here()
            if self.offboard:
                if not self._preflight_ok():
                    return
                self.origin = ((self.lp.x, self.lp.y, self.lp.z)
                               if self.lp is not None else (0.0, 0.0, 0.0))
                self.yaw0 = self.heading()
                self.t_engage = time.time()
                self.state = 'ENGAGED'
                self.get_logger().info(
                    f"\n  ★ OFFBOARD 진입 — 시퀀스 시작\n"
                    f"    origin = ({self.origin[0]:+.2f}, {self.origin[1]:+.2f}, {self.origin[2]:+.2f}) NED\n"
                    f"    yaw0   = {math.degrees(self.yaw0):+.1f}°  (이 방향 기준으로 궤적을 만듭니다)\n"
                    + (f"    [BENCH] 추력 {self.bench_thrust:.2f} — 프로펠러 제거 확인!\n"
                       if self.bench else ""))

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
        if not self.lp.valid:
            self.warn_once('valid', f'  ✗ 위치 추정 무효 (소스={self.lp.src}) — 시퀀스 시작 거부')
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
        lp = self.lp if self.lp is not None else Pose()   # bench 는 lp 없이도 돈다
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
