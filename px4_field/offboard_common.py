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
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    # Humble: rclpy.init() 이 심는 SIGINT 핸들러가 컨텍스트를 먼저 내리면
    # spin() 은 KeyboardInterrupt 가 아니라 이걸 던진다. 안 잡으면 postflight 가 통째로 날아간다.
    from rclpy.executors import ExternalShutdownException
except ImportError:                                   # 구버전 대비
    class ExternalShutdownException(Exception):
        pass

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
    NEED_ALT = 1.0             # 진입 최소 고도 [m] (2026-08-10: 1m 저고도 운용. bench 무시)
    MAX_RADIUS = 25.0          # origin 기준 수평 이탈 한계 [m]
    MAX_ALT_DEV = 15.0         # origin 기준 고도 이탈 한계 [m]

    BENCH_THRUST = 0.10        # 프로펠러 뺀 지상검증 시 추력. 모터가 저속으로만 돈다

    GRAPH_WAIT_S = 20.0        # 토픽 접미사 자동 감지 시 ROS 그래프 대기 [s]
    GRAPH_KEY = '/fmu/out/vehicle_status'   # 이게 보여야 해석을 신뢰할 수 있다

    # ── PX4 메시지 버전 접미사(_v<N>) 자동 해석 ──────────
    #  PX4 v1.16+ 는 MESSAGE_VERSION != 0 인 메시지의 DDS 토픽명 뒤에 _v<N> 을
    #  자동으로 붙인다 (uxrce_dds_client/utilities.hpp:35, dds_topics.h.em:83).
    #  버전은 yaml 이 아니라 msg 정의에서 오므로 빌드한 소스 버전이 곧 이름을 정한다.
    #
    #      실기 FC (v1.15.4)   : /fmu/out/vehicle_status
    #      Isaac SITL (v1.18)  : /fmu/out/vehicle_status_v4
    #                            /fmu/in/vehicle_attitude_setpoint_v1
    #                            /fmu/out/vehicle_local_position_v1
    #      (TrajectorySetpoint · VehicleCommand · VehicleOdometry ·
    #       VehicleAttitude · OffboardControlMode 은 VERSION=0 → 접미사 없음)
    #
    #  한 코드가 양쪽에서 그대로 돌도록 ROS 그래프를 보고 실재하는 이름을 고른다.
    #  같은 발상의 선례: online_rl_main.py 의 PX4 네임스페이스 자동 감지.
    def _scan_graph(self):
        """ROS 그래프가 안정될 때까지 기다린 뒤 토픽 이름 목록을 돌려준다.

        ⚠ 여기서 성급하게 반환하면 안 된다. ROS2 디스커버리는 비동기라
          '/fmu/ 토픽이 하나 보인다'가 '전부 보인다'를 뜻하지 않는다.
          실제 사고(2026-08-04, Isaac SITL): 첫 /fmu/ 토픽을 보고 즉시
          반환하는 바람에 아직 발견되지 않은 vehicle_status_v4 를 놓쳤고,
          접미사 없는 이름으로 폴백 → 구독이 영영 아무것도 못 받음 →
          오프보드로 전환해도 스크립트가 감지하지 못했다.

        그래서 두 조건을 모두 만족해야 반환한다:
          ① 핵심 토픽(vehicle_status 또는 vehicle_status_v<N>)이 보인다
          ② 토픽 개수가 0.6s 동안 더 늘지 않는다 (디스커버리 정착)
        """
        t0 = time.time()
        prev, stable, names = -1, 0, []
        while time.time() - t0 < self.GRAPH_WAIT_S:
            names = [n for n, _ in self.get_topic_names_and_types()]
            # ★ 이름 존재로 판정하면 안 된다 — 구독자만 있어도 목록에 뜬다.
            #   실제로 **발행자가 있는** 것을 봐야 한다.
            has_key = any(self.count_publishers(n) > 0 for n in names
                          if n == self.GRAPH_KEY or n.startswith(self.GRAPH_KEY + '_v'))
            stable = stable + 1 if len(names) == prev else 0
            prev = len(names)
            if has_key and stable >= 3:
                return names
            time.sleep(0.2)

        self.get_logger().error(
            f'\n  ✗ {self.GRAPH_WAIT_S:.0f}s 안에 {self.GRAPH_KEY}* 를 찾지 못했다.\n'
            f'    접미사 없는 이름으로 진행하지만 **오프보드 감지가 안 될 수 있다.**\n'
            f'    확인:  ros2 topic list | grep /fmu/out/vehicle_status\n'
            f'    · 아무것도 없다 → 기체/SITL 또는 MicroXRCEAgent 미기동,\n'
            f'                      혹은 ROS_DOMAIN_ID 불일치\n'
            f'    · _v<N> 이 보인다 → 이 스크립트를 다시 실행하면 잡힌다\n')
        return names

    def _make_resolver(self, suffix=None):
        """base 토픽명 → 그래프에 실재하는 이름(base 또는 base_v<N>) 으로 매핑."""
        if suffix == 'none':
            self.get_logger().info('  토픽 접미사 = 강제 없음 (PX4 v1.15 형식)')
            return lambda base: base

        names = self._scan_graph()
        self._remapped = []

        def resolve(base):
            """★ 이름이 그래프에 있는지로 고르면 안 된다.

            `ros2 topic list` 는 **구독자만 있어도** 이름을 보여준다. 우리가
            틀린 이름으로 구독을 만들면 그 이름이 그래프에 생기고, 다음 판정에서
            "있네" 하고 그대로 쓰게 된다. 발행자가 0 이라 영원히 아무것도 안 온다.
            (2026-08-04 Isaac SITL 실측 — 오프보드 전환이 감지되지 않던 원인)

                /fmu/out/vehicle_status      pub 0  sub 1   ← 우리가 만든 죽은 구독
                /fmu/out/vehicle_status_v4   pub 1  sub 1   ← PX4 가 실제 발행

            그래서 **상대편이 실제로 붙어 있는지**를 센다. 방향이 반대다:
                /fmu/out/*  PX4 가 발행 → 우리가 구독 → 발행자 수를 본다
                /fmu/in/*   PX4 가 구독 → 우리가 발행 → 구독자 수를 본다
            """
            count = (self.count_publishers if base.startswith('/fmu/out/')
                     else self.count_subscribers)
            k = len(base) + 2
            cands = sorted((n for n in names
                            if n.startswith(base + '_v') and n[k:].isdigit()),
                           key=lambda n: -int(n[k:]))          # 높은 버전 우선
            for c in [base] + cands:
                if count(c) > 0:
                    if c != base:
                        self._remapped.append(f'{base} → {c}')
                    return c
            # 상대가 아직 안 붙었다 — 접미사 후보가 보이면 그쪽이 맞을 확률이 높다
            if cands:
                self._remapped.append(f'{base} → {cands[0]} (상대 미확인, 추정)')
                return cands[0]
            return base
        return resolve

    def __init__(self, bench=False, outdir='field_logs', suffix=None):
        super().__init__(f'offb_{self.SEQ_NAME}')
        self.bench = bench
        self.bench_thrust = self.BENCH_THRUST
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        R = self._make_resolver(suffix)   # PX4 메시지 버전 접미사(_v<N>) 해석기

        # 해석된 이름을 보관한다 — 배너에서 "감시할 토픽"으로 그대로 안내해야
        # 사용자가 복사한 ros2 topic hz 가 실제로 잡힌다.
        self.t_ocm = R('/fmu/in/offboard_control_mode')
        self.t_traj = R('/fmu/in/trajectory_setpoint')
        self.t_att = R('/fmu/in/vehicle_attitude_setpoint')

        self.pub_ocm = self.create_publisher(OffboardControlMode, self.t_ocm, qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, self.t_traj, qos)
        self.pub_att = self.create_publisher(VehicleAttitudeSetpoint, self.t_att, qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, R('/fmu/in/vehicle_command'), qos)

        self.t_status = R('/fmu/out/vehicle_status')
        self.create_subscription(VehicleStatus, self.t_status, self._cb_status, qos)
        self.create_subscription(VehicleAttitude, R('/fmu/out/vehicle_attitude'), self._cb_att, qos)
        # 위치/속도 소스 이중화 — 설치본에 따라 둘 중 하나만 노출돼 있다
        self.create_subscription(VehicleLocalPosition, R('/fmu/out/vehicle_local_position'), self._cb_lp, qos)
        self.create_subscription(VehicleOdometry, R('/fmu/out/vehicle_odometry'), self._cb_odom, qos)

        # 상태
        self._t_start = time.time()
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
            + (f"  토픽 해석(PX4 버전 접미사): {' , '.join(self._remapped)}\n"
               if getattr(self, '_remapped', None) else
               "  토픽 해석: 접미사 없음 (PX4 v1.15 형식)\n")
            + f"  상태 토픽  : {self.t_status}   ← 이게 안 오면 오프보드 감지 불가\n"
            f"  감시할 토픽: ros2 topic hz {self.t_att if bench else self.t_traj}\n"
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

    def _slew_yaw(self, yaw):
        """★ 2026-08-13 요 슬루레이트 제한 (env YAW_SLEW_RATE, rad/s; 0=끔).
        sim online_rl_main._send_setpoint 와 동일 로직 — waypoint 코너의 yaw 순간회전이
        만드는 |ω| 스파이크를 없앤다. sim·실기 대칭이라야 궤적·NIS 비교가 성립한다."""
        yr = getattr(self, '_yaw_slew_rate', None)
        if yr is None:
            self._yaw_slew_rate = float(os.environ.get('YAW_SLEW_RATE', '1.57') or 0.0)
            yr = self._yaw_slew_rate
            self._yaw_cmd_prev = None
        yaw = wrap_pi(yaw)
        if yr <= 0.0:
            return yaw
        dt = 1.0 / PUB_HZ
        if getattr(self, '_yaw_cmd_prev', None) is None:
            self._yaw_cmd_prev = float(yaw)
        err = wrap_pi(float(yaw) - self._yaw_cmd_prev)
        step = max(-yr*dt, min(yr*dt, err))
        self._yaw_cmd_prev = self._yaw_cmd_prev + step
        return wrap_pi(self._yaw_cmd_prev)

    def send_position(self, x, y, z, yaw):
        if not self._may_stream():
            return
        self._ocm(position=True)
        m = TrajectorySetpoint()
        m.position = [float(x), float(y), float(z)]
        m.velocity = [NAN, NAN, NAN]
        m.acceleration = [NAN, NAN, NAN]
        m.yaw = float(self._slew_yaw(yaw))
        m.yawspeed = NAN
        m.timestamp = 0
        self.pub_traj.publish(m)

    def send_position_velocity(self, x, y, z, yaw, vx, vy, vz):
        """위치 + 속도 피드포워드를 함께 보낸다 (sim 의 _send_setpoint 와 동일 형식).

        ★ 왜 별도 메서드인가
          sim(online_rl_main._send_setpoint)은 TrajectorySetpoint 에 position 과
          velocity 를 **같이** 채우고 OffboardControlMode.position=True 로 보낸다.
          기존 send_position 은 velocity=NaN, send_velocity 는 position=NaN 이라
          어느 쪽도 sim 과 같은 명령이 아니다. 기동 패턴(F5~F7)을 sim 과 비교하려면
          명령 자체가 같아야 하므로 이 메서드를 쓴다.
        """
        if self.bench:
            self.send_attitude(0.0, 0.0, yaw, self.bench_thrust)
            return
        if not self._may_stream():
            return
        self._ocm(position=True)
        m = TrajectorySetpoint()
        m.position = [float(x), float(y), float(z)]
        m.velocity = [float(vx), float(vy), float(vz)]
        # ★2026-09-10 가속도 피드포워드 — sim(online_rl_main._send_setpoint, ACCEL_FF=1)과 동일.
        #   속도 FF 의 유한차분(PUB_HZ). 없으면 PX4 위치루프가 구심가속을 위치오차로만 만든다:
        #   e = a_c/(P_v·P_xy) = 0.49/(1.8·0.95) ≈ 0.29 m 바깥 (sim circle 실측 +0.32 m). 수평 노름 3.0·수직 ±2.0 클립.
        acc = [NAN, NAN, NAN]
        v = (float(vx), float(vy), float(vz))
        if getattr(self, 'accel_ff', True) and all(math.isfinite(c) for c in v):
            vp = getattr(self, '_vel_cmd_prev', None)
            if vp is not None:
                dt = 1.0 / PUB_HZ
                ax, ay, az = [(v[i] - vp[i]) / dt for i in range(3)]
                n = math.hypot(ax, ay)
                if n > 3.0:
                    ax, ay = ax * 3.0 / n, ay * 3.0 / n
                az = max(-2.0, min(2.0, az))
                acc = [ax, ay, az]
            self._vel_cmd_prev = v
        else:
            self._vel_cmd_prev = None
        m.acceleration = [float(a) for a in acc]
        m.yaw = float(self._slew_yaw(yaw))
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
        m.yaw = float(self._slew_yaw(yaw))
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

        # ★ 런타임 안전망: vehicle_status 가 안 오면 오프보드를 영영 감지 못 한다.
        #   조용히 대기만 하다 "스위치가 안 먹네" 로 오해하게 되므로 크게 알린다.
        if self.nav_state < 0 and time.time() - self._t_start > 5.0:
            self.warn_once('nostatus',
                           f'\n  ✗ {self.t_status} 를 5초간 한 번도 못 받았다.\n'
                           f'    → 이 상태로는 오프보드로 전환해도 감지하지 못한다.\n'
                           f'    확인:  ros2 topic list | grep /fmu/out/vehicle_status\n'
                           f'    · _v<N> 이 붙은 이름이 보인다 → 스크립트를 재시작하면 잡힌다\n'
                           f'    · 아무것도 없다 → agent 미기동 또는 ROS_DOMAIN_ID 불일치\n')
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
        if not self.lp.valid:
            self.warn_once('valid', f'  ✗ 위치 추정 무효 (소스={self.lp.src}) — 시퀀스 시작 거부')
            return False
        # ★ 고도 게이트 제거(2026-08-11): RNG_CTRL=2 로 저고도 EKF 를 신뢰할 수 있으므로
        #   어떤 고도든(지상 포함) 오프보드 진입 허용. (구: alt<NEED_ALT 면 "고도 부족" 거부)
        #   위치 추정 유효성(위)만 남긴다 — 이건 안전상 유지.
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


def add_postflight_args(ap):
    """비행 스크립트 4개가 공통으로 갖는 사후처리 인자. argparse 에 붙여 쓴다."""
    ap.add_argument('--no-fetch', dest='fetch', action='store_false',
                    help='비행 후 ulog 자동 회수·자동 판정을 하지 않는다')
    ap.add_argument('--fetch-conn', dest='fetch_conn', default=None,
                    help='ulog 회수용 MAVLink 연결. 생략하면 /dev/ttyACM0 → SITL 자동')
    import datetime as _dt
    ap.add_argument('--ulog-dir', dest='ulog_dir',
                    default='ulog_' + _dt.date.today().strftime('%Y%m%d'),
                    help='받은 ulog 를 둘 폴더 (기본 ./ulog_YYYYMMDD — 날짜별 자동분리)')
    ap.add_argument('--wait-disarm', dest='wait_disarm', type=float, default=600.0,
                    help='착륙·disarm 을 최대 몇 초 기다릴지 (로그는 disarm 때 닫힌다)')
    ap.add_argument('--mass', type=float, default=1.340, help='AUW [kg] — 판정에 쓴다')


def postflight(seq_name, check_type, args):
    """비행이 끝나면 ulog 를 **자동으로 받아 자동으로 판정**한다.

    왜 이 자리인가
      · 로그는 disarm 때 닫힌다(SDLOG_MODE=0). 그래서 fetch_ulog 가 disarm 을
        기다렸다가 받는다 — 시퀀스 종료 시점에는 아직 날고 있다.
      · 파일명에 **시퀀스 이름이 박힌다**(f5_circle_log47_...). "log_NN 이 뭐였는지"
        헷갈리던 문제가 사라진다 (2026-08-06 에 실제로 헷갈렸다).
      · check_ulog 가 종료코드로 PASS/FAIL 을 말한다. **사람이 판정하지 않는다.**
        FAIL 이면 짐 싸기 전에 다시 뜨면 된다.

    실패해도 비행 스크립트를 죽이지 않는다 — 로그는 SD 에 그대로 남아 있고,
    나중에 손으로 `fetch_ulog.py` 를 돌리면 된다.
    """
    import subprocess, glob, re, signal
    here = os.path.dirname(os.path.abspath(__file__))
    # ★ 비행 종료용 Ctrl-C 를 연타하면 그 두 번째가 회수 준비 구간을 죽인다.
    #   준비가 끝나고 실제 대기에 들어갈 때까지만 SIGINT 를 막는다(그 뒤엔 정상 동작).
    try:
        _prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):                            # 메인스레드가 아니면 못 건다
        _prev_sigint = None
    # 같은 시퀀스를 여러 번 날려도 안 덮어쓰게 run 번호를 붙인다.
    #  ulog_dir 의 {seq_name}_runN 중 최대 N+1 (예전 세션 것도 세므로 절대 안 겹친다).
    _rn = 0
    for _p in glob.glob(os.path.join(os.path.expanduser(args.ulog_dir), f'{seq_name}_run*')):
        _m = re.search(re.escape(seq_name) + r'_run(\d+)', os.path.basename(_p))
        if _m:
            _rn = max(_rn, int(_m.group(1)))
    name = f'{seq_name}_run{_rn + 1}'
    manual = f"python3 {here}/fetch_ulog.py --name {name} --out {args.ulog_dir}"
    print("\n" + "=" * 68)
    print(f"  사후처리 — ulog 자동 회수 + 자동 판정   ({name})")
    print("=" * 68)
    print(f"  저장이름: {name}_log<id>_<시각>.ulg   ← 시퀀스+run 번호가 박힌다(덮어쓰기 없음)")
    print("  ★ 지금 착륙 → disarm 하면 자동으로 받는다.  로그는 disarm 때 닫히므로")
    print("     arm 중엔 못 받는다. 착륙·disarm 만 하면 아래에 '대기→수신'이 실시간으로 뜬다.")
    print("     (정말 건너뛰려면 여기서 Ctrl-C — 그럼 위 이름으로 나중에 손수 받는다)")

    cmd = [sys.executable, os.path.join(here, 'fetch_ulog.py'),
           '--name', name, '--out', args.ulog_dir,
           '--wait-disarm', str(args.wait_disarm)]     # 스트리밍(진행 실시간) — quiet-list 뺌
    if args.fetch_conn:
        cmd += ['--conn', args.fetch_conn]
    sys.stdout.flush()
    # 자식을 별도 세션에 둔다 → 터미널 Ctrl-C 가 자식에게 **직접** 가지 않는다.
    #  (같은 프로세스그룹이면 SIGINT 가 부모·자식에 동시에 꽂혀 다운로드가 중간에 끊기고
    #   깨진 파일이 남는다.) 중단은 부모가 받아서 명시적으로 정리한다.
    proc = None
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
        if _prev_sigint is not None:                         # 여기서부터 Ctrl-C 정상 동작
            signal.signal(signal.SIGINT, _prev_sigint)
        proc.wait()
    except KeyboardInterrupt:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"\n  ⚠ 회수 중단(Ctrl-C). disarm 후 이 명령으로 받으세요:\n     {manual}")
        return
    except Exception as e:                                   # noqa: BLE001
        print(f"  ⚠ 회수 실패: {e}\n     손으로:  {manual}")
        return
    finally:
        if _prev_sigint is not None:                         # 어느 경로로 나가든 원복
            try:
                signal.signal(signal.SIGINT, _prev_sigint)
            except (ValueError, OSError):
                pass

    # 스트리밍이라 stdout 파싱 대신 이름으로 찾는다 (run 번호가 유일성 보장)
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(args.ulog_dir), f'{name}_*.ulg')),
                  key=os.path.getmtime)
    if not hits:
        print(f"  ⚠ 받은 파일이 없습니다(disarm 했는지 확인). 손으로:\n     {manual}")
        return
    if not check_type:
        print(f"  파일 받음: {hits[-1]}")
        return

    path = hits[-1]
    print("\n" + "=" * 68)
    print(f"  자동 판정 — check_ulog --type {check_type}")
    print("=" * 68)
    cmd2 = [sys.executable, os.path.join(here, 'check_ulog.py'), path,
            '--type', check_type, '--mass', str(args.mass)]
    sys.stdout.flush()
    try:
        rc = subprocess.run(cmd2).returncode
    except Exception as e:                                   # noqa: BLE001
        print(f"  ⚠ 판정 실패(pyulog/numpy 없음?): {e}")
        print(f"     파일은 받았습니다: {path}")
        return

    print()
    if rc == 0:
        print(f"  ★ 이 소티는 **쓸 수 있다**.  {path}")
    elif rc == 1:
        print(f"  ★ 이 소티는 **다시 떠야 한다.** 위 FAIL 항목을 볼 것.  {path}")
    else:
        print(f"  ⚠ 판정을 못 했다(로그 열기 실패). 파일은 받았다: {path}")


def run(node_cls, bench=False, outdir='field_logs', args=None, check_type=None):
    rclpy.init()
    node = node_cls(bench=bench, outdir=outdir)
    seq_name = getattr(node, 'SEQ_NAME', 'seq')
    check_type = check_type or getattr(node, 'CHECK_TYPE', None)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ★ 2026-08-12: ExternalShutdownException 을 안 잡아서 자동회수가 안 됐다.
        #   Humble 은 Ctrl-C 때 rclpy 의 SIGINT 핸들러가 컨텍스트를 먼저 내리고,
        #   그러면 spin 은 KeyboardInterrupt 가 아니라 이걸 던진다.
        print('\n중단 (Ctrl-C)')
    finally:
        # ★★ 여기서 예외가 새어 나가면 아래 postflight 가 **호출조차 안 된다** —
        #   자동회수가 매번 실패하고 손으로 받게 되던 진짜 원인이었다.
        #   구 코드의 rclpy.shutdown() 은 이미 내려간 컨텍스트에 대해
        #   RuntimeError('Context must be initialized before it can be shutdown') 를 던진다.
        try:
            node.destroy_node()
        except Exception as _e:                              # noqa: BLE001
            print(f'  (노드 정리 중 무시된 예외: {_e})')
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as _e:                              # noqa: BLE001
            print(f'  (rclpy 종료 중 무시된 예외: {_e})')

    # 벤치(프로펠러 제거)는 비행이 아니므로 회수할 로그가 없다
    if bench:
        return
    if args is None or not hasattr(args, 'fetch'):
        # 스크립트가 add_postflight_args 를 안 붙인 경우 — 조용히 넘어가지 말고 알려준다.
        print('\n  ⚠ 자동회수 인자가 없습니다(add_postflight_args 미적용).'
              '\n     손으로:  python3 fetch_ulog.py --name ' + seq_name)
        return
    if not args.fetch:
        print('\n  (--no-fetch 지정 — 자동회수 건너뜀)')
        return
    postflight(seq_name, check_type, args)
