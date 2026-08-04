#!/usr/bin/env python3
"""watch_offboard.py — 오프보드 상태 실시간 감시 (읽기 전용)

  "명령한 값 vs 실제 움직임" 을 한 화면에서 나란히 본다.
  둘이 벌어지는 순간이 실외에서 사고가 나는 순간이므로, 나가기 전에
  여기서 추종이 붙는지 눈으로 확인한다.

  ★ 아무것도 발행하지 않는다. 구독만 한다. 비행 중에 켜도 안전하다.

사용:
    python3 watch_offboard.py                # 4Hz 갱신
    python3 watch_offboard.py --hz 10
    python3 watch_offboard.py --csv w.csv    # 화면 + CSV 동시 기록

보는 법
    SP  : 우리가 PX4 에 준 명령 (trajectory_setpoint / attitude_setpoint)
    ACT : 기체가 실제로 한 것 (odometry / local_position)
    ERR : 둘의 차이.  ★ 이게 커지거나 계속 늘면 위험 신호
    u   : PX4 제어기가 뽑아낸 제어입력 (thrust/torque setpoint) = UKF 의 u
"""
import argparse
import csv
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleStatus,
    VehicleLocalPosition, VehicleOdometry,
    VehicleThrustSetpoint, VehicleTorqueSetpoint,
)
from offboard_common import OffboardSequenceNode, Pose

NAN = float('nan')
LINES = 8          # 화면에 고정으로 쓰는 줄 수


def f3(v, w=7, p=2):
    return ''.join(f'{x:+{w}.{p}f}' if x is not None and math.isfinite(x)
                   else f'{"nan":>{w}}' for x in v)


class Watch(Node):
    # 토픽 접미사 해석기를 offboard_common 에서 그대로 빌려 쓴다.
    # (평범한 함수를 클래스에 붙이면 메서드가 된다 — 원본은 건드리지 않는다)
    GRAPH_KEY = OffboardSequenceNode.GRAPH_KEY
    GRAPH_WAIT_S = OffboardSequenceNode.GRAPH_WAIT_S
    _scan_graph = OffboardSequenceNode._scan_graph
    _make_resolver = OffboardSequenceNode._make_resolver

    def __init__(self, csv_path=None):
        super().__init__('watch_offboard')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        R = self._make_resolver()

        self.nav = -1
        self.armed = False
        self.sp_pos = [NAN] * 3
        self.sp_vel = [NAN] * 3
        self.sp_yaw = NAN
        self.sp_q = None
        self.sp_thr_body = NAN
        self.pos = [NAN] * 3
        self.vel = [NAN] * 3
        self.u_thrust = NAN
        self.u_torque = [NAN] * 3
        self.t_sp = 0.0            # 마지막 setpoint 수신 시각
        self.t_u = 0.0

        self.create_subscription(VehicleStatus, R('/fmu/out/vehicle_status'),
                                 self._cb_status, qos)
        self.create_subscription(TrajectorySetpoint, R('/fmu/in/trajectory_setpoint'),
                                 self._cb_traj, qos)
        self.create_subscription(VehicleAttitudeSetpoint,
                                 R('/fmu/in/vehicle_attitude_setpoint'),
                                 self._cb_att_sp, qos)
        self.create_subscription(VehicleLocalPosition,
                                 R('/fmu/out/vehicle_local_position'), self._cb_lp, qos)
        self.create_subscription(VehicleOdometry,
                                 R('/fmu/out/vehicle_odometry'), self._cb_odom, qos)
        self.create_subscription(VehicleThrustSetpoint,
                                 R('/fmu/out/vehicle_thrust_setpoint'), self._cb_thr, qos)
        self.create_subscription(VehicleTorqueSetpoint,
                                 R('/fmu/out/vehicle_torque_setpoint'), self._cb_tq, qos)

        self._csv = None
        if csv_path:
            self._csv_f = open(csv_path, 'w', newline='')
            self._csv = csv.writer(self._csv_f)
            self._csv.writerow(['t', 'nav', 'armed',
                                'sp_x', 'sp_y', 'sp_z', 'sp_vx', 'sp_vy', 'sp_vz',
                                'x', 'y', 'z', 'vx', 'vy', 'vz',
                                'u_thrust', 'u_tx', 'u_ty', 'u_tz'])
        self._t0 = time.time()
        self._remap_note = ' , '.join(getattr(self, '_remapped', [])) or '없음'

    # ── 콜백 ─────────────────────────────────────────────
    def _cb_status(self, m):
        self.nav = m.nav_state
        self.armed = (m.arming_state == VehicleStatus.ARMING_STATE_ARMED)

    def _cb_traj(self, m):
        self.sp_pos = [float(v) for v in m.position]
        self.sp_vel = [float(v) for v in m.velocity]
        self.sp_yaw = float(m.yaw)
        self.t_sp = time.time()

    def _cb_att_sp(self, m):
        self.sp_q = [float(v) for v in m.q_d]
        self.sp_thr_body = float(m.thrust_body[2])
        self.t_sp = time.time()

    def _cb_lp(self, m):
        if all(math.isfinite(v) for v in (m.x, m.y, m.z)):
            self.pos = [float(m.x), float(m.y), float(m.z)]
            self.vel = [float(m.vx), float(m.vy), float(m.vz)]

    def _cb_odom(self, m):
        if not math.isfinite(self.pos[0]) and all(math.isfinite(v) for v in m.position):
            self.pos = [float(v) for v in m.position]
            self.vel = [float(v) for v in m.velocity]

    def _cb_thr(self, m):
        self.u_thrust = float(m.xyz[2])       # FRD: 아래가 +, 보통 음수
        self.t_u = time.time()

    def _cb_tq(self, m):
        self.u_torque = [float(v) for v in m.xyz]

    # ── 표시 ─────────────────────────────────────────────
    def render(self):
        now = time.time()
        off = self.nav == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        mode = ('OFFBOARD' if off else f'nav={self.nav}')
        arm = 'ARMED' if self.armed else 'disarmed'
        sp_age = now - self.t_sp if self.t_sp else 1e9
        u_age = now - self.t_u if self.t_u else 1e9
        live = '흐름 O' if sp_age < 0.5 else ('흐름 X' if sp_age > 1e8 else f'{sp_age:.1f}s 끊김')

        # 위치·속도 오차 (setpoint 가 nan 이면 그 항목은 제어 안 하는 것)
        pe = [s - a for s, a in zip(self.sp_pos, self.pos)]
        ve = [s - a for s, a in zip(self.sp_vel, self.vel)]
        pe_n = math.sqrt(sum(x * x for x in pe if math.isfinite(x))) \
            if any(math.isfinite(x) for x in pe) else NAN
        ve_n = math.sqrt(sum(x * x for x in ve if math.isfinite(x))) \
            if any(math.isfinite(x) for x in ve) else NAN

        alt = -self.pos[2] if math.isfinite(self.pos[2]) else NAN
        att = ('  ATT SP  q_d=' + f3(self.sp_q, 6, 3) +
               f'  thrust_body={-self.sp_thr_body:+.3f}') if self.sp_q else \
              '  ATT SP  (미사용 — 위치/속도 모드)'

        warn = ''
        if off and sp_age > 0.5:
            warn = '  ⚠ 오프보드인데 setpoint 가 끊겼다 (1s 넘으면 PX4 가 이탈)'
        elif math.isfinite(pe_n) and pe_n > 1.5:
            warn = f'  ⚠ 위치 오차 {pe_n:.2f} m — 추종 실패'
        elif math.isfinite(ve_n) and ve_n > 1.0:
            warn = f'  ⚠ 속도 오차 {ve_n:.2f} m/s — 추종 실패'
        elif u_age > 1.0:
            warn = '  ⚠ thrust/torque setpoint 미수신 (yaml 에 없거나 링크 문제)'

        return [
            f'  {mode:<10} {arm:<9} setpoint {live:<9} 고도 {alt:6.2f} m',
            f'  SP  pos {f3(self.sp_pos)}   vel {f3(self.sp_vel)}   yaw {math.degrees(self.sp_yaw):+7.1f}°'
            if math.isfinite(self.sp_yaw) else
            f'  SP  pos {f3(self.sp_pos)}   vel {f3(self.sp_vel)}   yaw     nan',
            f'  ACT pos {f3(self.pos)}   vel {f3(self.vel)}',
            f'  ERR pos {f3(pe)}   vel {f3(ve)}   |pos| {pe_n:5.2f}  |vel| {ve_n:5.2f}',
            att,
            f'  u   thrust {-self.u_thrust:+.4f}   torque {f3(self.u_torque, 9, 4)}',
            f'  {warn}',
            '',
        ]

    def log_csv(self):
        if not self._csv:
            return
        self._csv.writerow([f'{time.time() - self._t0:.3f}', self.nav, int(self.armed)]
                           + [f'{v:.4f}' for v in self.sp_pos + self.sp_vel]
                           + [f'{v:.4f}' for v in self.pos + self.vel]
                           + [f'{self.u_thrust:.5f}']
                           + [f'{v:.5f}' for v in self.u_torque])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hz', type=float, default=4.0, help='화면 갱신 [Hz]')
    ap.add_argument('--csv', default=None, help='CSV 로도 기록할 경로')
    a = ap.parse_args()

    rclpy.init()
    n = Watch(a.csv)
    print('\n  ' + '─' * 74)
    print(f'  watch_offboard — 읽기 전용. 토픽 해석: {n._remap_note}')
    print('  SP=명령  ACT=실제  ERR=차이  u=제어입력.  Ctrl-C 로 종료')
    print('  ' + '─' * 74)
    print('\n' * LINES, end='')

    period = 1.0 / max(0.5, a.hz)
    last = 0.0
    try:
        while rclpy.ok():
            try:
                rclpy.spin_once(n, timeout_sec=0.05)
            except Exception:
                if not rclpy.ok():      # SIGTERM 등으로 컨텍스트가 내려간 것
                    break
                raise
            now = time.time()
            if now - last >= period:
                last = now
                n.log_csv()
                sys.stdout.write(f'\033[{LINES}A')      # 커서를 위로
                for ln in n.render():
                    sys.stdout.write('\033[2K' + ln + '\n')
                sys.stdout.flush()
    except KeyboardInterrupt:
        print('\n  중단 (Ctrl-C)')
    finally:
        if n._csv:
            n._csv_f.close()
            print(f'  CSV 저장: {a.csv}')
        n.destroy_node()
        if rclpy.ok():          # SIGTERM 등으로 이미 내려갔으면 두 번 부르지 않는다
            rclpy.shutdown()


if __name__ == '__main__':
    main()
