#!/usr/bin/env python3
"""watch_ekf.py — 비행 중 위치/고도만 본다 (읽기 전용)

  목적: 오프보드 중 **고도나 위치가 튀는 순간**을 현장에서 눈으로 잡는다.
  ★ 아무것도 발행하지 않는다. 비행 중 켜고 꺼도 안전하다.

사용:
    python3 watch_ekf.py
    python3 watch_ekf.py --csv ekf.csv     # 화면 + CSV (분석은 돌아와서)
    python3 watch_ekf.py --no-map
"""
import argparse
import csv
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import TrajectorySetpoint, VehicleStatus, VehicleLocalPosition
from offboard_common import OffboardSequenceNode

NAN = float('nan')


def n(v, w=7, p=2):
    return f'{v:{w}.{p}f}' if v is not None and math.isfinite(v) else f'{"--":>{w}}'


class WatchEKF(Node):
    GRAPH_KEY = OffboardSequenceNode.GRAPH_KEY
    GRAPH_WAIT_S = OffboardSequenceNode.GRAPH_WAIT_S
    _scan_graph = OffboardSequenceNode._scan_graph
    _make_resolver = OffboardSequenceNode._make_resolver

    def __init__(self, csv_path=None, show_map=True):
        super().__init__('watch_ekf')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        R = self._make_resolver()
        self.show_map = show_map

        self.nav = -1
        self.armed = False
        self.sp = [NAN] * 3
        self.pos = [NAN] * 3
        self.epv = self.eph = NAN
        self.ok_xy = self.ok_z = True
        self.rst_xy = self.rst_z = None
        self.events = []
        self.t_lp = -1e9
        self._prev_n = 0

        self.create_subscription(VehicleStatus, R('/fmu/out/vehicle_status'),
                                 self._cb_status, qos)
        self.create_subscription(TrajectorySetpoint, R('/fmu/in/trajectory_setpoint'),
                                 self._cb_sp, qos)
        self.create_subscription(VehicleLocalPosition,
                                 R('/fmu/out/vehicle_local_position'), self._cb_lp, qos)

        self._csv = None
        if csv_path:
            self._csv_f = open(csv_path, 'w', newline='')
            self._csv = csv.writer(self._csv_f)
            self._csv.writerow(['t', 'nav', 'armed', 'sp_x', 'sp_y', 'sp_alt',
                                'x', 'y', 'alt', 'eph', 'epv',
                                'xy_valid', 'z_valid', 'rst_xy', 'rst_z'])
        self._t0 = time.time()

    def _cb_status(self, m):
        self.nav = m.nav_state
        self.armed = (m.arming_state == 2)

    def _cb_sp(self, m):
        self.sp = [m.position[0], m.position[1], m.position[2]]

    def _cb_lp(self, m):
        self.pos = [m.x, m.y, m.z]
        self.eph, self.epv = m.eph, m.epv
        self.ok_xy, self.ok_z = bool(m.xy_valid), bool(m.z_valid)
        t = time.time() - self._t0

        rx, rz = int(m.xy_reset_counter), int(m.z_reset_counter)
        if self.rst_xy is None:
            self.rst_xy, self.rst_z = rx, rz          # 시작값 = 기준
        else:
            if rz != self.rst_z:
                self.events.append((t, f'고도가 {getattr(m, "delta_z", NAN):+.2f} m 튀었습니다'))
            if rx != self.rst_xy:
                d = getattr(m, 'delta_xy', [NAN, NAN])
                self.events.append((t, f'위치가 ({d[0]:+.2f}, {d[1]:+.2f}) m 튀었습니다'))
            self.rst_xy, self.rst_z = rx, rz
        self.events = self.events[-3:]
        self.t_lp = t

    def render(self):
        t = time.time() - self._t0
        mode = {0: '수동', 2: 'POSITION', 3: 'AUTO', 4: 'AUTO',
                14: '★오프보드'}.get(self.nav, f'모드{self.nav}')
        arm = '시동' if self.armed else '정지'
        alt_sp = -self.sp[2] if math.isfinite(self.sp[2]) else NAN
        alt = -self.pos[2] if math.isfinite(self.pos[2]) else NAN

        L = []
        L.append(f'  [{t:6.1f}s]  {mode}  {arm}')
        L.append('')
        L.append('              목표      실제')
        L.append(f'    x     {n(self.sp[0])}   {n(self.pos[0])}   m')
        L.append(f'    y     {n(self.sp[1])}   {n(self.pos[1])}   m')
        L.append(f'    고도   {n(alt_sp)}   {n(alt)}   m')
        L.append('')

        # 경고 — 있을 때만, 쉬운 말로
        w = []
        if (t - self.t_lp) > 1.0:
            w.append('위치 정보가 끊겼습니다')
        if not self.ok_z:
            w.append('EKF 고도 추정 무효')
        if not self.ok_xy:
            w.append('EKF 위치 추정 무효')
        if math.isfinite(self.epv) and self.epv > 1.0:
            w.append(f'고도 추정 불안정 (±{self.epv:.1f} m)')
        if math.isfinite(self.eph) and self.eph > 1.0:
            w.append(f'위치 추정 불안정 (±{self.eph:.1f} m)')
        L.append(('    ⚠ ' + '   ⚠ '.join(w)) if w else '    상태 정상')

        for (et, msg) in self.events:
            L.append(f'    ★★ {et:6.1f}s   {msg}')

        if self.show_map:
            L.append('')
            L.append(self._map())

        out = '\n'.join(L)
        sys.stdout.write('\033[F' * self._prev_n + '\033[J' + out + '\n')
        sys.stdout.flush()
        self._prev_n = out.count('\n') + 1

        if self._csv:
            self._csv.writerow([f'{t:.3f}', self.nav, int(self.armed),
                                f'{self.sp[0]:.4f}', f'{self.sp[1]:.4f}', f'{alt_sp:.4f}',
                                f'{self.pos[0]:.4f}', f'{self.pos[1]:.4f}', f'{alt:.4f}',
                                f'{self.eph:.3f}', f'{self.epv:.3f}',
                                int(self.ok_xy), int(self.ok_z),
                                self.rst_xy or 0, self.rst_z or 0])

    def _map(self):
        W, H, RNG = 31, 13, 8.0     # ±8 m (2026-08-10: 8×8m 운용·circle 7m). H=13→세로 1.33 m/칸
        g = [[' '] * W for _ in range(H)]
        for r in range(H):
            g[r][W // 2] = '|'
        for c in range(W):
            g[H // 2][c] = '-'
        g[H // 2][W // 2] = '+'

        def put(x, y, ch):
            if math.isfinite(x) and math.isfinite(y):
                c = int(round(W / 2 + y / RNG * (W / 2 - 1)))
                r = int(round(H / 2 - x / RNG * (H / 2 - 1)))
                if 0 <= r < H and 0 <= c < W:
                    g[r][c] = ch

        put(self.sp[0], self.sp[1], '+')
        put(self.pos[0], self.pos[1], 'O')
        return ('    위에서 본 모습 (±8 m,  북↑ 동→,   O=드론  +=목표)\n'
                + '\n'.join('    ' + ''.join(r) for r in g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hz', type=float, default=4.0)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--no-map', action='store_true')
    a = ap.parse_args()

    rclpy.init()
    node = WatchEKF(csv_path=a.csv, show_map=not a.no_map)
    print('watch_ekf — 읽기 전용. Ctrl-C 로 종료.\n')
    try:
        period = 1.0 / max(0.5, a.hz)
        nxt = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.time() >= nxt:
                node.render()
                nxt += period
    except KeyboardInterrupt:
        pass
    finally:
        if node._csv:
            node._csv_f.close()
            print(f'\nCSV 저장: {a.csv}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
