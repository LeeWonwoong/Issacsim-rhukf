#!/usr/bin/env python3
"""f3_drag.py — F3 직선 왕복 (등속 구간)

얻는 것: drag_x (전후) · drag_y (좌우)

★ yaw 를 yaw0 로 고정하는 이유
  항력은 **바디 좌표**에서 정의된다(drag_x=전후, drag_y=좌우).
  둘을 분리하려면 속도가 바디 +x 인 구간과 바디 +y 인 구간이 따로 있어야 한다.
  yaw 를 진행방향으로 돌리면 항상 전진 비행이 되어 **drag_y 를 영영 못 잰다.**

★ 속도 setpoint 를 쓰는 이유
  위치 setpoint 는 EKF 원점 기준이라 좌표 매칭이 필요하지만,
  속도는 그런 문제가 없고 등속을 정확히 유지할 수 있다.

사용:
    python3 f3_drag.py                  # 실비행
    python3 f3_drag.py --bench          # 프로펠러 뺀 지상 검증
    python3 f3_drag.py --v 3.0 --leg 8

절차: Position 모드로 수동 이륙 → 고도 5m 안정 → 오프보드 스위치 ON
필요 공간: v × leg × 2 + 여유  (기본 3m/s × 8s = 24m, 왕복 48m + 여유)
"""
import argparse
from offboard_common import OffboardSequenceNode, run


class F3Drag(OffboardSequenceNode):
    SEQ_NAME = 'f3_drag'
    NEED_ALT = 3.0

    def __init__(self, bench=False, outdir='field_logs', v=3.0, leg=8.0, pause=3.0, settle=4.0):
        self.v, self.leg, self.pause, self.settle = v, leg, pause, settle
        # 구간: (축, 부호)  전진→후진→횡진+→횡진−, 각 2왕복
        self.legs = [('fwd', +1), ('fwd', -1), ('fwd', +1), ('fwd', -1),
                     ('lat', +1), ('lat', -1), ('lat', +1), ('lat', -1)]
        self.T_leg = self.leg + self.pause
        super().__init__(bench=bench, outdir=outdir)

    def step(self, t):
        if t < self.settle:
            self.hold_origin()
            self.set_stage(f'SETTLE {t:.0f}/{self.settle:.0f}s')
            return False
        te = t - self.settle
        i = int(te // self.T_leg)
        if i >= len(self.legs):
            self.hold_origin()
            self.set_stage('DONE_HOLD')
            return True
        tl = te - i * self.T_leg
        axis, sgn = self.legs[i]
        if tl >= self.leg:
            # 구간 사이 정지 — 속도 0 을 명령(위치유지로 가면 원점으로 돌아가버린다)
            self.send_velocity(0.0, 0.0, 0.0, self.yaw0)
            self.set_stage(f'{i+1}/{len(self.legs)} {axis}{sgn:+d} PAUSE')
            return False
        ex, ey = self.body_dir(forward=(axis == 'fwd'))
        self.send_velocity(sgn * self.v * ex, sgn * self.v * ey, 0.0, self.yaw0)
        self.set_stage(f'{i+1}/{len(self.legs)} {axis}{sgn:+d} CRUISE {tl:.0f}/{self.leg:.0f}s')
        return False


def build(bench, outdir, a):
    F3Drag.NEED_ALT = a.need_alt
    dist = a.v * a.leg
    # ★ 안전 반경을 실제 이동거리에 맞춰 자동 확대.
    #   구간 하나가 v×leg 만큼 가는데 기본 25m 로 두면 도중에 중단된다.
    F3Drag.MAX_RADIUS = a.max_radius if a.max_radius > 0 else max(30.0, dist * 1.4)
    node = F3Drag(bench, outdir, a.v, a.leg, a.pause, a.settle)
    node.get_logger().info(
        f"  등속 {a.v} m/s × {a.leg}s = 구간당 {dist:.0f}m,  구간 {len(node.legs)}개\n"
        f"  총 {a.settle + len(node.legs)*node.T_leg:.0f}s,  안전반경 {F3Drag.MAX_RADIUS:.0f}m\n"
        f"\n"
        f"  ★ 비행 범위 (오프보드 켠 지점·기수 기준)\n"
        f"     기수 방향으로 {dist:.0f}m,  기수 기준 오른쪽으로 {dist:.0f}m\n"
        f"     → 그 방향으로 각각 {dist*1.3:.0f}m 이상 트여 있어야 합니다\n"
        f"     → 스위치를 켜기 전에 **기수를 원하는 전진 방향으로** 두세요\n"
        f"\n"
        f"  yaw 는 yaw0 로 고정 — 전후/좌우 항력을 분리하려면 필수")
    return node


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--v', type=float, default=3.0, help='등속 속도 [m/s]')
    ap.add_argument('--leg', type=float, default=8.0, help='한 구간 시간 [s]')
    ap.add_argument('--pause', type=float, default=3.0, help='구간 사이 정지 [s]')
    ap.add_argument('--settle', type=float, default=4.0)
    ap.add_argument('--need-alt', type=float, default=3.0,
                    help='진입 최소 고도 [m]. 실내 지상검증은 0 으로')
    ap.add_argument('--max-radius', type=float, default=0.0,
                    help='안전 반경 [m]. 0=자동 (구간거리×1.4, 최소 30)')
    ap.add_argument('--outdir', default='field_logs')
    a = ap.parse_args()
    run(lambda bench, outdir: build(bench, outdir, a), a.bench, a.outdir)
