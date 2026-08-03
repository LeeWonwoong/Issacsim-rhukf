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

★ 기본값은 좁은 공간(5m) 기준이다
    v=1.5 m/s, leg=3s, reps=4  →  구간 4.5m, 등속 2.0s, 축당 표본 320
    같은 공간에서 속도를 낮추면 항력 신호(기울기)가 작아지지만
    --reps 로 왕복을 늘려 표본을 벌면 상쇄된다.
    실측 기준 SNR 59.0 — 넓은 공간 설정(v2/leg4/reps2, 10m)의 64.6 대비 91%.

사용:
    python3 f3_drag.py                            # 5m 공간 기본
    python3 f3_drag.py --v 2 --leg 4 --reps 2     # 10m 공간
    python3 f3_drag.py --bench                    # 프로펠러 뺀 지상 검증

절차: Position 모드로 수동 이륙 → 고도 2~2.5m 안정 → 기수 정렬 → 오프보드 ON
필요 공간: 기수 방향 v×leg×1.3, 기수 기준 오른쪽 같은 거리
"""
import argparse
from offboard_common import OffboardSequenceNode, run


class F3Drag(OffboardSequenceNode):
    SEQ_NAME = 'f3_drag'
    NEED_ALT = 3.0

    def __init__(self, bench=False, outdir='field_logs', v=3.0, leg=8.0, pause=3.0,
                 settle=4.0, reps=2):
        self.v, self.leg, self.pause, self.settle = v, leg, pause, settle
        # 구간: (축, 부호). 축마다 reps 회 왕복.
        #   공간이 좁으면 구간을 짧게 하고 reps 를 늘려 표본을 보충한다.
        self.legs = ([('fwd', +1), ('fwd', -1)] * reps +
                     [('lat', +1), ('lat', -1)] * reps)
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
    #  구간거리에 비례하되 여유는 고정 3m — 좁은 공간에서도 실제 보호가 되도록.
    F3Drag.MAX_RADIUS = a.max_radius if a.max_radius > 0 else dist * 1.5 + 3.0
    node = F3Drag(bench, outdir, a.v, a.leg, a.pause, a.settle, a.reps)
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
    t_acc = a.v / 3.0                      # MPC_ACC_HOR 기본 ~3 m/s^2 가정
    t_steady = a.leg - 2 * t_acc
    if t_steady < 1.5:
        node.get_logger().warn(
            f"  ⚠ 등속 구간이 {t_steady:.1f}s 뿐입니다 (가속·감속에 {2*t_acc:.1f}s 소모).\n"
            f"    --leg 를 늘리거나 --v 를 낮추세요. 항력은 등속에서만 보입니다.")
    else:
        node.get_logger().info(f"  구간당 등속 시간 약 {t_steady:.1f}s (가속·감속 제외)")
    return node


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--v', type=float, default=1.5, help='등속 속도 [m/s]')
    ap.add_argument('--leg', type=float, default=3.0, help='한 구간 시간 [s]')
    ap.add_argument('--pause', type=float, default=2.0, help='구간 사이 정지 [s]')
    ap.add_argument('--settle', type=float, default=4.0)
    ap.add_argument('--reps', type=int, default=4,
                    help='축당 왕복 횟수. 공간이 좁으면 구간을 줄이고 이걸 늘린다')
    ap.add_argument('--need-alt', type=float, default=1.5,
                    help='진입 최소 고도 [m]. 저고도(2m) 운용 기준. 실내 지상검증은 0')
    ap.add_argument('--max-radius', type=float, default=0.0,
                    help='안전 반경 [m]. 0=자동 (구간거리×1.4, 최소 30)')
    ap.add_argument('--outdir', default='field_logs')
    a = ap.parse_args()
    run(lambda bench, outdir: build(bench, outdir, a), a.bench, a.outdir)
