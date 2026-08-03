#!/usr/bin/env python3
"""f2_doublet.py — F2 자세 doublet (개루프 여기)

얻는 것: G_i = C_torque_i / I_i     (ω̇_i = G_i · cmd_torque_i)

★ 이 비행이 F2 인 이유
  자세 rate loop 는 폐루프라, 평범한 비행 로그로 ω̇ ~ τ_cmd 회귀를 돌리면
  계수가 심하게 편향된다 — sim 에서 14배(0.265 vs 3.57) 틀린 실증이 있고,
  실기 로그(test1)에서도 R² 0.02~0.09 로 전혀 안 맞았다.
  명령이 상태와 무관하게 흔들려야(개루프 여기) 한다.

★ 축을 섞지 않는다
  롤 N회 전부 → 피치 N회 → 요 N회.
  섞으면 회귀 설계행렬 조건수가 치솟아 대각계수를 분리할 수 없다
  (sim 에서 조건수 1412 로 롤/피치 분리에 실패한 전례).

사용:
    python3 f2_doublet.py               # 실비행 (고도 10m 이상에서 진입)
    python3 f2_doublet.py --bench       # 프로펠러 뺀 지상 검증
    python3 f2_doublet.py --amp 8 --n 8

절차: Position 모드로 수동 이륙 → 고도 10m 이상 안정 → 오프보드 스위치 ON
"""
import argparse
import math
from offboard_common import OffboardSequenceNode, run


class F2Doublet(OffboardSequenceNode):
    SEQ_NAME = 'f2_doublet'
    NEED_ALT = 8.0

    def __init__(self, bench=False, outdir='field_logs',
                 amp_deg=10.0, n=8, pulse=0.4, recover=3.0, settle=5.0, thrust=None):
        self.amp = math.radians(amp_deg)
        self.n = n
        self.pulse = pulse            # 한쪽 방향 유지 시간 [s]
        self.recover = recover        # doublet 사이 위치 회복 [s]
        self.settle = settle          # 시퀀스 시작 전 안정화 [s]
        self.thrust = thrust          # None 이면 진입 시점 자동 추정
        self._hover_thrust = 0.5
        self._axes = ('roll', 'pitch', 'yaw')
        # 한 doublet 주기 = 양방향 pulse + 회복
        self.T_one = 2 * self.pulse + self.recover
        self.T_axis = self.n * self.T_one
        super().__init__(bench=bench, outdir=outdir)

    def step(self, t):
        # ── 0) 안정화 ──
        if t < self.settle:
            self.hold_origin()
            self.set_stage(f'SETTLE {t:.0f}/{self.settle:.0f}s')
            return False

        te = t - self.settle
        ai = int(te // self.T_axis)
        if ai >= 3:
            self.hold_origin()
            self.set_stage('DONE_HOLD')
            return True

        axis = self._axes[ai]
        ta = te - ai * self.T_axis
        k = int(ta // self.T_one)                 # 몇 번째 doublet
        tp = ta - k * self.T_one                  # doublet 내부 시각

        if tp < self.pulse:
            sign, phase = +1.0, 'A'
        elif tp < 2 * self.pulse:
            sign, phase = -1.0, 'B'
        else:
            # 회복: 위치 유지 (attitude 모드에서 빠져나옴)
            self.hold_origin()
            self.set_stage(f'{axis} {k+1}/{self.n} RECOVER')
            return False

        r = p = 0.0
        y = self.yaw0
        if axis == 'roll':
            r = sign * self.amp
        elif axis == 'pitch':
            p = sign * self.amp
        else:
            y = self.yaw0 + sign * self.amp
        self.send_attitude(r, p, y, self._hover_thrust)
        self.set_stage(f'{axis} {k+1}/{self.n} PULSE{phase} {math.degrees(sign*self.amp):+.0f}°')
        return False

    def on_first_engage(self):
        pass


def build(bench, outdir, a):
    node = F2Doublet(bench, outdir, a.amp, a.n, a.pulse, a.recover, a.settle)
    node._hover_thrust = a.thrust
    node.get_logger().info(
        f"  doublet: 진폭 {a.amp}°, 축당 {a.n}회, 펄스 {a.pulse}s, 회복 {a.recover}s\n"
        f"  축당 {node.T_axis:.0f}s × 3축 + 안정화 {a.settle:.0f}s = 총 {3*node.T_axis+a.settle:.0f}s\n"
        f"  자세명령 중 추력 = {a.thrust:.2f} (고정)  ← 호버 스로틀과 비슷해야 고도가 유지된다")
    return node


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--amp', type=float, default=10.0, help='doublet 진폭 [deg]')
    ap.add_argument('--n', type=int, default=8, help='축당 반복 횟수')
    ap.add_argument('--pulse', type=float, default=0.4, help='한쪽 유지 [s]')
    ap.add_argument('--recover', type=float, default=3.0, help='doublet 사이 위치회복 [s]')
    ap.add_argument('--settle', type=float, default=5.0, help='시작 전 안정화 [s]')
    ap.add_argument('--thrust', type=float, default=0.35,
                    help='자세명령 중 고정 추력 (호버 스로틀 근처). test1 실측 호버 0.329')
    ap.add_argument('--outdir', default='field_logs')
    a = ap.parse_args()
    run(lambda bench, outdir: build(bench, outdir, a), a.bench, a.outdir)
