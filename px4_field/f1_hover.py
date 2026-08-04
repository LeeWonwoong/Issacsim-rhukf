#!/usr/bin/env python3
"""f1_hover.py — F1 정지 호버 (실기 계수 수집)

얻는 것: C_thrust · 결합항 k_norm · 자이로 σ · GPS 속도 σ
  정지 호버는 참 속도 ≈ 0 이므로 GPS 속도가 곧 GPS 노이즈다 (RTK 불필요).

사용:
    python3 f1_hover.py                 # 실비행
    python3 f1_hover.py --bench         # 프로펠러 뺀 지상 검증
    python3 f1_hover.py --dur 120       # 호버 길이 지정 (GPS 표본 ↑)

절차: Position 모드로 수동 이륙 → 고도 4~5m 안정 → 오프보드 스위치 ON
"""
import argparse
from offboard_common import OffboardSequenceNode, run


class F1Hover(OffboardSequenceNode):
    SEQ_NAME = 'f1_hover'
    NEED_ALT = 3.0

    def __init__(self, bench=False, outdir='field_logs', dur=90.0):
        self.dur = dur
        super().__init__(bench=bench, outdir=outdir)

    def step(self, t):
        # 진입 위치·방향을 그대로 유지하며 dur 초 카운트
        self.hold_origin()
        self.set_stage(f'HOVER {int(t)}/{int(self.dur)}s')
        return t >= self.dur


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true', help='지상 검증(사전조건 완화)')
    ap.add_argument('--dur', type=float, default=90.0, help='호버 유지 [s] (권장 90~120)')
    ap.add_argument('--need-alt', type=float, default=1.5,
                    help='진입 최소 고도 [m]. 저고도(2m) 운용 기준. 실내 지상검증은 0')
    ap.add_argument('--bench-thrust', type=float, default=0.10,
                    help='bench 모드 추력 0~1. 0.10 은 아이들에 가까워 반응이 안 보인다. '
                         '모터 반응을 보려면 0.20~0.25 (프로펠러 제거 상태에서만!)')
    ap.add_argument('--outdir', default='field_logs')
    a = ap.parse_args()

    def build(bench, outdir):
        F1Hover.NEED_ALT = a.need_alt
        node = F1Hover(bench, outdir, a.dur)
        node.bench_thrust = a.bench_thrust
        return node
    run(build, a.bench, a.outdir)
