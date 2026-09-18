#!/usr/bin/env python3
"""f5_pattern.py — 실기 기동 패턴 비행 (waypoint / circle / figure8)   [F5·F6·F7]

무엇을 얻나 (중요도 순)
  ① ★ **기동 중 실기 NIS 기준선** — 재정합 로드맵 [3] 의 필수 항목.
       지금 있는 건 호버 기준선뿐인데, 탐지 오탐은 기동 중에 난다.
  ② 모델 검증 영역 확대 — F1~F4 계수는 전부 호버 근방에서 뽑았다.
       기동 영역 예측잔차를 보면 sim 의 validate_by_regime.py 실기판이 된다.
  ③ drag 를 넓은 속도 범위에서 (F4 직선왕복보다 범위가 넓다)
  ④ RMSE 종단 검증 — 계수 하나하나가 아니라 사슬 전체를 본다

★ RMSE 비교가 유효한 근거 (2026-08-07 확인)
    실기  MC_ROLL_P 6.5 / MC_ROLLRATE_P 0.15 / MC_YAWRATE_P 0.2
    SITL  Iris 에어프레임에 MC 게인 오버라이드 없음 → PX4 기본값 = 동일
  실기가 오토튠 결과를 적용하지 않았으므로(MC_AT_APPLY=0) **양쪽 제어기가 같다.**
  → 추종 차이를 폐루프 튜닝이 아니라 **플랜트 차이로** 읽을 수 있다.

궤적은 sim(online_rl_main._compute_setpoint, 441행)을 그대로 이식했다.
명령 형식도 같다 — TrajectorySetpoint 에 위치+속도 피드포워드, OCM.position=True.
  ⚠ 비교가 성립하려면 **sim 도 같은 R·omega·고도**로 설정해야 한다.
    swrl_config.py: flight_radius / flight_omega / flight_altitude

기준점
  오프보드 진입 순간의 위치(origin)와 기수(yaw0)가 기준이다.
  패턴은 origin 을 원점으로, yaw0 방향을 +x 로 놓고 그린 뒤 NED 로 회전시킨다.
  → 조종사는 **트인 방향으로 기수를 두고** 스위치를 켜면 된다.

공간 (기본 R=3.0 m 기준, 8×8 m 안에 들어간다)
  circle   중심이 (-R,0) 이라 x 는 -2R~0, y 는 -R~R  →  6 × 6 m
  figure8  x 는 -R~R, y 는 -R/2~R/2                  →  6 × 3 m
  waypoint 사각 경로 x -0.6R~0.6R, y 0~0.6R          →  약 3.6 × 1.8 m
  ※ sim 기본값은 R=5.0 이다. 좁은 공간에서는 R 을 줄이되 **sim 도 같이 줄일 것.**

사용
    python3 f5_pattern.py --pattern circle              # F6
    python3 f5_pattern.py --pattern figure8             # F7
    python3 f5_pattern.py --pattern waypoint            # F5
    python3 f5_pattern.py --pattern circle --R 2.5 --laps 3
    python3 f5_pattern.py --pattern circle --bench      # 프로펠러 제거 지상검증

절차: 수동 이륙 → Position 모드 → 고도 안정 → **기수를 트인 방향으로** → 오프보드 ON
"""
import argparse
import math

from offboard_common import OffboardSequenceNode, run, add_postflight_args
from traj_common import TrajGen   # ★2026-09-10 sim _compute_setpoint 와 동치(check_traj_equiv.py 로 검증)

TWO_PI = 2.0 * math.pi


class F5Pattern(OffboardSequenceNode):
    SEQ_NAME = 'f5_pattern'
    CHECK_TYPE = 'pattern'   # 비행 후 check_ulog 가 자동으로 이 유형으로 판정
    NEED_ALT = 1.0

    GEN_PATTERNS = ('circle', 'figure8', 'waypoint', 'scurve', 'aggressive')   # TrajGen(sim 동치) 담당

    def __init__(self, bench=False, outdir='field_logs', pattern='circle',
                 R=3.4, omega=0.38, laps=2.0, settle=4.0, wp_speed=None, speed_mod=0.5,
                 agg_R=2.24, agg_phase=5.6):
        self.pattern = pattern
        # ★2026-09-10 sim frozen_v3 스펙 그대로: R 3.4·ω 0.38·속도변조 0.5/0.5Hz·waypoint 코너감속·scurve·aggressive Tp 4.7
        self.gen = TrajGen(pattern, R=R, omega=omega, agg_radius=agg_R, agg_phase_s=agg_phase,
                           speed_mod_amp=speed_mod) if pattern in self.GEN_PATTERNS else None
        self.R = float(R)
        self.w = float(omega)
        self.laps = float(laps)
        self.settle = float(settle)
        # waypoint 는 등속 직선 구간이라 속도를 따로 준다. 기본은 원과 같은 접선속도.
        self.wp_speed = float(wp_speed) if wp_speed else self.R * self.w
        self.no_ff = False
        self.yaw_amp = math.radians(90.0); self.yaw_T = 8.0      # F3-2
        self.acc_v = 2.0; self.acc_T = 10.0                      # F4-2
        self.agg_R = 2.24; self.agg_dz1 = 1.0; self.agg_dz2 = 0.7; self.agg_phase = 5.6   # ★2026-09-10 sim swrl_config.agg_radius/agg_phase_s 와 동일 (반원 접선 1.26 m/s, 변조 피크 1.88)
        # sim 의 waypoint 는 [[0,0],[5,0],[5,5],[-5,5],[-5,0],[0,0]] (R=5 기준).
        # R 로 스케일해 좁은 공간에 맞춘다. 비율은 sim 과 동일하게 유지.
        k = self.R / 5.0
        self.wps = [(0.0, 0.0), (5*k, 0.0), (5*k, 5*k), (-5*k, 5*k), (-5*k, 0.0), (0.0, 0.0)]
        self.seg_len = []
        for i in range(len(self.wps) - 1):
            dx = self.wps[i+1][0] - self.wps[i][0]
            dy = self.wps[i+1][1] - self.wps[i][1]
            self.seg_len.append(math.hypot(dx, dy))
        self.T_seg = [L / self.wp_speed for L in self.seg_len]
        self.T_wp = sum(self.T_seg)

        if self.gen is not None:
            self.T_run = self.laps * self.gen.lap_time()
        elif pattern == 'circle':
            self.T_run = self.laps * TWO_PI / self.w
        elif pattern == 'figure8':
            # x=R sin(wt), y=(R/2) sin(2wt) → 주기 2pi/w
            self.T_run = self.laps * TWO_PI / self.w
        elif pattern == 'yaw_spin':
            self.T_run = self.laps * 8.0
        elif pattern == 'accel_line':
            self.T_run = self.laps * 10.0
        elif pattern == 'aggressive':
            self.T_run = self.laps * 4 * self.agg_phase   # ★2026-09-10 agg_phase 연동 (구: 4×5.0 고정)
        else:
            self.T_run = self.laps * self.T_wp
        super().__init__(bench=bench, outdir=outdir)

    # ── 로컬(패턴) 좌표 → NED. origin 이동 + yaw0 회전 ──────────
    def _to_ned(self, xl, yl, vxl, vyl, yaw_l):
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        ox, oy, oz = self.origin
        x = ox + c * xl - s * yl
        y = oy + s * xl + c * yl
        vx = c * vxl - s * vyl
        vy = s * vxl + c * vyl
        return x, y, vx, vy, self.yaw0 + yaw_l

    def _local(self, t):
        """패턴 로컬 좌표. sim online_rl_main._compute_setpoint 와 동일한 식."""
        R, w = self.R, self.w
        if self.pattern == 'circle':
            th = w * t
            xl = R * math.cos(th) - R          # 중심 (-R,0): t=0 에서 원점 통과
            yl = R * math.sin(th)
            vxl = -R * w * math.sin(th)
            vyl = R * w * math.cos(th)
            yaw_l = (th + math.pi / 2 + math.pi) % TWO_PI - math.pi
            return xl, yl, vxl, vyl, yaw_l

        if self.pattern == 'figure8':
            # ★ |v|max = R*w*sqrt(2) 라 그대로 쓰면 실기 제한(MPC_XY_VEL_MAX=2.0)을 넘는다.
            #   반경만 1/sqrt(2) 배해 circle 과 같은 최대속도로 맞춘다.
            #   sim: swrl_config.fig8_radius_scale 과 동일해야 한다.
            R = R * 0.70710678
            xl = R * math.sin(w * t)
            yl = (R / 2.0) * math.sin(2 * w * t)
            vxl = R * w * math.cos(w * t)
            vyl = R * w * math.cos(2 * w * t)   # sim 과 동일(계수 생략도 그대로 이식)
            yaw_l = math.atan2(vyl, vxl) if (vxl or vyl) else 0.0
            return xl, yl, vxl, vyl, yaw_l

        if self.pattern == 'yaw_spin':
            # F3-2. 제자리에서 yaw 만 정현 왕복 → yaw 축 G 를 독립 경로로 검증.
            #   오토튠 yaw 가 maxVar 53.6 으로 미수렴이었다(2026-08-06).
            #   위치 이동이 없어 공간을 안 먹고 안전하다.
            yaw_l = self.yaw_amp * math.sin(TWO_PI * t / self.yaw_T)
            return 0.0, 0.0, 0.0, 0.0, yaw_l

        if self.pattern == 'accel_line':
            # F4-2. 기수 방향 직선을 정현 속도로 왕복 → drag 를 **속도의 함수**로 본다.
            #   F4-1(등속 왕복)은 단일 속도에서만 뽑으므로 drag 선형성을 확인할 수 없다.
            #   v(t)=V sin(2pi t/T) → x(t)=-(V T/2pi) cos(...) 로 유계.
            V, T = self.acc_v, self.acc_T
            A = V * T / TWO_PI
            xl = A * (1.0 - math.cos(TWO_PI * t / T))
            vxl = V * math.sin(TWO_PI * t / T)
            return xl, 0.0, vxl, 0.0, 0.0

        if self.pattern == 'aggressive':
            # sim online_rl_main._compute_setpoint 의 aggressive 를 **축소**해 이식.
            #   sim: 반경 4 m, 고도 ±3/±2 m, 5s/phase → 실기는 위험하므로 축소(기본 R=2).
            #   ★ 2026-08-07: sim 쪽에도 속도 FF 를 추가했으므로 여기도 FF 를 쓴다.
            #     (PX4 표준 — PositionControl.cpp:127-135, FlightTaskAuto.cpp:183)
            T = self.agg_phase
            ph = int(t // T) % 4
            f = (t % T) / T
            Ra, dz1, dz2 = self.agg_R, self.agg_dz1, self.agg_dz2
            k = math.pi / T
            # ★★ 2026-08-13: phase 경계 setpoint 불연속 제거 (sim 과 동일 수정).
            #   구: ph0=(0,0) / ph1 (Ra,0)→(−Ra,0) / ph2=(+Ra,0) → 경계마다 2.8·5.6 m 순간이동.
            #   08-12 실기 F10 의 RMSE 2.90 m·최대오차 6.76 m 가 이 아티팩트였다.
            #   신: 반원을 원점 기준으로 옮겨 ph0(0,0)→ph1→ph2(−2Ra,0)→ph3→(0,0) 로 닫는다.
            if ph == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0, -dz1*math.sin(math.pi*f), -dz1*k*math.cos(math.pi*f)
            if ph == 1:
                a = math.pi * f
                return (Ra*math.cos(a)-Ra, Ra*math.sin(a),
                        -Ra*k*math.sin(a), Ra*k*math.cos(a), a, 0.0, 0.0)
            if ph == 2:
                return (-2.0*Ra, 0.0, 0.0, 0.0, math.pi,
                        +dz2*math.sin(math.pi*f), dz2*k*math.cos(math.pi*f))
            a = math.pi * (1 - f)
            return (Ra*math.cos(a)-Ra, Ra*math.sin(a),
                    Ra*k*math.sin(a), -Ra*k*math.cos(a), a, 0.0, 0.0)

        # waypoint — 구간별 등속
        tm = t % self.T_wp
        i, acc = 0, 0.0
        for i in range(len(self.T_seg)):
            if tm < acc + self.T_seg[i]:
                break
            acc += self.T_seg[i]
        f = (tm - acc) / self.T_seg[i] if self.T_seg[i] > 0 else 0.0
        x0, y0 = self.wps[i]
        x1, y1 = self.wps[i + 1]
        dx, dy = x1 - x0, y1 - y0
        xl = x0 + dx * f
        yl = y0 + dy * f
        vxl = dx / self.T_seg[i]
        vyl = dy / self.T_seg[i]
        yaw_l = math.atan2(dy, dx)
        return xl, yl, vxl, vyl, yaw_l

    def step(self, t):
        if t < self.settle:
            self.hold_origin()
            self.set_stage(f'SETTLE {t:.0f}/{self.settle:.0f}s')
            return False
        te = t - self.settle
        if te >= self.T_run:
            self.hold_origin()
            self.set_stage('DONE_HOLD')
            return True
        if self.gen is not None:
            # ★2026-09-10 sim 동치 생성기: 매 발행 틱(1/PUB_HZ)마다 상태 전진. 속도+가속 FF 는 send_position_velocity 가 처리.
            from offboard_common import PUB_HZ
            xl, yl, dz, yaw_l, vxl, vyl, vz = self.gen.step(1.0 / PUB_HZ)
            x, y, vx, vy, yaw = self._to_ned(xl, yl, vxl, vyl, yaw_l)
            if self.no_ff:
                self.send_position(x, y, self.origin[2] + dz, yaw)
            else:
                self.send_position_velocity(x, y, self.origin[2] + dz, yaw, vx, vy, vz)
            frac = te / self.T_run * 100.0
            self.set_stage(f'{self.pattern.upper()} {te:.0f}/{self.T_run:.0f}s ({frac:.0f}%)')
            return False
        r = self._local(te)
        if self.pattern == 'aggressive':
            xl, yl, vxl, vyl, yaw_l, dz, vz = r
            x, y, vx, vy, yaw = self._to_ned(xl, yl, vxl, vyl, yaw_l)
            if self.no_ff:
                self.send_position(x, y, self.origin[2] + dz, yaw)
            else:
                self.send_position_velocity(x, y, self.origin[2] + dz, yaw, vx, vy, vz)
        else:
            xl, yl, vxl, vyl, yaw_l = r
            x, y, vx, vy, yaw = self._to_ned(xl, yl, vxl, vyl, yaw_l)
            if self.no_ff:
                self.send_position(x, y, self.origin[2], yaw)
            else:
                self.send_position_velocity(x, y, self.origin[2], yaw, vx, vy, 0.0)
        frac = te / self.T_run * 100.0
        self.set_stage(f'{self.pattern.upper()} {te:.0f}/{self.T_run:.0f}s ({frac:.0f}%)')
        return False


def build(bench, outdir, a):
    # ★ SEQ_NAME 은 __init__ 에서 CSV 경로에 쓰이므로 **생성 전에** 바꿔야 한다.
    #   패턴 → F번호 매핑 (2026-08-11): 자동회수 파일명을 F번호에 맞춘다.
    _FNUM = {'accel_line': 'f5', 'yaw_spin': 'f6', 'waypoint': 'f7',
             'circle': 'f8', 'figure8': 'f9', 'aggressive': 'f10'}
    F5Pattern.SEQ_NAME = f'{_FNUM.get(a.pattern, "f")}_{a.pattern}'
    F5Pattern.NEED_ALT = a.need_alt
    # 안전반경: circle 은 origin 에서 최대 2R 까지 나간다(중심이 -R). 여유 3 m.
    reach = {'yaw_spin': 0.5, 'accel_line': a.acc_v * a.acc_T / math.pi}.get(a.pattern)
    if reach is None:
        reach = TrajGen(a.pattern, R=a.R, omega=a.omega, agg_radius=a.agg_R, agg_phase_s=a.agg_phase).reach()
    F5Pattern.MAX_RADIUS = a.max_radius if a.max_radius > 0 else reach * 1.4 + 3.0
    node = F5Pattern(bench, outdir, a.pattern, a.R, a.omega, a.laps, a.settle, a.wp_speed,
                     speed_mod=(0.0 if a.no_speed_mod else a.speed_mod), agg_R=a.agg_R, agg_phase=a.agg_phase)
    node.bench_thrust = a.bench_thrust
    node.no_ff = a.no_ff; node.accel_ff = not a.no_ff   # ★2026-09-10 속도 FF 와 가속 FF 는 함께 (sim ACCEL_FF=1 과 동일)
    node.yaw_amp = math.radians(a.yaw_amp); node.yaw_T = a.yaw_T
    node.acc_v = a.acc_v; node.acc_T = a.acc_T
    node.agg_R = a.agg_R; node.agg_phase = a.agg_phase
    v = a.R * a.omega
    node.get_logger().info(
        f"\n  패턴 {a.pattern}   R={a.R} m   omega={a.omega} rad/s   접선속도 {v:.2f} m/s\n"
        f"  {a.laps} 바퀴 = {node.T_run:.0f}s  (+ settle {a.settle:.0f}s)\n"
        f"  안전반경 {F5Pattern.MAX_RADIUS:.1f} m,  origin 최대이탈 {reach:.1f} m\n"
        f"\n"
        f"  ★ 필요 공간 (오프보드 켠 지점·기수 기준)\n"
        f"     circle   기수 뒤쪽으로 {2*a.R:.1f} m, 좌우 각 {a.R:.1f} m\n"
        f"     figure8  기수 앞뒤 각 {a.R:.1f} m, 좌우 각 {a.R/2:.1f} m\n"
        f"     waypoint 기수 앞뒤 각 {0.6*a.R:.1f} m, 왼쪽 {0.6*a.R:.1f} m\n"
        f"     → 스위치를 켜기 전에 **기수를 트인 방향으로** 두세요\n"
        f"\n"
        f"  ⚠ sim 과 비교하려면 swrl_config.py 의 flight_radius={a.R},\n"
        f"    flight_omega={a.omega} 를 같은 값으로 맞출 것")
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='circle',
                    choices=['waypoint', 'circle', 'figure8',
                             'yaw_spin', 'accel_line', 'aggressive', 'scurve'])
    ap.add_argument('--speed-mod', dest='speed_mod', type=float, default=0.5,
                    help='속도변조 진폭 (sim SPEED_MOD_AMP=0.5, 0.5 Hz). waypoint 는 코너감속이 대체')
    ap.add_argument('--no-speed-mod', dest='no_speed_mod', action='store_true', help='속도변조 끔 (등속)')
    ap.add_argument('--no-ff', dest='no_ff', action='store_true',
                    help='속도 FF 없이 위치만 보낸다 (sim 도 같이 바꿔야 비교 성립)')
    ap.add_argument('--yaw-amp', dest='yaw_amp', type=float, default=90.0, help='yaw_spin 진폭 [deg]')
    ap.add_argument('--yaw-T', dest='yaw_T', type=float, default=8.0, help='yaw_spin 주기 [s]')
    ap.add_argument('--acc-v', dest='acc_v', type=float, default=1.75,
                    help='accel_line 최대속도 [m/s]. 실기 MPC_XY_VEL_MAX=2.0 대비 여유 12.5%%')
    ap.add_argument('--acc-T', dest='acc_T', type=float, default=10.0, help='accel_line 주기 [s]')
    ap.add_argument('--agg-R', dest='agg_R', type=float, default=2.24,
                    help='aggressive 반경 [m]. sim swrl_config.agg_radius 와 동일 (2.24)')
    ap.add_argument('--agg-phase', dest='agg_phase', type=float, default=5.6,
                    help='aggressive phase 길이 [s]. sim swrl_config.agg_phase_s 와 동일 (5.6 → 반원 접선 1.26 m/s, 속도변조 피크 1.88 < MPC_XY_VEL_MAX 2.0)')
    ap.add_argument('--R', type=float, default=3.4,
                    help='반경 [m]. sim swrl_config.flight_radius 와 같아야 한다 (3.4)')
    ap.add_argument('--omega', type=float, default=0.38, help='각속도 [rad/s]. sim swrl_config.flight_omega=0.38 (접선 1.29 m/s)')
    ap.add_argument('--laps', type=float, default=2.0, help='바퀴 수')
    ap.add_argument('--settle', type=float, default=4.0)
    ap.add_argument('--wp-speed', dest='wp_speed', type=float, default=None,
                    help='waypoint 등속 [m/s]. 생략하면 R*omega')
    ap.add_argument('--need-alt', dest='need_alt', type=float, default=1.0)
    ap.add_argument('--max-radius', dest='max_radius', type=float, default=0.0)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--bench-thrust', dest='bench_thrust', type=float, default=0.10)
    ap.add_argument('--outdir', default='field_logs')
    add_postflight_args(ap)
    a = ap.parse_args()
    run(lambda bench, outdir: build(bench, outdir, a), bench=a.bench, outdir=a.outdir, args=a)


if __name__ == '__main__':
    main()
