#!/usr/bin/env python3
"""rc_attack_trigger.py — 조종기(RC) 신호로 공격을 주입한다 [F9]   (2026-08-08)

무엇을 하나
  수동조종 비행 중, 조종기의 **스위치 + 노브**로 additive 공격 δ 를 켜고 크기를 조절한다.
  이 노드는 RC 상태를 읽어 **`/fmu/in/actuator_attack`(ActuatorAttack) 을 발행**한다 →
  uXRCE-DDS → PX4 SITL 의 ControlAllocator 가 c[0] 에 정규화 δ 가산 → Pegasus → Isaac.

  ★ 2026-08-08 변경: 외부 wrench(/attack_config→run_sim) 폐기. 이제 **실기와 동일 경로**
    (PX4 allocator)로 주입한다. 배분·포화·MOTOR_TAU·추력곡선 결합이 sim·실기에서 같아진다.
      실기 : RC → ControlAllocator 가 manual_control aux 를 직접 읽어 c[0] 가산
      sim  : RC → 이 노드 → DDS actuator_attack → 같은 ControlAllocator 가 c[0] 가산
    둘 다 **같은 apply_attack()**. 절차·mechanism 이 1:1.  근거: ATTACK_INJECTION.md §6-2.

  ⚠ δ 는 이제 **정규화값**(권한 대비 비율)이다. allocator 가 정규화 c[0] 를 쓰므로 환산 불필요.
    (구버전은 물리 N·m 로 환산했었다 — 외부 wrench 용. 폐기.)

  ⚠ PX4 SITL 이 ATK_EN=1 이어야 동작. 지상(param)에서 켤 것.

조종기 매핑 (실기와 동일하게)
  aux1 = enable 스위치   (2단 스위치.  up > 0.5 → 공격 무장)
  aux2 = 토크 크기 노브  (다이얼.  [-1,1] → [0,1] 로 매핑, full = 최대)
  aux3 = 추력 크기 노브
  ※ 실기 PX4 는 RC_MAP_AUX1/2/3 로 채널→aux 매핑. sim(QGC 조이스틱)도 조이스틱 축을
    aux 로 매핑하면 같다. 매핑이 안 되면 --stdin 으로 키보드 대체(아래).

크기 = "제어권한 대비 비율" (ATTACK_INJECTION.md §F9)
  실기 c[0] 은 정규화 토크/추력이라 δ 가 곧 권한의 몇 %. sim 은 물리단위(N·m,N)를 쓰므로
  여기서 비율 → 물리로 환산한다:  δ_Nm = knob · max_frac · 권한(N·m).
      토크 권한(롤) ≈ 4.36 N·m,  추력 권한 ≈ mg ≈ 13.15 N   (동결 플랜트 기준, override 가능)
  기본 max_frac = 0.08 = **권한의 8%** → "아주 약하지만 식별 가능"의 출발점.
  ⚠ 값 자체는 F9(가능성 테스트)에선 안 중요하다. 절차·경로가 도는지가 목적.

사용
    # 실기와 동일: RC aux 를 읽는다 (조이스틱 축이 aux 로 매핑돼 있을 때)
    python3 rc_attack_trigger.py

    # SITL 에서 aux 가 없을 때: 표준입력으로 조종 (SSH 로도 됨)
    python3 rc_attack_trigger.py --stdin
        t 0.5   → 토크 노브 50%     h 0.3 → 추력 노브 30%
        on / off → 스위치           q → 종료

    python3 rc_attack_trigger.py --axis pitch          # 토크 축 (roll/pitch/yaw)
    python3 rc_attack_trigger.py --tq-max 0.05 --th-max 0.05   # 더 약하게
"""
import argparse
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from px4_msgs.msg import ManualControlSetpoint, ActuatorAttack
except Exception:                                            # noqa: BLE001
    ManualControlSetpoint = None
    ActuatorAttack = None

_AXIS_IDX = {'roll': 0, 'pitch': 1, 'yaw': 2}                 # ActuatorAttack.torque 인덱스


class RCAttackTrigger(Node):
    def __init__(self, a):
        super().__init__('rc_attack_trigger')
        self.a = a
        self.axis_idx = _AXIS_IDX[a.axis]

        if ActuatorAttack is None:
            self.get_logger().error(
                "px4_msgs 에 ActuatorAttack 이 없습니다 — px4_msgs 재빌드 필요 "
                "(colcon build --packages-select px4_msgs)")
            raise SystemExit(1)

        q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1,
                       durability=DurabilityPolicy.VOLATILE)
        # /fmu/in/* 는 uXRCE-DDS 가 BEST_EFFORT 로 받는다
        self.pub = self.create_publisher(ActuatorAttack, '/fmu/in/actuator_attack', q)

        # 상태
        self.sw = False          # enable
        self.k_tq = 0.0          # 토크 노브 [0,1]
        self.k_th = 0.0          # 추력 노브 [0,1]
        self._last_sent = None

        if a.stdin:
            self.get_logger().info(
                "STDIN 모드. 명령:  on / off / t <0..1> / h <0..1> / q")
            threading.Thread(target=self._stdin_loop, daemon=True).start()
        else:
            if ManualControlSetpoint is None:
                self.get_logger().error("px4_msgs 없음 — --stdin 을 쓰세요")
                raise SystemExit(1)
            topic = self._resolve_mc_topic(a.mc_topic)
            self.get_logger().info(f"RC 구독: {topic}  (aux{a.aux_sw}=스위치, "
                                   f"aux{a.aux_tq}=토크, aux{a.aux_th}=추력)")
            self.create_subscription(ManualControlSetpoint, topic, self._cb_mc, q)

        self.create_timer(0.1, self._tick)      # 10Hz 로 상태 반영·발행
        self.get_logger().info(
            f"공격 준비(정규화=권한 비율): axis={a.axis}  "
            f"토크 full={a.tq_max:.3f} ({a.tq_max*100:.0f}% 권한), "
            f"추력 full={a.th_max:.3f} ({a.th_max*100:.0f}% 권한)\n"
            f"  ★ 스위치 up + 노브 올릴 때만 공격. 스위치 down = 즉시 0.\n"
            f"  ⚠ PX4 SITL 이 ATK_EN=1 이어야 δ 가 실제로 먹는다.")

    # ── RC 경로 ────────────────────────────────────────────
    def _resolve_mc_topic(self, override):
        if override:
            return override
        # PX4 v1.16+ 는 _v<N> 접미사가 붙는다. 그래프에서 실재하는 이름을 고른다.
        base = '/fmu/out/manual_control_setpoint'
        names = [n for n, _ in self.get_topic_names_and_types()]
        cands = [n for n in names
                 if n == base or n.startswith(base + '_v')]
        if cands:
            # 높은 버전 우선
            cands.sort(key=lambda n: -(int(n.split('_v')[-1]) if '_v' in n[len(base):] else 0))
            return cands[0]
        return base

    @staticmethod
    def _aux(msg, idx):
        return float(getattr(msg, f'aux{idx}', 0.0))

    def _cb_mc(self, msg):
        if not getattr(msg, 'valid', True):
            return
        self.sw = self._aux(msg, self.a.aux_sw) > 0.5
        # aux_tq>0: 노브 [-1,1]→[0,1] 로 크기 조절.  aux_tq=0: **고정 크기**(=tq_max),
        #   스위치(aux_sw=VRA)가 on/off 만 함. 당신 워크플로: VRA 올리면 argparse 크기로 공격.
        self.k_tq = (min(1.0, max(0.0, (self._aux(msg, self.a.aux_tq) + 1.0) * 0.5))
                     if self.a.aux_tq > 0 else 1.0)
        self.k_th = (min(1.0, max(0.0, (self._aux(msg, self.a.aux_th) + 1.0) * 0.5))
                     if self.a.aux_th > 0 else 1.0)

    # ── STDIN 경로 (SITL/aux 없을 때) ──────────────────────
    def _stdin_loop(self):
        for line in sys.stdin:
            p = line.split()
            if not p:
                continue
            c = p[0].lower()
            try:
                if c == 'on':
                    self.sw = True
                elif c == 'off':
                    self.sw = False
                elif c == 't':
                    self.k_tq = min(1.0, max(0.0, float(p[1])))
                elif c == 'h':
                    self.k_th = min(1.0, max(0.0, float(p[1])))
                elif c == 'q':
                    self.sw = False
                    self._publish(force=True)
                    rclpy.shutdown()
                    return
                print(f"  → sw={self.sw} k_tq={self.k_tq:.2f} k_th={self.k_th:.2f}")
            except (IndexError, ValueError):
                print("  형식: on / off / t <0..1> / h <0..1> / q")

    # ── 발행 ───────────────────────────────────────────────
    def _tick(self):
        self._publish()

    def _values(self):
        """(active, torque[3], thrust) — 전부 정규화값(권한 대비 비율).
        스위치 off 면 전부 0. active 는 실제 크기(>1e-4)로 판정 → tq_max/th_max=0 인 축은 자동 제외.
        """
        torque = [0.0, 0.0, 0.0]
        thrust = 0.0
        if self.sw:
            torque[self.axis_idx] = self.k_tq * self.a.tq_max     # 선택 축에만
            thrust = self.k_th * self.a.th_max
        active = abs(torque[self.axis_idx]) > 1e-4 or abs(thrust) > 1e-4
        return bool(active), torque, float(thrust)

    def _publish(self, force=False):
        active, torque, thrust = self._values()

        # ★ 매 tick(10Hz) 무조건 발행한다 (dedup 제거).
        #   /fmu/in/* 는 BEST_EFFORT 라 재전송이 없다. "값 바뀔 때만 1회" 발행하면
        #   그 샘플이 유실되면 PX4 가 영영 못 받는다. 오프보드 setpoint 처럼 계속 쏜다.
        msg = ActuatorAttack()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)  # PX4 clock us
        msg.active = active
        msg.torque = [float(v) for v in torque]
        msg.thrust = thrust
        self.pub.publish(msg)

        # 로그만 값 바뀔 때 (콘솔 도배 방지)
        key = (active, round(torque[0], 4), round(torque[1], 4),
               round(torque[2], 4), round(thrust, 4))
        if key != self._last_sent:
            self._last_sent = key
            if active:
                self.get_logger().info(
                    f"[공격 ON] τ=[{torque[0]:+.3f},{torque[1]:+.3f},{torque[2]:+.3f}] "
                    f"F={thrust:+.3f}  (정규화=권한비율)")
            else:
                self.get_logger().info("[공격 OFF]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', default='roll', choices=['roll', 'pitch', 'yaw'],
                    help='토크 공격 축 (기본 roll)')
    ap.add_argument('--tq-max', dest='tq_max', type=float, default=0.08,
                    help='토크 최대비율 (권한 대비, 기본 0.08=8%%)')
    ap.add_argument('--th-max', dest='th_max', type=float, default=0.08,
                    help='추력 최대비율 (권한 대비, 기본 0.08=8%%). 정규화값을 그대로 δ 로 쓴다')
    ap.add_argument('--aux-sw', dest='aux_sw', type=int, default=1, help='enable 스위치 aux')
    ap.add_argument('--aux-tq', dest='aux_tq', type=int, default=2, help='토크 노브 aux (0=off)')
    ap.add_argument('--aux-th', dest='aux_th', type=int, default=3, help='추력 노브 aux (0=off)')
    ap.add_argument('--mc-topic', dest='mc_topic', default=None,
                    help='manual_control_setpoint 토픽 (생략시 자동감지)')
    ap.add_argument('--stdin', action='store_true', help='RC 대신 표준입력으로 조종')
    a = ap.parse_args()

    rclpy.init()
    node = RCAttackTrigger(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 반드시 공격 끔
        try:
            node.sw = False
            node._publish(force=True)
        except Exception:                                    # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
