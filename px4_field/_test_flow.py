"""_test_flow.py — 실제 비행 흐름 전체를 모사해 setpoint 발행 시점을 검증.

시나리오:
  ① 스크립트 시작 (아직 지상, 조종기 Position)
  ② 수동 이륙 → 고도 안정          ← 여기서 setpoint 가 나가면 조종기와 충돌
  ③ 조종사가 오프보드 스위치 ON     ← intent 만 14, nav_state 는 아직 2
  ④ PX4 가 수락 → nav_state 14      ← origin 스냅샷, 시퀀스 시작
  ⑤ 조종사가 스위치 OFF             ← 즉시 SAFE, 발행 중단
"""
import types
import f1_hover
from offboard_common import OffboardSequenceNode as N

OFFB = 14
POS = 2

class Mock:
    def __init__(self, bench=False):
        o = f1_hover.F1Hover.__new__(f1_hover.F1Hover)
        o.dur = 5.0
        o.bench = bench
        o.bench_thrust = 0.10
        o._ocm_kind = 'attitude' if bench else 'position'
        o.stream_armed = False
        o.user_intent = POS
        o._intent_seen = False
        o.nav_state = POS
        o.arming = 2
        o.lp = None
        o.state = 'WAIT'
        o.origin = None
        o.yaw0 = 0.0
        o.t_engage = None
        o.stage = 'idle'
        o._last_stage = None
        o._warned = set()
        o.NEED_ALT = 1.5
        self.pub = []

        def _ocm(position=False, velocity=False, attitude=False):
            kind = ('position' if position else 'velocity' if velocity
                    else 'attitude' if attitude else 'none')
            o._ocm_kind = kind if kind != 'none' else o._ocm_kind
            self.pub.append('OCM:' + kind)
        o._ocm = _ocm
        o.pub_traj = types.SimpleNamespace(publish=lambda m: self.pub.append('TRAJ'))
        o.pub_att = types.SimpleNamespace(publish=lambda m: self.pub.append('ATT'))
        o._log_row = lambda: None
        o.get_logger = lambda: types.SimpleNamespace(
            info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
        for f in ('_may_stream', 'heading', 'warn_once', 'set_stage', 'hold_here',
                  'hold_origin', 'send_position', 'send_attitude', '_keepalive_ocm',
                  '_preflight_ok', '_bounds_ok', 'step'):
            setattr(o, f, types.MethodType(getattr(type(o), f, getattr(N, f)), o))
        o.set_stage = lambda n: setattr(o, 'stage', n)
        self.o = o
        self.t = 0.0

    def set_pose(self, alt, valid=True):
        from offboard_common import Pose
        p = Pose(); p.x = p.y = 0.0; p.z = -alt
        p.vx = p.vy = p.vz = 0.0; p.heading = 0.3
        p.valid = valid; p.src = 'odometry'
        self.o.lp = p

    def tick(self, n=1):
        self.pub.clear()
        for _ in range(n):
            N._tick(self.o)
        return list(self.pub)


def report(label, pub, expect_sp):
    has = any(x in ('TRAJ', 'ATT') for x in pub)
    ok = (has == expect_sp)
    mark = 'OK  ' if ok else 'FAIL'
    kinds = sorted(set(pub)) or ['없음']
    print(f"  [{mark}] {label:44s} 발행={','.join(kinds):12s} "
          f"(setpoint {'있어야' if expect_sp else '없어야'} 함)")
    return ok


print("=" * 78)
print("실제 비행 흐름 검증 — setpoint 발행 시점")
print("=" * 78)
m = Mock()
allok = True

allok &= report('① 시작 직후 (위치 수신 전, 지상)', m.tick(), False)

m.set_pose(0.0)
allok &= report('② 지상 대기 (조종기 Position)', m.tick(), False)

m.set_pose(2.2)
allok &= report('③ 수동 이륙·호버 중 (조종기 Position)', m.tick(3), False)

m.o.user_intent = OFFB          # 조종사가 스위치 ON — nav_state 는 아직 POS
p = m.tick()
allok &= report('④ 스위치 ON 직후 (intent=14, nav=2)', p, True)

m.o.nav_state = OFFB            # PX4 가 수락
p = m.tick()
allok &= report('⑤ 오프보드 진입 (nav=14)', p, True)
print(f"        → state={m.o.state}  origin={m.o.origin}  yaw0={m.o.yaw0:.3f}")

m.o.t_engage = __import__('time').time() - 1.0
p = m.tick(3)
allok &= report('⑥ 시퀀스 진행 중', p, True)
print(f"        → stage={m.o.stage}")

m.o.nav_state = POS; m.o.user_intent = POS      # 조종사 인계
p = m.tick()
allok &= report('⑦ 스위치 OFF (조종사 인계)', p, False)
print(f"        → state={m.o.state}  (WAIT 로 복귀해야 정상)")

p = m.tick(3)
allok &= report('⑧ 인계 후 수동비행 계속', p, False)


# ══════════════════════════════════════════════════════════════════════════
#  BENCH (실내·프로펠러 제거) — 오프보드 수락 조건 검증
#
#  PX4 offboardCheck.cpp: OffboardControlMode 에 position 을 선언했는데
#  local position 이 무효이면 오프보드를 **거부**한다 ("Offboard requires
#  local position"). 실내에서는 위치 추정이 무효이므로 bench 는 반드시
#  attitude 를 선언해야 스위치가 먹는다.
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 78)
print("BENCH 흐름 검증 — 실내(위치 추정 없음)에서 오프보드가 열리는가")
print("=" * 78)

b = Mock(bench=True)
b.o.NEED_ALT = 0.0

def report_ocm(label, pub, expect_kind, expect_sp):
    kinds = [x.split(':')[1] for x in pub if x.startswith('OCM:')]
    bad = [k for k in kinds if k != expect_kind]
    has = any(x in ('TRAJ', 'ATT') for x in pub)
    ok = (not bad) and bool(kinds) and (has == expect_sp)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:44s} OCM={','.join(sorted(set(kinds))) or '없음':10s} "
          f"setpoint={'있음' if has else '없음'}  (기대 OCM={expect_kind}, setpoint={'있음' if expect_sp else '없음'})")
    if bad:
        print(f"         ✗ position 선언이 섞이면 실내에서 오프보드가 거부된다: {sorted(set(bad))}")
    return ok

# lp 가 끝내 안 오는 상황(실내에서 흔함) — 그래도 OCM 은 계속 나가야 한다
allok &= report_ocm('⑨ 실내·위치 수신 없음 (lp=None)', b.tick(2), 'attitude', False)

b.o.user_intent = OFFB                          # 조종사가 스위치 ON
allok &= report_ocm('⑩ 스위치 ON (intent=14)', b.tick(), 'attitude', True)

b.o.nav_state = OFFB                            # PX4 수락
p = b.tick()
allok &= report_ocm('⑪ 오프보드 진입 (nav=14)', p, 'attitude', True)
print(f"        → state={b.o.state}  origin={b.o.origin}")

b.o.t_engage = __import__('time').time() - 1.0
allok &= report_ocm('⑫ 시퀀스 진행 (자세 명령)', b.tick(3), 'attitude', True)

b.o.nav_state = POS; b.o.user_intent = POS      # 인계
allok &= report_ocm('⑬ 스위치 OFF (인계)', b.tick(2), 'attitude', False)

print()
print("=" * 78)
print(f"  {'✓ 전부 통과 — 조종기 간섭 없음 + 실내 오프보드 진입 가능' if allok else '✗ 실패 항목 있음'}")
print("=" * 78)
