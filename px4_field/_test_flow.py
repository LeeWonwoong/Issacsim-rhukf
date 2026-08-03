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
    def __init__(self):
        o = f1_hover.F1Hover.__new__(f1_hover.F1Hover)
        o.dur = 5.0
        o.bench = False
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
        o._ocm = lambda **k: self.pub.append('OCM')
        o.pub_traj = types.SimpleNamespace(publish=lambda m: self.pub.append('TRAJ'))
        o.pub_att = types.SimpleNamespace(publish=lambda m: self.pub.append('ATT'))
        o._log_row = lambda: None
        o.get_logger = lambda: types.SimpleNamespace(
            info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
        for f in ('_may_stream', 'heading', 'warn_once', 'set_stage', 'hold_here',
                  'hold_origin', 'send_position', '_preflight_ok', '_bounds_ok', 'step'):
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

print()
print("=" * 78)
print(f"  {'✓ 전부 통과 — 조종기 간섭 없음' if allok else '✗ 실패 항목 있음'}")
print("=" * 78)
