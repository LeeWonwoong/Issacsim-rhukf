"""시퀀스 타임라인 오프라인 검증 — ROS2 통신 없이 step() 만 돌린다."""
import math, sys, types
import f1_hover, f2_doublet, f3_drag
from offboard_common import wrap_pi

class Fake:
    """OffboardSequenceNode 를 상속하지 않고 step() 만 빌려 쓰기 위한 껍데기."""
    def __init__(self, cls, **kw):
        self.o = cls.__new__(cls)
        for k, v in kw.items(): setattr(self.o, k, v)
        self.o.yaw0 = math.radians(37.0)          # 임의의 진입 방향
        self.o.origin = (10.0, -5.0, -5.0)
        class LP: x,y,z,vx,vy,vz,heading = 10.0,-5.0,-4.7,0.0,0.0,0.3,0.0
        self.o.lp = LP()
        self.o.t_engage = 0.0
        self.o._last_stage = None
        self.stages, self.cmds = [], []
        self.o.hold_origin = lambda dz=0.0: self.cmds.append(('POS', self.o.origin))
        self.o.send_velocity = lambda vx,vy,vz,yaw: self.cmds.append(('VEL',(round(vx,3),round(vy,3),round(vz,3))))
        self.o.send_attitude = lambda r,p,y,th: self.cmds.append(('ATT',(round(math.degrees(r)),round(math.degrees(p)),round(math.degrees(wrap_pi(y-self.o.yaw0))))))
        self.o.body_dir = types.MethodType(type(self.o).body_dir, self.o)
        def set_stage(name):
            if name != self.o._last_stage:
                self.stages.append((round(self.t,2), name)); self.o._last_stage = name
            self.o.stage = name
        self.o.set_stage = set_stage
    def run(self, dt=0.05, tmax=600):
        self.t = 0.0
        while self.t < tmax:
            if self.o.step(self.t): return self.t
            self.t += dt
        return None

print("="*72); print("F1 HOVER  (dur=90)"); print("="*72)
f = Fake(f1_hover.F1Hover, dur=90.0); T = f.run()
print(f"  완료 시각 {T:.1f}s   (기대 90)")
print(f"  단계 수 {len(f.stages)},  명령 종류 {set(c[0] for c in f.cmds)}")

print(); print("="*72); print("F2 DOUBLET  (amp=10°, n=8, pulse=0.4, recover=3, settle=5)"); print("="*72)
f = Fake(f2_doublet.F2Doublet, amp=math.radians(10), n=8, pulse=0.4, recover=3.0,
         settle=5.0, _hover_thrust=0.35, _axes=('roll','pitch','yaw'),
         T_one=2*0.4+3.0, T_axis=8*(2*0.4+3.0))
T = f.run()
print(f"  완료 시각 {T:.1f}s   (기대 5 + 3×30.4 = 96.2)")
axes = {}
for tt, s in f.stages:
    a = s.split()[0]
    if a in ('roll','pitch','yaw'): axes.setdefault(a, []).append(s)
for a in ('roll','pitch','yaw'):
    pulses = [s for s in axes.get(a,[]) if 'PULSE' in s]
    print(f"  {a:6s}: 단계 {len(axes.get(a,[])):3d}개, PULSE {len(pulses)}회 (기대 16 = 8×2방향)")
att = [c for c in f.cmds if c[0]=='ATT']
print(f"  자세명령 표본 {len(att)}개, 고유 명령 {sorted(set(c[1] for c in att))}")
print("  ★ 축이 섞이지 않는지 확인:")
order = [s.split()[0] for tt,s in f.stages if s.split()[0] in ('roll','pitch','yaw')]
seq = [k for i,k in enumerate(order) if i==0 or k!=order[i-1]]
print(f"     축 등장 순서 = {seq}   (기대 ['roll','pitch','yaw'])")

print(); print("="*72); print("F3 DRAG  (v=3, leg=8, pause=3, settle=4)"); print("="*72)
legs = [('fwd',+1),('fwd',-1),('fwd',+1),('fwd',-1),('lat',+1),('lat',-1),('lat',+1),('lat',-1)]
f = Fake(f3_drag.F3Drag, v=3.0, leg=8.0, pause=3.0, settle=4.0, legs=legs, T_leg=11.0)
T = f.run()
print(f"  완료 시각 {T:.1f}s   (기대 4 + 8×11 = 92)")
vel = [c[1] for c in f.cmds if c[0]=='VEL']
uniq = sorted(set(vel))
print(f"  속도명령 고유값 {len(uniq)}개:")
yaw0 = math.radians(37.0)
for v in uniq:
    sp = math.hypot(v[0],v[1])
    ang = math.degrees(math.atan2(v[1],v[0])) if sp>0.01 else 0
    rel = math.degrees(wrap_pi(math.radians(ang)-yaw0)) if sp>0.01 else 0
    print(f"    NED{v}  속력 {sp:.2f} m/s  yaw0 기준 {rel:+.0f}°")
print("  ★ yaw0 기준 0°=전진, ±180°=후진, +90°=우횡진, -90°=좌횡진 이면 정상")
