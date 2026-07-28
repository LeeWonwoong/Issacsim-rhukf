"""verify_yaw_wrap.py — 각도/wrap 정밀 검증 (2026-07-29)

검증 대상
  A. online_rl_main._quat_to_euler  : 왕복 정확도 / 출력 범위 / 브랜치컷 연속성
  B. CLAUDE.md 축약식 (phi, -th, 90-psi) 이 행렬식과 같은가 + wrap 필요 여부
  C. _hover_yaw 래치 → PX4 setpoint 범위
  D. 비행패턴 명령 yaw 범위 (circle 의 theta 무한누적 여부)
  E. UKF _f() 의 psi 적분 — wrap 없음이 안전한가 (wrapped/unwrapped 혼용 지점 탐색)
"""
import numpy as np
from scipy.spatial.transform import Rotation

RAD = np.pi / 180.0

# ── 검증 대상 함수 원본 복사 (online_rl_main.py:361) ──────────────────
def quat_to_euler(w, x, y, z):
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    T_i = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    T_b = np.diag([1.0, -1.0, -1.0])
    Rn = T_i @ R @ T_b
    return [float(np.arctan2(Rn[2, 1], Rn[2, 2])),
            float(-np.arcsin(np.clip(Rn[2, 0], -1.0, 1.0))),
            float(np.arctan2(Rn[1, 0], Rn[0, 0]))]


T_i = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
T_b = np.diag([1.0, -1.0, -1.0])


def ned_euler_to_isaac_quat(roll, pitch, yaw):
    """알려진 NED/FRD 자세 → Isaac 이 내보낼 ENU/FLU 쿼터니언 (w,x,y,z)."""
    R_ned = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    R_enu = T_i @ R_ned @ T_b            # T_i, T_b 는 involution (T²=I)
    q = Rotation.from_matrix(R_enu).as_quat()   # (x,y,z,w)
    return q[3], q[0], q[1], q[2]


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


print("=" * 72)
print("A. _quat_to_euler 왕복 정확도  (알려진 NED 자세 → ENU quat → 복원)")
print("=" * 72)
worst = 0.0
worst_case = None
rolls = np.arange(-60, 61, 15) * RAD
pitches = np.arange(-60, 61, 15) * RAD
yaws = np.concatenate([np.arange(-180, 181, 15), [179.9, -179.9, 180.0, -180.0]]) * RAD
n = 0
for r in rolls:
    for p in pitches:
        for y in yaws:
            q = ned_euler_to_isaac_quat(r, p, y)
            out = quat_to_euler(*q)
            err = np.array([wrap_pi(out[0] - r), wrap_pi(out[1] - p), wrap_pi(out[2] - y)])
            e = np.abs(err).max()
            n += 1
            if e > worst:
                worst, worst_case = e, (r/RAD, p/RAD, y/RAD, np.array(out)/RAD)
print(f"  검사 {n}개 조합, 최대 오차 = {worst:.3e} rad ({worst/RAD:.3e} deg)")
print(f"  최악 케이스 입력(deg) = {worst_case[:3]}  →  출력(deg) = {worst_case[3]}")
print(f"  판정: {'PASS' if worst < 1e-9 else 'FAIL'}")

print()
print("=" * 72)
print("B. 출력 범위 (wrap 이 되어 있는가)")
print("=" * 72)
outs = []
rng = np.random.default_rng(0)
for _ in range(20000):
    q = Rotation.random(random_state=None).as_quat()
    outs.append(quat_to_euler(q[3], q[0], q[1], q[2]))
outs = np.array(outs)
print(f"  roll  ∈ [{outs[:,0].min():+.6f}, {outs[:,0].max():+.6f}]   기대 [-pi, pi]     "
      f"= [{-np.pi:+.6f}, {np.pi:+.6f}]")
print(f"  pitch ∈ [{outs[:,1].min():+.6f}, {outs[:,1].max():+.6f}]   기대 [-pi/2, pi/2] "
      f"= [{-np.pi/2:+.6f}, {np.pi/2:+.6f}]")
print(f"  yaw   ∈ [{outs[:,2].min():+.6f}, {outs[:,2].max():+.6f}]   기대 [-pi, pi]")
ok = (np.abs(outs[:, 0]) <= np.pi + 1e-12).all() and \
     (np.abs(outs[:, 1]) <= np.pi/2 + 1e-12).all() and \
     (np.abs(outs[:, 2]) <= np.pi + 1e-12).all()
print(f"  판정: {'PASS — arctan2/arcsin 이므로 구조적으로 wrap 보장' if ok else 'FAIL'}")

print()
print("=" * 72)
print("C. 브랜치컷 연속성  (yaw 를 -200°→+200° 연속 스윕)")
print("=" * 72)
sweep = np.arange(-200, 200.5, 0.5) * RAD
psi_out = np.array([quat_to_euler(*ned_euler_to_isaac_quat(10*RAD, -5*RAD, y))[2] for y in sweep])
jumps = np.abs(np.diff(psi_out))
big = np.where(jumps > np.pi)[0]
print(f"  |Δpsi| > pi 인 지점 개수 = {len(big)}  (기대: ±pi 통과 시마다 1회씩)")
for i in big:
    print(f"    입력 yaw {sweep[i]/RAD:+8.2f}° → {sweep[i+1]/RAD:+8.2f}° 에서 "
          f"출력 {psi_out[i]/RAD:+8.2f}° → {psi_out[i+1]/RAD:+8.2f}°")
unwrapped = np.unwrap(psi_out)
resid = np.abs(np.diff(unwrapped - sweep)).max()
print(f"  unwrap 후 입력 대비 잔차 기울기 최대 = {resid:.3e}  "
      f"(0 이면 브랜치컷 외 불연속 없음)")
print(f"  판정: {'PASS — 정상적인 ±pi 컷만 존재' if resid < 1e-9 else 'FAIL — 추가 불연속'}")

print()
print("=" * 72)
print("D. CLAUDE.md 축약식 (phi, -th, 90°-psi) 검증")
print("=" * 72)
print("  ENU/FLU 쿼터니언에 '표준 ZYX' 를 적용한 각을 (phi_e, th_e, psi_e) 라 할 때")
print("  축약식이 행렬식과 일치하는가? 그리고 wrap 이 필요한가?")
bad_short = 0
out_of_range = 0
samples = []
for y_deg in np.arange(-180, 181, 5):
    q = ned_euler_to_isaac_quat(20*RAD, -10*RAD, y_deg*RAD)
    w, x, yq, z = q
    R = Rotation.from_quat([x, yq, z, w]).as_matrix()
    phi_e = np.arctan2(R[2, 1], R[2, 2])
    th_e = -np.arcsin(np.clip(R[2, 0], -1, 1))
    psi_e = np.arctan2(R[1, 0], R[0, 0])
    short_raw = np.array([phi_e, -th_e, np.pi/2 - psi_e])      # wrap 안 함
    ref = np.array(quat_to_euler(w, x, yq, z))
    if np.abs(wrap_pi(short_raw - ref)).max() > 1e-9:
        bad_short += 1
    if np.abs(short_raw[2]) > np.pi + 1e-12:
        out_of_range += 1
        if len(samples) < 3:
            samples.append((y_deg, short_raw[2], wrap_pi(short_raw[2]), ref[2]))
print(f"  축약식 ≡ 행렬식 (mod 2pi):  불일치 {bad_short}/73  "
      f"→ {'PASS' if bad_short == 0 else 'FAIL'}")
print(f"  축약식 raw 출력이 [-pi,pi] 를 벗어난 횟수: {out_of_range}/73")
for s in samples:
    print(f"    입력 yaw {s[0]:+5.0f}° : 축약식 raw = {s[1]/RAD:+8.2f}°  "
          f"wrap 후 = {s[2]/RAD:+8.2f}°  행렬식 = {s[3]/RAD:+8.2f}°")
print("  → 축약식을 코드로 쓸 거면 wrap_pi 필수. **현재 코드는 행렬식이라 구조적으로 안전.**")

print()
print("=" * 72)
print("E. 비행패턴 명령 yaw 범위  (step_dt=0.02, omega=0.5, 300 RL스텝=30s=1500 tick)")
print("=" * 72)
dt, w_om, R_r = 0.02, 0.5, 5.0
n_tick = 1500
theta = 0.0
circle_yaw = []
for k in range(n_tick):
    circle_yaw.append(theta + np.pi/2)
    theta += w_om * dt
circle_yaw = np.array(circle_yaw)
print(f"  circle   : yaw ∈ [{circle_yaw.min():+.3f}, {circle_yaw.max():+.3f}] rad "
      f"= [{circle_yaw.min()/RAD:+.1f}°, {circle_yaw.max()/RAD:+.1f}°]  "
      f"→ {'★ [-pi,pi] 벗어남 (무한누적)' if circle_yaw.max() > np.pi else 'OK'}")
t = np.arange(n_tick) * dt
vx = R_r*w_om*np.cos(w_om*t); vy = R_r*w_om*np.cos(2*w_om*t)
f8 = np.arctan2(vy, vx)
print(f"  figure8  : yaw ∈ [{f8.min():+.3f}, {f8.max():+.3f}] rad  → OK (arctan2)")
print(f"  waypoint : yaw = arctan2(dy,dx)                          → OK (arctan2)")
print(f"  aggressive: yaw ∈ [0, pi] (a=pi*f, pi, pi*(1-f))          → OK")
print(f"  hover    : yaw = 0.0                                     → OK")
print(f"  circle 최대치 {circle_yaw.max():.2f} rad = 회전 {circle_yaw.max()/(2*np.pi):.2f} 바퀴분")

print()
print("=" * 72)
print("F. UKF _f() 의 psi 적분 — wrap 없음이 안전한가")
print("=" * 72)
print("  ukf_filter._f: s[5] += (sp/ct*q + cp/ct*r)*sdt   ← wrap 없음, 무한누적")
r_yaw = 0.5
psi = 0.0
sdt = 0.005
for _ in range(int(30 / sdt)):
    psi += r_yaw * sdt
print(f"  요레이트 {r_yaw} rad/s 로 30s → UKF 상태 psi = {psi:.3f} rad "
      f"({psi/RAD:.1f}°), wrap 값 = {wrap_pi(psi)/RAD:+.1f}°")
print("  안전성 판정 기준: wrapped 값과 unwrapped 값이 '비교/차감'되는 지점이 있는가?")
print("   · _f 내부 사용처는 전부 cos/sin/tan (주기함수) → 무한누적 무해")
print("   · 시그마포인트 평균 Wm@pts_f 는 선형평균 — 모든 포인트가 같은 브랜치라 무해")
print("     (오히려 wrap 을 넣으면 ±pi 근처에서 시그마포인트가 갈라져 평균이 붕괴한다)")
print("   · 관측 z = [pos, vel, gyro] 에 각도 없음 → res 에 각도 차 없음")
print("   · 유일한 혼용 지점: online_rl_main:823  ukf.x[3:6] = cur_euler (wrapped)")
print("     → 에피소드당 1회 초기화 시점뿐 (is_ukf_initialized 게이트). 이후 재시드 없음.")
print("     → 초기화 시 psi=0 근방이므로 브랜치 문제 없음.")
print()
print("=" * 72)
