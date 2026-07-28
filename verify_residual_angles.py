"""verify_residual_angles.py — UKF 잔차 경로의 각도/각속도 정밀 감사 (2026-07-29)

질문: Isaac Sim → PX4/Pegasus → UKF 잔차 계산에서 각도·각속도가 제대로 다뤄지는가?
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, load_calibration

RAD = np.pi / 180.0
calib = load_calibration('calibration.json')

print("=" * 78)
print("1. 잔차 경로 프레임 감사  (소스별 규약 추적)")
print("=" * 78)
rows = [
    ("z[0:3] gps_pos", "/fmu/out/vehicle_gps_position", "PX4", "lat/lon/alt",
     "online:301  ENU 조립 → :817 [1],[0],-[2] → NED", "OK"),
    ("z[3:6] gps_vel", "  동일 (vel_n/e/d_m_s)", "PX4", "NED",
     "online:305  ENU 로 재배열 → :818 다시 NED", "OK (왕복)"),
    ("z[6:9] gyro", "/fmu/out/sensor_combined.gyro_rad", "PX4", "**FRD 바디**",
     "online:308  변환 없이 그대로 z 에 투입", "OK"),
    ("u[1:4] torque", "/fmu/out/vehicle_torque_setpoint", "PX4", "**FRD 정규화**",
     "to_physical_u  ×C_torque_x/y/z (FRD 로 적합됨)", "OK"),
    ("u[0] thrust", "/fmu/out/vehicle_thrust_setpoint", "PX4", "FRD z(아래+)",
     "to_physical_u  |thrust[2]|×C_thrust", "OK"),
    ("x[3:6] euler(초기값)", "/gt/odometry", "**Isaac GT**", "**ENU/FLU**",
     "_quat_to_euler 로 NED/FRD 변환 (2026-07-28 수정)", "OK"),
]
print(f"  {'항목':22s} {'소스':38s} {'출처':8s} {'원 규약':14s} {'처리':46s} 판정")
for r in rows:
    print(f"  {r[0]:22s} {r[1]:38s} {r[2]:8s} {r[3]:14s} {r[4]:46s} {r[5]}")
print()
print("  ★ 핵심: 각도만 Isaac GT(ENU/FLU) 출처이고, 각속도·토크·추력·GPS 는 전부 PX4(NED/FRD) 출처다.")
print("    → 그래서 2026-07-28 프레임 버그가 **각도에서만** 났고 각속도는 멀쩡했던 것. 구조적으로 일관.")
print()
print("  ★ 잔차 res = z - z_bar 의 구성: [pos(3), vel(3), gyro(3)]  — **각도 항이 없다.**")
print("    → 각도 wrap 은 잔차에 직접 영향을 줄 수 없다. nis_gyro 는 '각속도 차'라 wrap 무관.")

print()
print("=" * 78)
print("2. 각도는 관측되지 않는다 — 칼만이득 K 의 오일러 행 크기")
print("=" * 78)
ukf = DynamicsUKF(dt=0.02, calib=calib)
ukf.x[:] = 0.0
ukf.x[6:9] = [1.0, 0.5, 0.0]
u_hover = np.array([calib['drone']['mass'] * calib['drone']['g'], 0.0, 0.0, 0.0])
z = np.concatenate([[0, 0, -5.0], [1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
for _ in range(200):                       # P 를 정상상태로
    ukf.step(z, u_hover)

n = ukf.nx
ukf.P = 0.5 * (ukf.P + ukf.P.T)
S_root = np.linalg.cholesky((n + ukf.lam) * ukf.P + 1e-7 * np.eye(n))
pts = np.vstack([ukf.x, ukf.x + S_root.T, ukf.x - S_root.T])
pts_f = np.array([ukf._f(p, u_hover) for p in pts])
x_bar = ukf.Wm @ pts_f
z_pts = np.array([ukf._h(p) for p in pts_f])
z_bar = ukf.Wm @ z_pts
Pzz = ukf.R + sum(ukf.Wc[i] * np.outer(z_pts[i] - z_bar, z_pts[i] - z_bar) for i in range(2 * n + 1))
Pxz = sum(ukf.Wc[i] * np.outer(pts_f[i] - x_bar, z_pts[i] - z_bar) for i in range(2 * n + 1))
K = Pxz @ np.linalg.inv(Pzz)

names = ['pos_N', 'pos_E', 'pos_D', 'phi', 'theta', 'psi', 'vel_N', 'vel_E', 'vel_D', 'p', 'q', 'r']
print("  상태별 |K| 행합 (잔차 1 단위가 그 상태를 얼마나 움직이나)")
for i, nm in enumerate(names):
    bar = '#' * int(min(40, np.abs(K[i]).sum() * 40))
    mark = '  ← 각도' if i in (3, 4, 5) else ''
    print(f"    {nm:6s} {np.abs(K[i]).sum():8.5f}  {bar}{mark}")
eul_gain = np.abs(K[3:6]).sum()
gyr_gain = np.abs(K[9:12]).sum()
print(f"\n  각도 3축 합계 = {eul_gain:.5f}   각속도 3축 합계 = {gyr_gain:.5f}"
      f"   비율 = {eul_gain/gyr_gain:.4f}")
print("  → 각도는 pos/vel 잔차와의 상관(Pxz)으로만 간접 보정된다. 직접 관측 없음.")
print("  → 단 '개루프 적분'은 아니다: '추력방향 → 수평가속 → vel 잔차' 경로로 **간접 가관측**이라")
print("    자이로 바이어스는 발산하지 않고 정상상태 기울기 오차로 정착한다.")
print("    정착값과 nis_vel 바닥 기여는 verify_attitude_drift.py 참조 (bias 0.05 에서도 0.053σ).")

print()
print("=" * 78)
print("3. 자세 오차가 잔차에 얼마나 새는가 (추력벡터 회전 경로)")
print("=" * 78)
T_h = calib['drone']['mass'] * calib['drone']['g']
print(f"  호버 추력 T = {T_h:.3f} N,  질량 m = {calib['drone']['mass']} kg")
print(f"  {'자세오차':>10s} {'수평 추력오차':>14s} {'가속도오차':>12s} {'1초 후 속도오차':>16s}")
for e_deg in [0.5, 1, 2, 5, 10]:
    dF = T_h * np.sin(e_deg * RAD)
    da = dF / calib['drone']['mass']
    print(f"  {e_deg:8.1f}°  {dF:12.4f} N  {da:10.4f} m/s²  {da*1.0:14.4f} m/s")
print(f"  ※ vel 측정 노이즈 R = {ukf.R[3,3]}  →  롤/피치 오차 1~2° 면 이미 vel 잔차와 같은 크기.")
print("  ※ yaw 오차는 추력을 기울이지 않으므로(수직축) 훨씬 덜 새고, 항력 방향에만 영향.")

print()
print("=" * 78)
print("4. ★ _f() 의 자세 클립 — 공격 영역에서 모델이 포화한다")
print("=" * 78)
print("  ukf_filter._f:  limit = 0.8;  phi = np.clip(phi, -limit, limit)   (매 서브스텝, 상태에 대입)")
print(f"  클립 한계 0.8 rad = {0.8/RAD:.1f}°   vs   crash_flip 판정 1.05 rad = {1.05/RAD:.1f}°")
print(f"  → 즉 {0.8/RAD:.1f}°~{1.05/RAD:.1f}° 구간에서 UKF 는 실제 자세를 **따라갈 수 없다**(상태가 상한에 붙음).")
print()
print(f"  {'실제 롤':>8s} {'모델 롤':>8s} {'수평추력 실제':>14s} {'모델':>10s} {'오차':>10s} {'가속도오차':>12s}")
for r_deg in [30, 40, 45.8, 50, 55, 60]:
    r = r_deg * RAD
    r_m = np.clip(r, -0.8, 0.8)
    Fh, Fh_m = T_h * np.sin(r), T_h * np.sin(r_m)
    print(f"  {r_deg:6.1f}°  {r_m/RAD:6.1f}°  {Fh:12.4f} N {Fh_m:9.4f} N {Fh-Fh_m:9.4f} N"
          f" {(Fh-Fh_m)/calib['drone']['mass']:10.4f} m/s²")
print("  → 공격이 성공해 기체가 크게 기울수록 모델오차가 **인위적으로** 커진다.")
print("    탐지에는 유리하게 작용하지만, 그 신호의 일부는 물리가 아니라 클립 아티팩트다.")
print("    (실기에서도 같은 클립이 걸리면 동일하게 재현되므로 sim2real 불일치는 아님)")

print()
print("=" * 78)
print("5. wrap 관련 최종 판정")
print("=" * 78)
verdict = [
    ("_quat_to_euler 출력", "arctan2/arcsin → 구조적 wrap 보장", "OK"),
    ("UKF 상태 psi (x[5])", "무한누적, wrap 없음 — 그러나 사용처가 전부 cos/sin/tan", "OK (의도적)"),
    ("시그마포인트 평균", "모든 포인트 같은 브랜치 → 선형평균 유효. wrap 넣으면 오히려 붕괴", "OK"),
    ("잔차 res", "각도 항 없음 (pos/vel/gyro) → wrap 무관", "OK"),
    ("nis_gyro", "각속도 차 — 각도가 아니므로 wrap 불필요", "OK"),
    ("_hover_yaw 래치", "cur_euler[2] ∈ [-pi,pi] → PX4 setpoint", "OK"),
    ("verify_calibration.check_yaw", "np.unwrap 사용 (84.5° 측정 유효)", "OK"),
    ("circle 패턴 명령 yaw", "theta 무한누적 → 30s 에 +948° (2.6바퀴). 타 패턴은 arctan2 로 wrap", "★ 불일치"),
]
for v in verdict:
    print(f"  [{v[2]:>10s}] {v[0]:32s} {v[1]}")
print()
print("=" * 78)
