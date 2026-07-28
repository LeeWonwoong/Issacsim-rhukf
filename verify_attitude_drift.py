"""verify_attitude_drift.py — 자세가 직접 관측되지 않는데 발산하는가? (2026-07-29)

세팅: 진짜 상태 = 완전 정지 호버(수평, 속도 0). 자이로 측정에 상수 바이어스 b 주입.
     GPS pos/vel 측정은 진값(0). 자세 관측 없음.
질문: UKF 의 롤 추정이 b·t 로 발산하는가, 아니면 어떤 값에 정착하는가?
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, load_calibration

RAD = np.pi / 180.0
calib = load_calibration('calibration.json')
m, g = calib['drone']['mass'], calib['drone']['g']
DT = 0.02

print("=" * 76)
print("자이로 바이어스 → 자세 오차: 발산하는가 정착하는가")
print("=" * 76)
print(f"  진값: 수평 정지 호버 (roll=pitch=0, vel=0).  UKF dt={DT}, 60s = 3000 step")
print(f"  vel 측정노이즈 R = {DynamicsUKF(dt=DT, calib=calib).R[3,3]}")
print()
print(f"  {'자이로 bias':>12s} {'개루프 예측':>12s} | {'실제 정착 롤':>12s} {'60s 시점':>10s} "
      f"{'vel잔차 RMS':>12s} {'판정':>8s}")

for b in [0.001, 0.005, 0.01, 0.03, 0.05]:
    ukf = DynamicsUKF(dt=DT, calib=calib)
    ukf.x[:] = 0.0
    u = np.array([m * g, 0.0, 0.0, 0.0])
    z = np.concatenate([[0, 0, 0], [0, 0, 0], [b, 0.0, 0.0]])
    roll_hist, res_hist = [], []
    for k in range(3000):
        res, _ = ukf.step(z, u)
        roll_hist.append(ukf.x[3])
        res_hist.append(res[3:6].copy())
    roll_hist = np.array(roll_hist)
    res_hist = np.array(res_hist)
    open_loop = b * 60.0                       # 보정 없으면 b·t
    settled = roll_hist[-500:].mean()          # 마지막 10s 평균
    at60 = roll_hist[-1]
    velrms = np.sqrt((res_hist[-500:] ** 2).sum(axis=1).mean())
    diverging = abs(at60) > abs(settled) * 1.5 + 1e-6
    print(f"  {b:10.3f}    {open_loop/RAD:9.1f}°   |  {settled/RAD:10.3f}°  {at60/RAD:8.3f}° "
          f"  {velrms:10.5f}   {'발산' if diverging else '정착'}")

print()
print("  → 개루프였다면 60s 에 b·60 rad 로 커져야 한다. 실제로는 훨씬 작은 값에 정착한다.")
print("    자세는 '추력방향 → 수평가속 → vel 잔차' 경로로 **간접 가관측**이기 때문.")
print("    즉 자이로 바이어스는 발산이 아니라 **정상상태 기울기 오차**를 만든다.")
print("    그 정상상태 오차가 nis_vel 의 바닥(floor)을 결정한다 → sim2real 에서 맞춰야 할 것.")

print()
print("=" * 76)
print("정착 자세오차가 만드는 nis_vel 바닥")
print("=" * 76)
print(f"  {'bias':>8s} {'정착 롤':>10s} {'수평가속 오차':>14s} {'vel잔차':>10s} {'R대비':>8s}")
for b in [0.001, 0.005, 0.01, 0.03, 0.05]:
    ukf = DynamicsUKF(dt=DT, calib=calib)
    ukf.x[:] = 0.0
    u = np.array([m * g, 0.0, 0.0, 0.0])
    z = np.concatenate([[0, 0, 0], [0, 0, 0], [b, 0.0, 0.0]])
    rh, rr = [], []
    for k in range(3000):
        res, _ = ukf.step(z, u)
        rh.append(ukf.x[3]); rr.append(res[3:6].copy())
    settled = np.array(rh)[-500:].mean()
    velrms = np.sqrt((np.array(rr)[-500:] ** 2).sum(axis=1).mean())
    da = g * np.tan(abs(settled))
    print(f"  {b:6.3f}   {settled/RAD:8.3f}°   {da:11.5f} m/s²  {velrms:9.5f}  "
          f"{velrms/np.sqrt(0.3):7.3f}σ")

print()
print("=" * 76)
print("자세 클립(0.8 rad)이 탐지 창 안에서 실제로 걸리는가")
print("=" * 76)
print("  탐지 창 = 온셋 후 d<=3 스텝(=0.3s @10Hz). 그 사이 기체가 45.8°를 넘는가?")
Ixx = calib['drone']['Ixx']
print(f"  Ixx = {Ixx:.6f} kg·m²")
print(f"  {'bias 토크':>10s} {'각가속도':>12s} {'0.3s 후 롤':>12s} {'1.0s 후':>10s} {'클립(45.8°) 도달시각':>20s}")
for s in [0.9, 1.1, 1.34, 1.5]:
    alpha = s / Ixx
    r03 = 0.5 * alpha * 0.3 ** 2
    r10 = 0.5 * alpha * 1.0 ** 2
    t_clip = np.sqrt(2 * 0.8 / alpha)
    print(f"  {s:8.2f} N·m {alpha:9.2f} rad/s² {r03/RAD:10.2f}° {r10/RAD:8.1f}° "
          f"{t_clip:16.3f} s ({t_clip*10:.1f} 스텝)")
print()
print("  ※ 무제어 자유적분 기준(PX4 가 저항하므로 실제는 이보다 훨씬 느리다) = **최악 상한**.")
print("    이 상한에서조차 클립 도달이 탐지 창(0.3s) 밖이면, 클립은 탐지에 개입하지 않는다.")
print("=" * 76)
