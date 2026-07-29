"""verify_ctorque_bounds.py — C_torque 축별 독립 검증 (2026-07-29)

사용: ~/isaacsim/python.sh verify_ctorque_bounds.py results_ctorque/rotor_log.npz

왜 별도 스크립트인가: fit_static_from_rotor.py 가 낸 C_torque_y=4.017 이
**기하학적 최대 피치 토크 2.555 N·m 를 157% 초과**한다. 같은 코드로 재적합하면
같은 오류를 반복할 수 있으므로, 독립 경로로 계산해 대조한다.

방법 (동역학 불사용 — 순수 정적 매핑):
  1. 실제 로터 각속도 ω_i 로 **적용된** 토크를 Pegasus 자신의 할당 규약으로 직접 계산
       τ_x(FLU) = Σ y_i·k·ω_i²      (multirotor.py:198)
       τ_y(FLU) = Σ (−x_i)·k·ω_i²   (multirotor.py:199)
       τ_z(FLU) = Σ c_roll·dir_i·ω_i²
     → 관성·각가속도가 전혀 안 들어가므로 I 오차·미분잡음·폐루프 편향에서 자유롭다.
  2. FLU → FRD 변환 후 PX4 정규화 명령에 회귀 → 기울기 = C_torque
  3. 기하학적 상한과 대조. 상한 초과 = 물리적으로 불가능한 값.
  4. 대각만 / 교차항 포함 두 가지로 적합해, 교차항이 대각을 오염시키는지 본다.
"""
import os, sys, json
import numpy as np

K_ROTOR = 8.54858e-6
C_ROLL_MOMENT = 1e-6
ROT_DIR = np.array([-1, -1, 1, 1])
W_MAX, W_MIN = 1100.0, 100.0
ROTOR_POS = np.array([                     # 바디(FLU) 로터 위치 [m] (iris.usd)
    [0.1379854530096054, -0.206716388463974, 0.023],
    [-0.1251116842031479, 0.2187541425228119, 0.023],
    [0.1379999965429306, 0.20257577300071716, 0.023],
    [-0.12415359914302826, -0.22234579920768738, 0.023],
])

npz = sys.argv[1] if len(sys.argv) > 1 else 'results_ctorque/rotor_log.npz'
d = np.load(npz, allow_pickle=True)
A = d['data']
cols = str(d['cols']).split(',')
c = {n.strip(): i for i, n in enumerate(cols)}

t = A[:, c['t']]
cmd_thr = A[:, [c['cmd_thr_x'], c['cmd_thr_y'], c['cmd_thr_z']]]
cmd_tq = A[:, [c['cmd_tq_x'], c['cmd_tq_y'], c['cmd_tq_z']]]
W = A[:, [c['w0'], c['w1'], c['w2'], c['w3']]]

print("=" * 80)
print(f"C_torque 축별 독립 검증 — {npz}")
print("=" * 80)
print(f"  샘플 {len(A)}개,  물리 dt = {float(d['dt'])*1000:.1f} ms")

# ── 1) 기하학적 상한 ─────────────────────────────────────
Tmax, Tmin = K_ROTOR * W_MAX**2, K_ROTOR * W_MIN**2
def geo_max(coef):
    return sum(Tmax * v if v > 0 else Tmin * v for v in coef)
bound = np.array([geo_max(ROTOR_POS[:, 1]),
                  geo_max(-ROTOR_POS[:, 0]),
                  geo_max(C_ROLL_MOMENT * ROT_DIR / K_ROTOR)])
print(f"\n[기하학적 상한] 전 추력을 한 축에 몰빵했을 때 (로터 최대추력 {Tmax:.3f} N)")
for nm, b in zip(('롤 τx', '피치 τy', '요 τz'), bound):
    print(f"  {nm}_max = {b:7.3f} N·m")
print(f"  롤/피치 비 = {bound[0]/bound[1]:.2f}  (암 길이 비 "
      f"{np.abs(ROTOR_POS[:,1]).mean()/np.abs(ROTOR_POS[:,0]).mean():.2f} 와 일치해야 정상)")

# ── 2) 실제 로터 속도 → 적용 토크 (FLU) → FRD ────────────
Ti = K_ROTOR * W**2                                   # (N,4) 로터별 추력
T_tot = Ti.sum(axis=1)
tau_flu = np.column_stack([
    Ti @ ROTOR_POS[:, 1],
    Ti @ (-ROTOR_POS[:, 0]),
    (C_ROLL_MOMENT * ROT_DIR * W**2).sum(axis=1),
])
tau_frd = tau_flu * np.array([1.0, -1.0, -1.0])       # FLU → FRD

# 시동 후 + 유효 추력 구간만
m = (T_tot > 0.5 * T_tot.max() * 0.3) & (W.min(axis=1) > W_MIN * 1.01)
print(f"\n  유효 샘플 {m.sum()}/{len(A)}  (시동 후 · 로터 포화 아님)")
cmd_tq, tau_frd, T_tot = cmd_tq[m], tau_frd[m], T_tot[m]

print(f"\n[명령 범위] PX4 정규화 토크 setpoint")
for j, nm in enumerate(('cmd_tq_x', 'cmd_tq_y', 'cmd_tq_z')):
    v = cmd_tq[:, j]
    print(f"  {nm}: 중앙 {np.median(v):+7.4f}  |.|95pct {np.percentile(np.abs(v),95):7.4f}  "
          f"|.|최대 {np.abs(v).max():7.4f}")

# ── 3) 회귀 ──────────────────────────────────────────────
AX = ('롤 x', '피치 y', '요 z')
print("\n" + "-" * 80)
print("[적합 A] 대각만:  τ_i = C_i·cmd_i + K_i·T")
print("-" * 80)
print(f"  {'축':>7s} {'C_torque':>10s} {'K(결합)':>12s} {'R²':>7s} {'상한':>9s} {'상한대비':>9s}")
diagC = []
for j in range(3):
    X = np.column_stack([cmd_tq[:, j], T_tot])
    y = tau_frd[:, j]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b)**2).sum() / ((y - y.mean())**2).sum()
    diagC.append(b[0])
    flag = ' ★초과' if b[0] > bound[j] else ''
    print(f"  {AX[j]:>7s} {b[0]:10.3f} {b[1]:12.6f} {r2:7.3f} {bound[j]:9.3f} "
          f"{b[0]/bound[j]*100:8.1f}%{flag}")

print("\n" + "-" * 80)
print("[적합 B] 교차항 포함:  τ_i = Σ_j C_ij·cmd_j + K_i·T")
print("-" * 80)
X = np.column_stack([cmd_tq[:, 0], cmd_tq[:, 1], cmd_tq[:, 2], T_tot])
print(f"  설계행렬 조건수 = {np.linalg.cond(X):.1f}   (>30 이면 축 분리 불량 = 대각계수 신뢰 저하)")
print(f"  {'축':>7s} {'C_i,x':>9s} {'C_i,y':>9s} {'C_i,z':>9s} {'K(결합)':>11s} {'R²':>7s}")
crossC = []
for j in range(3):
    y = tau_frd[:, j]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b)**2).sum() / ((y - y.mean())**2).sum()
    crossC.append(b[j])
    print(f"  {AX[j]:>7s} {b[0]:9.3f} {b[1]:9.3f} {b[2]:9.3f} {b[3]:11.6f} {r2:7.3f}")

# ── 4) 판정 ──────────────────────────────────────────────
cal = json.load(open('calibration/calibration.json'))
cur = [cal.get('C_torque_x', cal['C_torque_xy']),
       cal.get('C_torque_y', cal['C_torque_xy']),
       cal['C_torque_z']]
print("\n" + "=" * 80)
print("판정")
print("=" * 80)
print(f"  {'축':>7s} {'현재 calib':>11s} {'적합A(대각)':>12s} {'적합B(교차)':>12s} {'기하상한':>10s}")
for j in range(3):
    print(f"  {AX[j]:>7s} {cur[j]:11.3f} {diagC[j]:12.3f} {crossC[j]:12.3f} {bound[j]:10.3f}")
print()
bad = [AX[j] for j in range(3) if cur[j] > bound[j]]
if bad:
    print(f"  ★ 현재 calib 이 물리 상한을 초과하는 축: {', '.join(bad)}")
ok = [AX[j] for j in range(3) if abs(cur[j] - crossC[j]) / max(abs(crossC[j]), 1e-9) < 0.15]
print(f"  현재 calib 이 독립적합(교차)과 15% 내 일치하는 축: {', '.join(ok) if ok else '없음'}")
print("\n  ※ 이 검증은 관성·각가속도를 쓰지 않는다 → I 오차·미분잡음·폐루프 편향과 무관.")
print("     적합 A·B 가 갈리면 축 분리가 안 된 것(조건수 확인). 상한 초과는 데이터와 무관한 물리 위반.")
print("=" * 80)
