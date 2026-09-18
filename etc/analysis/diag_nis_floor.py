"""diag_nis_floor.py — 평시 NIS 바닥의 축별 원인 분해 (2026-07-29)

사용: ~/isaacsim/python.sh diag_nis_floor.py results_p0_tau0

zu_log 의 (z, u) 를 그대로 UKF 에 다시 먹여 잔차를 축별로 분해한다.
질문: 평시(정지호버)인데 nis_g_raw 가 왜 1 보다 훨씬 큰가?
  - 측정 자이로는 ~0.03 rad/s 로 조용하다 → 잔차가 크다면 **모델 예측**이 튄다는 뜻.
  - 예측을 튀게 하는 경로는 u[1:4]/I (명령토크 × C_torque / 관성) 뿐이다.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, load_calibration

outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_p0_tau0'
calib = load_calibration('calibration.json')

d = np.load(os.path.join(outdir, 'zu_log.npz'), allow_pickle=True)
A = d['data']
cols = [s.strip() for s in str(d['cols']).replace(',', ' ').split()]
c = {n: i for i, n in enumerate(cols)}
sm = list(csv.DictReader(open(os.path.join(outdir, 'sweep_summary.csv'))))
bnd = np.where(A[:, c['reset']] == 1)[0]
starts = list(bnd) if (len(bnd) and bnd[0] == 0) else [0] + list(bnd)
segs = [(s, starts[k + 1] if k + 1 < len(starts) else len(A)) for k, s in enumerate(starts)]

ZC = [c[k] for k in ('z0_gpsN', 'z1_gpsE', 'z2_gpsD', 'z3_velN', 'z4_velE', 'z5_velD',
                     'z6_gyrx', 'z7_gyry', 'z8_gyrz')]
UC = [c[k] for k in ('u0_thrust', 'u1_tx', 'u2_ty', 'u3_tz')]

# hover 패턴 세그먼트만 (정지호버 = 최저 여기)
pick = [k for k in range(min(len(segs), len(sm))) if sm[k]['pattern'] == 'hover']
print("=" * 76)
print(f"평시 NIS 바닥 분해 — {outdir}  (hover 패턴 {len(pick)} 에피소드)")
print("=" * 76)

res_all, pzz_all, nis_all, pred_all, meas_all = [], [], [], [], []
for k in pick:
    s, e = segs[k]
    ukf = DynamicsUKF(dt=0.02, calib=calib)
    ukf.x[0:3] = A[s, ZC[0:3]]
    ukf.x[6:9] = A[s, ZC[3:6]]
    ukf.x[9:12] = A[s, ZC[6:9]]
    for i in range(s + 1, e):
        z = A[i, ZC]
        u = A[i, UC]
        xb = ukf.x.copy()
        res, Pzz = ukf.step(z, u)
        res_all.append(res)
        pzz_all.append(np.diag(Pzz))
        g = res[6:9]
        nis_all.append(g @ np.linalg.solve(Pzz[6:9, 6:9], g) / 3.0)
        meas_all.append(z[6:9])
        pred_all.append(z[6:9] - res[6:9])      # z_bar = z - res
res_all = np.array(res_all); pzz_all = np.array(pzz_all)
nis_all = np.array(nis_all); pred_all = np.array(pred_all); meas_all = np.array(meas_all)

print(f"\n재계산 스텝 {len(nis_all)}   nis_g 중앙 {np.median(nis_all):.3f} "
      f"(원본 detail 과 대조용)")
print(f"\n{'축':>6s} {'측정 |z|':>12s} {'예측 |z_bar|':>14s} {'잔차 |r|':>12s} {'Pzz 대각':>11s} "
      f"{'r²/Pzz':>10s}")
for j, nm in enumerate(('gyro_x', 'gyro_y', 'gyro_z')):
    m = np.median(np.abs(meas_all[:, j]))
    p = np.median(np.abs(pred_all[:, j]))
    r = np.median(np.abs(res_all[:, 6 + j]))
    pz = np.median(pzz_all[:, 6 + j])
    print(f"{nm:>6s} {m:12.4f} {p:14.4f} {r:12.4f} {pz:11.5f} {r*r/pz:10.3f}")

print(f"\n{'축':>6s} {'측정 |z|':>12s} {'예측 |z_bar|':>14s} {'잔차 |r|':>12s} {'Pzz 대각':>11s} "
      f"{'r²/Pzz':>10s}")
for j, nm in enumerate(('vel_N', 'vel_E', 'vel_D')):
    m = np.median(np.abs(meas_all[:, 0] * 0 + 0))  # placeholder
    r = np.median(np.abs(res_all[:, 3 + j]))
    pz = np.median(pzz_all[:, 3 + j])
    print(f"{nm:>6s} {'—':>12s} {'—':>14s} {r:12.4f} {pz:11.5f} {r*r/pz:10.3f}")

print("\n" + "-" * 76)
print("해석")
print("-" * 76)
mg = np.median(np.abs(meas_all))
pg = np.median(np.abs(pred_all))
print(f"  자이로 측정 중앙 |z| = {mg:.4f} rad/s   (센서 노이즈 σ≈0.031 수준이면 '조용함')")
print(f"  자이로 예측 중앙 |z_bar| = {pg:.4f} rad/s")
if pg > 5 * max(mg, 1e-6):
    print("  → ★ 예측이 측정보다 훨씬 크다 = **모델이 없는 각속도를 만들어낸다**.")
    print("     경로는 u[1:4]/I 뿐 — 명령토크 × C_torque / 관성. C_torque_xy 14배 상향이 직접 용의자.")
elif mg > 5 * max(pg, 1e-6):
    print("  → 측정이 예측보다 크다 = 기체가 실제로 흔들리는데 모델이 못 따라간다(플랜트 문제).")
else:
    print("  → 측정·예측 크기가 비슷 = 위상/타이밍 어긋남이 잔차의 주원인.")
print(f"\n  Pzz 자이로 대각 중앙 = {np.median(pzz_all[:, 6:9]):.5f}  (R_gyro = 0.5)")
print(f"  → Pzz 가 R 보다 훨씬 작으면 NIS 분모가 작아져 NIS 가 부풀려진다.")
print("=" * 76)
