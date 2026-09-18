#!/usr/bin/env python3
"""replay_g_qr — 오프라인 G(=C_torque/I 오차) × Q/R 재생 진단. zu_log.npz 재실행.
  ctq e → u_torque×(1+e) · I e → calib I×(1+e) · Q/R 격자.
판정: 급기동 gyro 가 공격과 겹치나(thrF1↓=자명함 해소) vs 순항 바닥(부작용)."""
import sys, os, copy, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf')
os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled

Z = sys.argv[1] if len(sys.argv) > 1 else 'results_calibcap/zu_log.npz'
d = np.load(Z, allow_pickle=True)
data = d['data'].astype(np.float64); dt = float(d['dt'])
base_calib = load_calibration('calibration/calibration.json')
print(f"# zu_log: {data.shape[0]} rows @ dt={dt}")

gps_fresh = np.ones(len(data), bool)
gps_fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1)
gps_fresh[data[:, 1] > 0.5] = True

# 에피소드 → bias 매핑 (sweep_summary 행 순서 = 실행 순서. col20 은 하이재킹 모드서 죽어있음)
import csv as _csv
_summ = list(_csv.DictReader(open(os.path.join(os.path.dirname(Z) or '.', 'sweep_summary.csv'))))
_st = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
seg_bias = np.zeros(len(data))
for k in range(len(_st) - 1):
    seg_bias[_st[k]:_st[k+1]] = float(_summ[k]['bias']) if k < len(_summ) else 0.0
print(f"# segs={len(_st)-1} summary={len(_summ)} biases={sorted(set(seg_bias.tolist()))}")

atk = (data[:, 2] > 0.5) & (seg_bias > 0.1)          # 진짜 공격 (bias>0 ∧ 플래그)
gyr_mag = np.linalg.norm(data[:, 10:13], axis=1)
normal = (seg_bias < 0.1) & (data[:, 2] < 0.5) & (data[:, 1] < 0.5)   # bias0 에피 평시
cruise = normal & (gyr_mag < 0.3)
maneuv = normal & (gyr_mag >= 0.5)
print(f"# frames: attack={atk.sum()} cruise={cruise.sum()} maneuver={maneuv.sum()}")

def replay(ctq=0.0, iI=0.0, qg=None, rg=None):
    calib = copy.deepcopy(base_calib)
    if iI:
        for k in ('Ixx', 'Iyy', 'Izz'):
            calib['drone'][k] = float(calib['drone'][k]) * (1.0 + iI)
    ukf = DynamicsUKF(dt=dt, calib=calib)
    if qg is not None:
        for i in (9, 10, 11): ukf.Q[i, i] = qg
    if rg is not None:
        for i in (6, 7, 8): ukf.R[i, i] = rg
    ng = np.zeros(len(data))
    for k, row in enumerate(data):
        z = row[4:13].copy(); u = row[13:17].copy()
        u[1:4] *= (1.0 + ctq)
        if row[1] > 0.5:
            ukf.x = np.zeros(12)
            ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]
            ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]
            ukf.P = np.eye(12) * 0.1
            ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(gps_fresh[k]))
        _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
    return ng

def metrics(ng):
    a = ng[atk]; c = ng[cruise]; m = ng[maneuv]; na = ng[normal]
    best = 0.0
    for th in np.arange(0.3, 2.9, 0.05):
        tp = (a > th).sum(); fp = (na > th).sum(); fn = (a <= th).sum()
        p = tp / (tp + fp or 1); rc = tp / (tp + fn or 1)
        if p + rc: best = max(best, 2 * p * rc / (p + rc))
    x = np.r_[a, m]; y = np.r_[np.ones(len(a)), np.zeros(len(m))]
    o = np.argsort(x); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
    auc = ((np.arange(1, len(ys)+1)[ys == 1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1*n0 else 0.5
    return dict(atk=np.median(a), cru=np.median(c), cru95=np.percentile(c, 95),
                man=np.median(m), man95=np.percentile(m, 95), thrF1=best, auc=auc)

def show(name, mm):
    print(f"  {name:22s} atk={mm['atk']:.2f} | cruise {mm['cru']:.2f}/{mm['cru95']:.2f} | "
          f"man {mm['man']:.2f}/{mm['man95']:.2f} | thrF1={mm['thrF1']:.3f} AUC(a/m)={mm['auc']:.3f}", flush=True)

print("\n■ G 축 (기본 Q/R)  [thrF1↓ = 자명함 해소 · cruise/man↑ = 그 수단 · cruise 과대 = 부작용]")
for nm, c, i in [('base', 0, 0), ('ctq+5%', .05, 0), ('ctq+10%', .10, 0), ('ctq-10%', -.10, 0),
                 ('i-5% (G+5)', 0, -.05), ('i-10% (G+11)', 0, -.10),
                 ('G+22 (c10,i-10)', .10, -.10), ('G+41 (c20,i-15)', .20, -.15)]:
    show(nm, metrics(replay(ctq=c, iI=i)))

print("\n■ Q/R 격자 — baseG · G+22  [현행 Q_gyro 2e-2 · R_gyro 0.02]")
for gtag, gc, gi in (('baseG', 0, 0), ('G+22', .10, -.10)):
    for qg in (5e-3, 2e-2, 5e-2):
        for rg in (0.005, 0.02, 0.1):
            show(f'{gtag} Q{qg:g} R{rg:g}', metrics(replay(ctq=gc, iI=gi, qg=qg, rg=rg)))
print("\ndone")
