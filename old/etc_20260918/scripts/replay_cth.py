#!/usr/bin/env python3
"""replay_cth — C_thrust 오차 재생: vel/gyro 양채널 애매화 판정.
  cth e → u_thrust×(1+e). 호버 추력 13.4N 이라 10%=1.34N — 토크(0.05~0.12Nm)와 달리 절대크기 큼.
  vel 채널(병진 가속 예측) + 결합 K×T(자세) 양쪽으로 전파."""
import sys, os, copy, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf')
os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
Z = 'results_calibcap/zu_log.npz'
d = np.load(Z, allow_pickle=True)
data = d['data'].astype(np.float64); dt = float(d['dt'])
base_calib = load_calibration('calibration/calibration.json')
gps_fresh = np.ones(len(data), bool)
gps_fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1)
gps_fresh[data[:, 1] > 0.5] = True
import csv as _csv
_summ = list(_csv.DictReader(open('results_calibcap/sweep_summary.csv')))
_st = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
seg_bias = np.zeros(len(data))
for k in range(len(_st) - 1):
    seg_bias[_st[k]:_st[k+1]] = float(_summ[k]['bias']) if k < len(_summ) else 0.0
atk = (data[:, 2] > 0.5) & (seg_bias > 0.1)
gyr_mag = np.linalg.norm(data[:, 10:13], axis=1)
normal = (seg_bias < 0.1) & (data[:, 2] < 0.5) & (data[:, 1] < 0.5)
cruise = normal & (gyr_mag < 0.3); maneuv = normal & (gyr_mag >= 0.5)

def replay(cth=0.0, ctq=0.0):
    ukf = DynamicsUKF(dt=dt, calib=copy.deepcopy(base_calib))
    ng = np.zeros(len(data)); nv = np.zeros(len(data))
    for k, row in enumerate(data):
        z = row[4:13].copy(); u = row[13:17].copy()
        u[0] *= (1.0 + cth); u[1:4] *= (1.0 + ctq)
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]
            ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]
            ukf.P = np.eye(12)*0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(gps_fresh[k]))
        _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
        _, nv[k] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3)
    return ng, nv

def met(x, mask=None):
    mk = np.ones(len(x), bool) if mask is None else mask
    a=x[atk&mk]; c=x[cruise&mk]; m=x[maneuv&mk]; na=x[normal&mk]
    best=0.0
    for th in np.arange(0.3,2.9,0.05):
        tp=(a>th).sum(); fp=(na>th).sum(); fn=(a<=th).sum()
        p=tp/(tp+fp or 1); rc=tp/(tp+fn or 1)
        if p+rc: best=max(best,2*p*rc/(p+rc))
    return f"atk={np.median(a):.2f} cru={np.median(c):.2f}/{np.percentile(c,95):.2f} man={np.median(m):.2f}/{np.percentile(m,95):.2f} thrF1={best:.3f}"

print("■ C_thrust 오차 재생 (gyro | vel 양채널)")
for nm, cth, ctq in [('base',0,0),('cth+5%',.05,0),('cth+10%',.10,0),('cth-10%',-.10,0),
                     ('cth+10+ctq+10',.10,.10)]:
    ng, nv = replay(cth=cth, ctq=ctq)
    print(f"  {nm:16s} GYRO {met(ng)}")
    print(f"  {'':16s} VEL  {met(nv, gps_fresh)}", flush=True)
print("done")
