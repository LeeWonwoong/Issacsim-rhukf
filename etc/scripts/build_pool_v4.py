#!/usr/bin/env python3
"""build_pool_v4 — FF-on 무대(bandcap ON5/15/30·weakfloor·arm축) zu 로그를 **선택 필터(E|G2)** 로 재생해 surrogate 풀 생성.
풀 키(v3 동일): track_clean / hover_entry(dwell≤2) / hover_settled / post_track / post_hover / {track,hover}_atk_b0..7 (δ 0.1~0.9 폭 0.1)
행 라벨은 zu 컬럼(action, attack, atk_scale)으로: 정책 track/dhover3 의 실제 행동을 그대로 사용. GPS-fresh 행(10 Hz)만.
바람: ws0 셀 전부 + 지정 arm 의 ws6 셀 (arm 외 폴더의 ws6 셀은 제외).
사용: python3 build_pool_v4.py --filter G2 --arm 0.05 --out etc/scripts/train_pool_v4.npz"""
import sys, os, csv, argparse, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
ap = argparse.ArgumentParser()
ap.add_argument('--filter', default='E'); ap.add_argument('--arm', type=float, default=0.07)
ap.add_argument('--out', default='etc/scripts/train_pool_v4.npz'); ap.add_argument('--dirs', nargs='*', default=None)
a = ap.parse_args()
CFG = {'E': dict(), 'G2': dict(qg=2e-3, rg=0.2, qe=5e-4, rv=0.1)}[a.filter]
R = 'results/claudecodefortest'
ARMDIR = {0.07: 'results_bandcap', 0.05: f'{R}/bandcap_arm005', 0.03: f'{R}/bandcap_arm003'}
DIRS = a.dirs or ['results_bandcap', f'{R}/bandcap_on5', f'{R}/bandcap_on15', 'results_weakfloor', f'{R}/bandcap_arm005', f'{R}/bandcap_arm003']
calib = load_calibration('calibration/calibration.json')
P = {k: [] for k in ['track_clean', 'hover_entry', 'hover_settled', 'post_track', 'post_hover']}
for i in range(8): P[f'track_atk_b{i}'] = []; P[f'hover_atk_b{i}'] = []
def binof(d): return min(7, max(0, int((d - 0.1) / 0.1)))
def replay(data, dt):
    ukf = DynamicsUKF(dt=dt, calib=calib)
    for sl, v in ((slice(3, 6), CFG.get('qe')), (slice(9, 12), CFG.get('qg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.Q[i, i] = v
    for sl, v in ((slice(3, 6), CFG.get('rv')), (slice(6, 9), CFG.get('rg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.R[i, i] = v
    gps_fresh = np.ones(len(data), bool); gps_fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); gps_fresh[data[:, 1] > 0.5] = True
    nv = np.zeros(len(data)); ng = np.zeros(len(data))
    for k, row in enumerate(data):
        z = row[4:13].copy(); u = row[13:17].copy()
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]
            ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(gps_fresh[k]))
        _, nv[k] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3); _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
    return nv, ng, gps_fresh
tot = 0
for D in DIRS:
    zp = os.path.join(D, 'zu_log.npz')
    if not os.path.exists(zp): print(f'  (skip {D}: zu 없음)', flush=True); continue
    d = np.load(zp, allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(os.path.join(D, 'sweep_summary.csv'))))
    starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    is_armdir = any(os.path.normpath(D) == os.path.normpath(v) for v in ARMDIR.values())
    keep_ws6 = os.path.normpath(D) == os.path.normpath(ARMDIR[a.arm])
    print(f'[{D}] rows={len(data)} segs={len(starts)-1} summary={len(summ)} filter={a.filter} ws6포함={keep_ws6}', flush=True)
    nv, ng, fresh = replay(data, dt)
    n_used = 0
    for k in range(len(starts) - 1):
        if k >= len(summ): break
        r = summ[k]; ws = float(r['wind_speed']); d_cell = float(r['bias']) / 4.36   # zu 의 atk_scale 은 스윕에서 0 → 요약 bias 사용
        if ws > 0 and not keep_ws6: continue
        seg = range(starts[k], starts[k+1]); dwell = -1; since_end = 99; prevh = False; preva = False; rl = 0
        for i in seg:
            if not fresh[i]: continue
            rl += 1
            if rl < 15: continue                         # 이륙/안정화 초기 제외
            h = data[i, 3] > 0.5; at = (data[i, 2] > 0.5) and d_cell > 0.05; dl = d_cell if at else 0.0
            dwell = (dwell + 1 if prevh else 0) if h else -1
            since_end = 0 if at else min(since_end + 1, 99)
            g = float(ng[i]); v = float(nv[i])
            if at and dl >= 0.08: P[f'{"hover" if h else "track"}_atk_b{binof(dl)}'].append((g, v))
            elif not at:
                if 1 <= since_end <= 2: P['post_hover' if h else 'post_track'].append((g, v))
                elif h and dwell <= 2: P['hover_entry'].append((g, v))
                elif h: P['hover_settled'].append((g, v))
                else: P['track_clean'].append((g, v))
            prevh = h; preva = at; n_used += 1
    tot += n_used; print(f'   사용 프레임 {n_used}', flush=True)
out = {}
for k, rows in P.items():
    arr = np.array(rows) if rows else np.zeros((0, 2))
    out[f'g_{k}'] = np.sort(arr[:, 0]) if len(arr) else np.array([0.3]); out[f'v_{k}'] = np.sort(arr[:, 1]) if len(arr) else np.array([0.3])
    print(f'  {k:16s} n={len(arr):6d}  g med={np.median(arr[:,0]) if len(arr) else 0:.3f} p95={np.percentile(arr[:,0],95) if len(arr) else 0:.3f}  v med={np.median(arr[:,1]) if len(arr) else 0:.3f}')
np.savez(a.out, **out); print(f'saved {a.out}  (총 {tot} 프레임, filter {a.filter}, arm {a.arm})')
