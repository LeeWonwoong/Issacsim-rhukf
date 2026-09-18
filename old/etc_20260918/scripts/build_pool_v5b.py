#!/usr/bin/env python3
"""build_pool_v5b (2026-09-12 밤) — v5 + 공격없는 hover 진입 4디렉토리(bandcap_arm003/005·slowcap·slowcap30, bias0 dhover3) → hover_entry 키 채움, entry dwell<=1. 원본 v5 — 확정 환경 풀: G2+ 필터 · 클립 4.0 · failsafe hover 데이터(cert/A-1/A-2/A-4/A-5/B/W/D/failsafe_dh/nomod) + 약공격(weakfloor, gapcheck, on15/on5 무풍)
풀 키 = v4 동일 (track_clean / hover_entry / hover_settled / post_track / post_hover / {track,hover}_atk_b0..7, δ 0.1~0.9 폭 0.1). 세그먼트 청크 병렬."""
import sys, os, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
R = 'results/claudecodefortest'; A = 4.36; CLIP = 4.0
CFG = dict(qg=2e-3, rg=0.2, qe=2e-3, qv=1e-3, rv=0.1)   # G2+
NEW = [f'{R}/{d}' for d in ('bandcap_arm003','bandcap_arm005','slowcap','slowcap30','cert_A','cert_A1','cert_A2lo','cert_A2hi','cert_A4_090','cert_A5_085','cert_B15','cert_B20','cert_B25','cert_B40','cert_B84_15','cert_B84_20','cert_B84_25','cert_W','cert_D_off','failsafe_dh','nomod')]
OLD = ['results_weakfloor', f'{R}/gapcheck', f'{R}/bandcap_on15', f'{R}/bandcap_on5']   # 구 hover(명목)·약공격: ws0 만
ENTRY_DWELL = 1   # v5b: hover 진입 스파이크는 1스텝 -> dwell 0,1 만 entry
KEYS = ['track_clean','hover_entry','hover_settled','post_track','post_hover'] + [f'{h}_atk_b{i}' for h in ('track','hover') for i in range(8)]
calib = load_calibration('calibration/calibration.json')
def binof(d): return min(7, max(0, int((d - 0.1) / 0.1)))
def work(task):
    D, s0, s1, keep_ws6 = task
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    P = {k: [] for k in KEYS}
    for k in range(s0, min(s1, len(starts) - 1, len(summ))):
        r = summ[k]; ws = float(r['wind_speed']); d_cell = float(r['bias']) / A
        if ws > 0 and not keep_ws6: continue
        a, b = starts[k], starts[k + 1]; seg = data[a:b]
        ukf = DynamicsUKF(dt=dt, calib=calib)
        for sl, key in ((slice(3, 6), 'qe'), (slice(6, 9), 'qv'), (slice(9, 12), 'qg')):
            for i in range(sl.start, sl.stop): ukf.Q[i, i] = CFG[key]
        for sl, key in ((slice(3, 6), 'rv'), (slice(6, 9), 'rg')):
            for i in range(sl.start, sl.stop): ukf.R[i, i] = CFG[key]
        fresh = np.ones(len(seg), bool); fresh[1:] = np.any(np.abs(seg[1:, 4:10] - seg[:-1, 4:10]) > 1e-12, axis=1); fresh[0] = True
        dwell = -1; since_end = 99; prevh = False; rl = 0
        for i, row in enumerate(seg):
            z = row[4:13].copy(); u = row[13:17].copy()
            if row[1] > 0.5:
                ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
            res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[i]))
            if not fresh[i]: continue
            _, v = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3, clip=CLIP); _, g = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3, clip=CLIP)
            rl += 1
            if rl < 15: continue
            h = row[3] > 0.5; at = (row[2] > 0.5) and d_cell > 0.05; dl = d_cell if at else 0.0
            dwell = (dwell + 1 if prevh else 0) if h else -1
            since_end = 0 if at else min(since_end + 1, 99)
            if at and dl >= 0.08: P[f'{"hover" if h else "track"}_atk_b{binof(dl)}'].append((g, v))
            elif not at:
                if 1 <= since_end <= 2: P['post_hover' if h else 'post_track'].append((g, v))
                elif h and dwell <= ENTRY_DWELL: P['hover_entry'].append((g, v))
                elif h: P['hover_settled'].append((g, v))
                else: P['track_clean'].append((g, v))
            prevh = h
    return P
if __name__ == '__main__':
    tasks = []
    for D in NEW + OLD:
        if not os.path.exists(f'{D}/zu_log.npz'): print('skip', D, flush=True); continue
        n = len(list(csv.DictReader(open(f'{D}/sweep_summary.csv'))))
        for s0 in range(0, n, 30): tasks.append((D, s0, s0 + 30, D in NEW))
    print(len(tasks), 'tasks', flush=True)
    with Pool(24) as p: parts = p.map(work, tasks, chunksize=1)
    P = {k: [] for k in KEYS}
    for part in parts:
        for k in KEYS: P[k].extend(part[k])
    out = {}; tot = 0
    for k, rows in P.items():
        arr = np.array(rows) if rows else np.zeros((0, 2)); tot += len(arr)
        out[f'g_{k}'] = np.sort(arr[:, 0]) if len(arr) else np.array([0.3]); out[f'v_{k}'] = np.sort(arr[:, 1]) if len(arr) else np.array([0.3])
        print(f'  {k:16s} n={len(arr):6d}  g med={np.median(arr[:,0]) if len(arr) else 0:.2f} p95={np.percentile(arr[:,0],95) if len(arr) else 0:.2f}  v med={np.median(arr[:,1]) if len(arr) else 0:.2f}', flush=True)
    os.makedirs(f'{R}/night', exist_ok=True); np.savez(f'{R}/night/train_pool_v5b.npz', **out); print(f'saved {R}/night/train_pool_v5b.npz 총 {tot} 프레임', flush=True)
