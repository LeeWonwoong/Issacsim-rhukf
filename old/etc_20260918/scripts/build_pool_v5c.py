#!/usr/bin/env python3
"""build_pool_v5c (2026-09-14) — v5b + ① 스텔스 빈 실측(cert_S: δ0.03/0.05/0.08 → {track,hover}_atk_s0..2) ② 바람 티어 분리
   (ws0 = 기본 키, ws>0 → 키에 _ws{n} 접미: track_clean_ws6/7/10, hover_settled_ws7/10, {h}_atk_b{i}_ws{n}, {h}_atk_s{i}_ws6).
   v5b 와 달리 기본 키는 ws0 만(v5b 는 NEW 디렉토리의 ws6 을 기본 키에 섞었음) — surrogate 가 티어를 명시적으로 뽑기 위함.
   사용: python3 build_pool_v5c.py [--dirs cert_S,cert_WIND,...] [--out path] [--procs 12]"""
import sys, os, csv, argparse, zlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
R = 'results/claudecodefortest'; A = 4.36; CLIP = 4.0
CFG = dict(qg=2e-3, rg=0.2, qe=2e-3, qv=1e-3, rv=0.1)   # G2+
NEW = ['bandcap_arm003', 'bandcap_arm005', 'slowcap', 'slowcap30', 'cert_A', 'cert_A1', 'cert_A2lo', 'cert_A2hi', 'cert_A4_090', 'cert_A5_085', 'cert_B15', 'cert_B20', 'cert_B25', 'cert_B40', 'cert_C', 'cert_D', 'cert_S', 'cert_WIND', 'cert_DEAD', 'cert_SW']   # v5d: +cert_DEAD(지연 hover) +cert_SW(바람 속 약공격·스텔스)
OLD = ['results_weakfloor', f'{R}/gapcheck', f'{R}/bandcap_on15', f'{R}/bandcap_on5']   # 구 hover(명목)·약공격: ws0 만
ENTRY_DWELL = 1
BASE_KEYS = ['track_clean', 'hover_entry', 'hover_settled', 'post_track', 'post_hover'] + [f'{h}_atk_b{i}' for h in ('track', 'hover') for i in range(8)] + [f'{h}_atk_s{i}' for h in ('track', 'hover') for i in range(3)]
WS_TIERS = (6, 7, 10)
KEYS = BASE_KEYS + [f'{k}_ws{w}' for w in WS_TIERS for k in BASE_KEYS]
calib = load_calibration('calibration/calibration.json')
# ★09-14 발견: Isaac 온라인 UKF 는 frozen_v3.env COM_BIAS_STD=0.05 로 에피소드마다 U(−0.05,0.05) N·m 토크 편향을 u 에 더해 필터링하는데
#   (online_rl_main._run_ukf_step, 로그엔 u_phys 만 기록) v5b 오프라인 재생은 이를 빼먹어 평시 gyro 바닥이 Isaac 보다 깨끗했다
#   (cert_S track ws0: 온라인 med 0.62 vs 오프라인 0.42, 에피소드별 0.21–0.81 편차 소실). 여기선 같은 분포로 에피소드별 편향을 재생한다(추첨은 k 기반 시드).
COM_BIAS_STD = float(os.environ.get('POOL_COM_BIAS', '0.05') or 0)
# ★09-15 저녁: aggressive 설정점 위상 버그(수정 전 캡처의 평시 gyro 꼬리 부풀림) 대응 — 기본 off.
#   POOL_EXCLUDE_PATTERN=aggressive 이면 POOL_EXCLUDE_KEEP(폴더 basename 목록) 밖 폴더의 해당 패턴 에피소드를 뺀다. POOL_EXTRA_DIRS 는 기본 폴더 목록에 덧붙인다.
EXC_PAT = set(x for x in os.environ.get('POOL_EXCLUDE_PATTERN', '').split(',') if x)
EXC_KEEP = set(x for x in os.environ.get('POOL_EXCLUDE_KEEP', '').split(',') if x)
def binof(d): return min(7, max(0, int((d - 0.1) / 0.1 + 1e-9)))   # ★09-15 저녁: 부동소수 절사 수정(δ0.3→b1·δ0.7→b5 오분류, PRECHAIN §0.3 P1)
def sbin(d): return 0 if d < 0.04 else (1 if d < 0.065 else 2)      # δ 0.03 / 0.05 / 0.08
def work(task):
    D, s0, s1 = task
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    P = {k: [] for k in KEYS}
    for k in range(s0, min(s1, len(starts) - 1, len(summ))):
        r = summ[k]; ws = int(round(float(r['wind_speed']))); d_cell = float(r['bias']) / A
        if EXC_PAT and r.get('pattern') in EXC_PAT and os.path.basename(D.rstrip('/')) not in EXC_KEEP: continue
        if ws not in (0,) + WS_TIERS: continue
        sfx = '' if ws == 0 else f'_ws{ws}'
        a, b = starts[k], starts[k + 1]; seg = data[a:b]
        ukf = DynamicsUKF(dt=dt, calib=calib)
        for sl, key in ((slice(3, 6), 'qe'), (slice(6, 9), 'qv'), (slice(9, 12), 'qg')):
            for i in range(sl.start, sl.stop): ukf.Q[i, i] = CFG[key]
        for sl, key in ((slice(3, 6), 'rv'), (slice(6, 9), 'rg')):
            for i in range(sl.start, sl.stop): ukf.R[i, i] = CFG[key]
        fresh = np.ones(len(seg), bool); fresh[1:] = np.any(np.abs(seg[1:, 4:10] - seg[:-1, 4:10]) > 1e-12, axis=1); fresh[0] = True
        dwell = -1; since_end = 99; prevh = False; rl = 0
        cb = np.random.default_rng(zlib.crc32(f'{D}:{k}'.encode())).uniform(-COM_BIAS_STD, COM_BIAS_STD, 2) if COM_BIAS_STD > 0 else np.zeros(2)
        # ★09-14 에피소드 스텝 정렬: zu 세그먼트는 에피소드 시작 전(이륙·재배치 ~40 스텝)부터 기록된다. 온라인 detail 은 step 10..steps 만 있으므로
        #   fresh 인덱스 i ↔ 온라인 step s = i − (n_fresh − (steps−10)) + 10. 에피소드 밖(s<10)과 바람 티어의 바람 전/후(WIND 60..290 밖)는 제외
        #   — 빼지 않으면 ws10 키 중앙값이 0.92 로 온라인(1.29)보다 낮게 나온다(cert_WIND 검증).
        n_fresh = int(fresh.sum()); n_on = max(1, int(float(r.get('steps', 301))) - 10); noff = n_fresh - n_on
        s_lo, s_hi = (10, 10 ** 6) if ws == 0 else (int(os.environ.get('POOL_WIND_S_LO', '70')), int(os.environ.get('POOL_WIND_S_HI', '290')))
        for i, row in enumerate(seg):
            z = row[4:13].copy(); u = row[13:17].copy(); u[1] += cb[0]; u[2] += cb[1]
            if row[1] > 0.5:
                ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
            res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[i]))
            if not fresh[i]: continue
            _, v = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3, clip=CLIP); _, g = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3, clip=CLIP)
            rl += 1
            s = rl - noff + 9
            if rl < 15 or s < s_lo or s > s_hi: continue
            h = row[3] > 0.5; at = (row[2] > 0.5) and d_cell > 0.02; dl = d_cell if at else 0.0
            dwell = (dwell + 1 if prevh else 0) if h else -1
            since_end = 0 if at else min(since_end + 1, 99)
            hn = 'hover' if h else 'track'
            if at and dl >= 0.1 - 1e-9: P[f'{hn}_atk_b{binof(dl)}{sfx}'].append((g, v))   # ★09-15: δ0.1(=0.436/4.36=0.0999…) 이 스텔스 빈으로 빠지던 절사 수정
            elif at: P[f'{hn}_atk_s{sbin(dl)}{sfx}'].append((g, v))
            else:
                if 1 <= since_end <= 2: P[f'post_hover{sfx}' if h else f'post_track{sfx}'].append((g, v))
                elif h and dwell <= ENTRY_DWELL: P[f'hover_entry{sfx}'].append((g, v))
                elif h: P[f'hover_settled{sfx}'].append((g, v))
                else: P[f'track_clean{sfx}'].append((g, v))
            prevh = h
    return P
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dirs', default=None); ap.add_argument('--out', default=f'{R}/night/train_pool_v5c.npz'); ap.add_argument('--procs', type=int, default=12)
    ar = ap.parse_args()
    dirs = [f'{R}/{x}' for x in ar.dirs.split(',')] if ar.dirs else [f'{R}/{x}' for x in NEW] + OLD
    dirs += [f'{R}/{x}' for x in os.environ.get('POOL_EXTRA_DIRS', '').split(',') if x]
    print('EXCLUDE', sorted(EXC_PAT), 'KEEP', sorted(EXC_KEEP), flush=True)
    tasks = []
    for D in dirs:
        if not os.path.exists(f'{D}/zu_log.npz'): print('skip', D, flush=True); continue
        n = len(list(csv.DictReader(open(f'{D}/sweep_summary.csv'))))
        for s0 in range(0, n, 30): tasks.append((D, s0, s0 + 30))
    print(len(tasks), 'tasks from', len(dirs), 'dirs', flush=True)
    with Pool(ar.procs) as p: parts = p.map(work, tasks, chunksize=1)
    P = {k: [] for k in KEYS}
    for part in parts:
        for k in KEYS: P[k].extend(part[k])
    out = {}; tot = 0
    for k, rows in P.items():
        arr = np.array(rows) if rows else np.zeros((0, 2)); tot += len(arr)
        out[f'g_{k}'] = np.sort(arr[:, 0]) if len(arr) else np.array([0.3]); out[f'v_{k}'] = np.sort(arr[:, 1]) if len(arr) else np.array([0.3])
        if len(arr): print(f'  {k:22s} n={len(arr):6d}  g med={np.median(arr[:,0]):.2f} p90={np.percentile(arr[:,0],90):.2f} p99={np.percentile(arr[:,0],99):.2f}  v med={np.median(arr[:,1]):.2f} p99={np.percentile(arr[:,1],99):.2f}', flush=True)
    os.makedirs(os.path.dirname(ar.out), exist_ok=True); np.savez(ar.out, **out); print(f'saved {ar.out} 총 {tot} 프레임', flush=True)
