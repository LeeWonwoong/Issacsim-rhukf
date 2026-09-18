#!/usr/bin/env python3
"""filter_full_scan — Q(12D)·R(9D) 유효 5축 격자 오프라인 재생 (pos 블록은 무효 실증이라 제외).
   축: Q_gyro{2e-3,5e-3,1e-2,2e-2} × R_gyro{0.02,0.2} × Q_euler{2e-4,5e-4,2e-3} × Q_vel{1e-3,5e-3,2e-2} × R_vel{0.01,0.1} = 144
   데이터(세그먼트 부분집합): bandcap_arm005 ws6 track δ{0,0.25,0.74,0.8} + weakfloor δ0.15 ws{0,6} track
   조건(사용자): ①공격 중 잔차 유지 ②온셋 즉각 ③OFF 후 빠른 하강 ④vel 보조채널(지연 포함, 진폭) ⑤gyro 고원 낮게(바람 대비 애매) ⑥약공격 무클립(clip 3.5)
   출력: night/filter_full_scan.json + .log"""
import sys, os, csv, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
R = 'results/claudecodefortest'; A = 4.36; ATK0 = 180; CLIP = 3.5
GRID = dict(qg=(2e-3, 5e-3, 1e-2, 2e-2), rg=(0.02, 0.2), qe=(2e-4, 5e-4, 2e-3), qv=(1e-3, 5e-3, 2e-2), rv=(0.01, 0.1))
CFGS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
calib = load_calibration('calibration/calibration.json')
def load(D, want):
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    fresh = np.ones(len(data), bool); fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); fresh[data[:, 1] > 0.5] = True
    atk = data[:, 2] > 0.5; segs = []
    for k in range(min(len(starts) - 1, len(summ))):
        r = summ[k]; dl = round(float(r['bias']) / A, 2); ws = int(round(float(r['wind_speed'])))
        if r['policy'] != 'track' or (dl, ws) not in want: continue
        s0, s1 = starts[k], starts[k+1]; a = np.where(atk[s0:s1])[0]
        if dl > 0 and not len(a): continue
        o = a[0] if len(a) else int(0.6 * (s1 - s0)); on = (a[-1] - a[0] + 1) * dt * 10 if len(a) else 0
        t = ATK0 + (np.arange(s1 - s0) - o) * dt * 10
        segs.append(dict(D=D, dl=dl, ws=ws, s0=s0, s1=s1, t=t, on=on, seg=data[s0:s1], fresh=fresh[s0:s1], dt=dt))
    return segs
SEGS = load(f'{R}/bandcap_arm005', {(0.0, 6), (0.25, 6), (0.74, 6), (0.8, 6)}) + load('results_weakfloor', {(0.15, 0), (0.15, 6)})
def replay(seg, cfg):
    ukf = DynamicsUKF(dt=seg['dt'], calib=calib)
    for sl, key in ((slice(3, 6), 'qe'), (slice(6, 9), 'qv'), (slice(9, 12), 'qg')):
        for i in range(sl.start, sl.stop): ukf.Q[i, i] = cfg[key]
    for sl, key in ((slice(3, 6), 'rv'), (slice(6, 9), 'rg')):
        for i in range(sl.start, sl.stop): ukf.R[i, i] = cfg[key]
    data = seg['seg']; fresh = seg['fresh']; nv = np.zeros(len(data)); ng = np.zeros(len(data))
    for k, row in enumerate(data):
        z = row[4:13].copy(); u = row[13:17].copy()
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[k]))
        _, nv[k] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3, clip=CLIP); _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3, clip=CLIP)
    m = fresh; return seg['t'][m], ng[m], nv[m]
def auc(x, y):
    o = np.argsort(x); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
    return ((np.arange(1, len(ys)+1)[ys == 1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1*n0 else float('nan')
def sm(x, w=4): return np.convolve(x, np.ones(w)/w, 'same')
def work(ci):
    cfg = CFGS[ci]; C = {'clean': ([], []), 'wind': ([], []), 'a25': ([], []), 'a15': ([], []), 'a74': ([], []), 'a80': ([], [])}
    ons_g = []; ons_v = []; dec_g = []; dec_v = []; post_g = []; post_v = []
    for seg in SEGS:
        t, g, v = replay(seg, cfg); dl, ws = seg['dl'], seg['ws']; on = seg['on']
        if ws == 0 and dl == 0.15: key = None
        cl = (t >= 15) & (t < 55); wd = (t >= 70) & (t < 178) & (ws > 0)
        if dl == 0.0: C['clean'][0].extend(g[cl]); C['clean'][1].extend(v[cl])
        if ws > 0: C['wind'][0].extend(g[wd]); C['wind'][1].extend(v[wd])
        if dl > 0:
            am = (t >= ATK0 + 1) & (t < ATK0 + on); key = {0.25: 'a25', 0.15: 'a15', 0.74: 'a74', 0.8: 'a80'}[dl]
            C[key][0].extend(g[am]); C[key][1].extend(v[am])
            if dl == 0.25:
                pre = (t >= 70) & (t < 178); gb = np.median(g[pre]); vb = np.median(v[pre]); gp = np.median(g[am]); vp = np.median(v[am])
                for arr, base, plat, out in ((g, gb, gp, ons_g), (v, vb, vp, ons_v)):
                    idx = np.where((t >= ATK0) & (arr >= base + 0.5 * (plat - base)))[0]
                    out.append(t[idx[0]] - ATK0 if len(idx) else 60)
                off = ATK0 + on; pm = (t >= off) & (t < off + 40)
                for arr, base, plat, out, po in ((g, gb, gp, dec_g, post_g), (v, vb, vp, dec_v, post_v)):
                    s = sm(arr); idx = np.where((t >= off) & (s <= base + 0.25 * (plat - base)))[0]
                    out.append(t[idx[0]] - off if len(idx) else 40); po.append(np.median(arr[(t >= off + 1) & (t < off + 16)]))
    o = dict(cfg); G = {k: np.array(x[0]) for k, x in C.items()}; V = {k: np.array(x[1]) for k, x in C.items()}
    o.update(g_wind_p95=np.percentile(G['wind'], 95), g_wind_p99=np.percentile(G['wind'], 99), g_wind_med=np.median(G['wind']), g_clean_med=np.median(G['clean']),
             g25_5=np.percentile(G['a25'], 5), g25_med=np.median(G['a25']), g15_5=np.percentile(G['a15'], 5), g74_clip=np.mean(G['a74'] >= CLIP - 1e-3), g80_med=np.median(G['a80']),
             g_ratio=np.percentile(G['wind'], 95) / max(np.percentile(G['a25'], 5), 1e-6), g_on=np.mean(ons_g), g_dec=np.mean(dec_g), g_post=np.mean(post_g),
             v_wind_med=np.median(V['wind']), v_wind_p95=np.percentile(V['wind'], 95), v_clean_med=np.median(V['clean']),
             v25_med=np.median(V['a25']), v25_amp=np.median(V['a25']) - np.median(V['wind']), v15_amp=np.median(V['a15']) - np.median(V['wind']), v80_med=np.median(V['a80']),
             v25_auc=auc(np.r_[V['wind'], V['a25']], np.r_[np.zeros(len(V['wind'])), np.ones(len(V['a25']))]),
             v15_auc=auc(np.r_[V['wind'], V['a15']], np.r_[np.zeros(len(V['wind'])), np.ones(len(V['a15']))]),
             v_wind_auc=auc(np.r_[V['clean'], V['wind']], np.r_[np.zeros(len(V['clean'])), np.ones(len(V['wind']))]),
             v_on=np.mean(ons_v), v_dec=np.mean(dec_v), v_post=np.mean(post_v))
    print(f"[{ci:3d}] qg{cfg['qg']:g} rg{cfg['rg']:g} qe{cfg['qe']:g} qv{cfg['qv']:g} rv{cfg['rv']:g} | g wind p95 {o['g_wind_p95']:.2f} a25 {o['g25_5']:.2f}/{o['g25_med']:.2f} a15 {o['g15_5']:.2f} ratio {o['g_ratio']:.2f} clip74 {o['g74_clip']:.2f} on {o['g_on']:.1f} dec {o['g_dec']:.1f} | v wind {o['v_wind_med']:.2f} a25 {o['v25_med']:.2f} amp {o['v25_amp']:.2f} auc {o['v25_auc']:.2f}/{o['v15_auc']:.2f} wauc {o['v_wind_auc']:.2f} on {o['v_on']:.1f} dec {o['v_dec']:.1f} post {o['v_post']:.2f}", flush=True)
    return o
if __name__ == '__main__':
    print(f'{len(CFGS)} configs × {len(SEGS)} segs', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 20) as p: res = p.map(work, range(len(CFGS)), chunksize=1)
    json.dump(res, open(f'{R}/night/filter_full_scan.json', 'w'), default=float); print('DONE', flush=True)
