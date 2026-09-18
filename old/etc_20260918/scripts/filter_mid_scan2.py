#!/usr/bin/env python3
"""filter_mid_scan2 — 2차: 자세지연(vel 채널) 메커니즘을 유지한 채 gyro 만 낮추는 축 = Q_gyro 고정 2e-3(또는 1e-3) × R_gyro 상향 {0.3,0.5,1.0}."""
import sys, os, csv, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
R = 'results/claudecodefortest'; A = 4.36
CFGS = {'G2': dict(qg=2e-3, rg=0.2, qe=5e-4, rv=0.1)}
for qg in ('2e-3', '1e-3'):
    for rg in ('0.3', '0.5', '1.0'):
        CFGS[f'Qg{qg}_Rg{rg}'] = dict(qg=float(qg), rg=float(rg), qe=5e-4, rv=0.1)
def load(D):
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    return data, dt, summ, starts
def replay(data, dt, cfg):
    calib = load_calibration('calibration/calibration.json'); ukf = DynamicsUKF(dt=dt, calib=calib)
    for sl, v in ((slice(3, 6), cfg.get('qe')), (slice(9, 12), cfg.get('qg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.Q[i, i] = v
    for sl, v in ((slice(3, 6), cfg.get('rv')), (slice(6, 9), cfg.get('rg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.R[i, i] = v
    fresh = np.ones(len(data), bool); fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); fresh[data[:, 1] > 0.5] = True
    nv = np.zeros(len(data)); ng = np.zeros(len(data))
    for k, row in enumerate(data):
        z = row[4:13].copy(); u = row[13:17].copy()
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[k]))
        _, nv[k] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3); _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
    return nv, ng, fresh
def auc(x, y):
    o = np.argsort(x); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
    return ((np.arange(1, len(ys)+1)[ys == 1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1*n0 else float('nan')
def labels(data, summ, starts, fresh):
    atk = data[:, 2] > 0.5; n = len(data); lab = np.full(n, '', dtype=object); dl = np.zeros(n); wsv = np.zeros(n); rl = np.zeros(n)
    for k in range(len(starts) - 1):
        if k >= len(summ): break
        r = summ[k]; d = round(float(r['bias']) / A, 2); w = float(r['wind_speed']); seg = slice(starts[k], starts[k+1])
        a = np.where(atk[seg])[0]; o = a[0] if len(a) else int(0.6 * (starts[k+1] - starts[k]))
        t = 180 + (np.arange(starts[k+1] - starts[k]) - o) * 0.2
        dl[seg] = d; wsv[seg] = w; rl[seg] = t
    lab[(rl >= 15) & (rl < 55) & fresh] = 'clean'
    lab[(rl >= 70) & (rl < 178) & fresh & (wsv > 0)] = 'wind'
    lab[(rl >= 181) & fresh & atk & (dl > 0.05)] = 'atk'
    post = np.zeros(n, bool)
    for e in np.where(np.diff(atk.astype(int)) == -1)[0] + 1:
        fi = np.where(fresh[e:e+80])[0][:15]; post[e + fi] = True
    lab[post & (dl > 0.05)] = 'post'
    return lab, dl, wsv
def work(name):
    cfg = CFGS[name]; out = {'name': name}
    for tag, D in (('a', f'{R}/bandcap_arm005'), ('w', 'results_weakfloor')):
        data, dt, summ, starts = load(D); nv, ng, fresh = replay(data, dt, cfg); lab, dl, wsv = labels(data, summ, starts, fresh)
        fi = np.where(fresh)[0]; GW = np.zeros(len(ng)); VW = np.zeros(len(nv)); GW[fi] = np.convolve(ng[fi], np.ones(4)/4, 'same'); VW[fi] = np.convolve(nv[fi], np.ones(4)/4, 'same')
        cl = lab == 'clean'; wd = lab == 'wind'
        if tag == 'a':
            out['wind_med'], out['wind_p95'], out['wind_p99'] = np.median(ng[wd]), np.percentile(ng[wd], 95), np.percentile(ng[wd], 99)
            y = np.r_[np.zeros(cl.sum()), np.ones(wd.sum())]; out['wind_vel_auc'] = auc(np.r_[VW[cl], VW[wd]], y); out['wind_g_auc'] = auc(np.r_[GW[cl], GW[wd]], y)
            for d in (0.25, 0.74, 0.8):
                m = (lab == 'atk') & (np.abs(dl - d) < 0.01)
                if not m.sum(): continue
                out[f'g{d}_min'] = np.percentile(ng[m], 5); out[f'g{d}_med'] = np.median(ng[m]); out[f'clip{d}'] = np.mean(ng[m] >= 2.999)
                y = np.r_[np.zeros(wd.sum()), np.ones(m.sum())]; out[f'v{d}_auc'] = auc(np.r_[VW[wd], VW[m]], y); out[f'v{d}_med'] = np.median(nv[m])
                mp = (lab == 'post') & (np.abs(dl - d) < 0.01); out[f'post{d}_v'] = np.median(nv[mp]) if mp.sum() else float('nan'); out[f'post{d}_g'] = np.median(ng[mp]) if mp.sum() else float('nan')
        else:
            for d in (0.1, 0.15, 0.2):
                m = (lab == 'atk') & (np.abs(dl - d) < 0.01) & (wsv == 0)
                if m.sum(): out[f'wg{d}_min'] = np.percentile(ng[m], 5); out[f'wg{d}_med'] = np.median(ng[m]); y = np.r_[np.zeros(cl.sum()), np.ones(m.sum())]; out[f'wv{d}_auc'] = auc(np.r_[VW[cl], VW[m]], y)
    return out
if __name__ == '__main__':
    names = list(CFGS)
    with Pool(len(names)) as p: res = p.map(work, names)
    json.dump(res, open(f'{R}/night/filter_mid_scan2.json', 'w'), default=float)
    print(f"{'config':14s} {'wind med/p95/p99':>18s} {'δ.25 min/med':>13s} {'clip.25/.74':>11s} {'여유 p95/min':>11s} {'δ.15 min':>8s} {'vel AUC wind/.25/.15':>20s} {'post.25 v/g':>11s} {'vel med .25/.74':>15s}")
    for o in res:
        print(f"{o['name']:14s} {o['wind_med']:.2f}/{o['wind_p95']:.2f}/{o['wind_p99']:.2f}    {o['g0.25_min']:.2f}/{o['g0.25_med']:.2f}     {o['clip0.25']:.2f}/{o['clip0.74']:.2f}    {o['wind_p95']/o['g0.25_min']:.2f}      {o.get('wg0.15_min', float('nan')):.2f}    {o['wind_vel_auc']:.2f}/{o['v0.25_auc']:.2f}/{o.get('wv0.15_auc', float('nan')):.2f}      {o['post0.25_v']:.2f}/{o['post0.25_g']:.2f}   {o['v0.25_med']:.2f}/{o['v0.74_med']:.2f}")
