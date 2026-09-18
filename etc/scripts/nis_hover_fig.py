#!/usr/bin/env python3
"""nis_hover_fig (failsafe hover 중 NIS: track vs 명목 hover vs failsafe hover)
원본: nis_nomod_fig — 속도변조 on(bandcap_arm005/results_bandcap) vs off(nomod) time-step NIS, G2+ 필터·클립 4.0, 6패널"""
import sys, os, csv, collections, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
A = 4.36; ATK0 = 180; CFG = dict(qg=2e-3, rg=0.2, qe=2e-3, qv=1e-3, rv=0.1); calib = load_calibration('calibration/calibration.json')
def load(D):
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    fresh = np.ones(len(data), bool); fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); fresh[data[:, 1] > 0.5] = True
    for r in summ: r['delta'] = round(float(r['bias']) / A, 2); r['ws'] = int(round(float(r['wind_speed'])))
    return data, dt, summ, starts, fresh
def replay_seg(data, dt, fresh, s0, s1):
    ukf = DynamicsUKF(dt=dt, calib=calib)
    for sl, key in ((slice(3, 6), 'qe'), (slice(6, 9), 'qv'), (slice(9, 12), 'qg')):
        for i in range(sl.start, sl.stop): ukf.Q[i, i] = CFG[key]
    for sl, key in ((slice(3, 6), 'rv'), (slice(6, 9), 'rg')):
        for i in range(sl.start, sl.stop): ukf.R[i, i] = CFG[key]
    nv = np.zeros(s1 - s0); ng = np.zeros(s1 - s0)
    for j, k in enumerate(range(s0, s1)):
        row = data[k]; z = row[4:13].copy(); u = row[13:17].copy()
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[k]))
        _, nv[j] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3, clip=4.0); _, ng[j] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3, clip=4.0)
    return nv, ng
CACHE = {}
def series(D, delta, ws, pattern, T=np.arange(0, 300), policy='track'):
    if D not in CACHE: CACHE[D] = load(D)
    data, dt, summ, starts, fresh = CACHE[D]; atkf = data[:, 2] > 0.5
    ks = [k for k in range(min(len(starts) - 1, len(summ))) if (summ[k]['delta'], summ[k]['ws'], summ[k]['pattern'], summ[k]['policy']) == (delta, ws, pattern, policy)]
    offs = [np.where(atkf[starts[k]:starts[k+1]])[0][0] for k in range(len(starts) - 1) if atkf[starts[k]:starts[k+1]].any()]; off_med = int(np.median(offs))
    G = []; V = []; crashes = []
    for k in ks:
        s0, s1 = starts[k], starts[k+1]; a = np.where(atkf[s0:s1])[0]; o = a[0] if len(a) else off_med
        t = ATK0 + (np.arange(s1 - s0) - o) * dt * 10; m = fresh[s0:s1]
        if m.sum() < 10: continue
        nv, ng = replay_seg(data, dt, fresh, s0, s1)
        G.append(np.interp(T, t[m], ng[m], left=np.nan, right=np.nan)); V.append(np.interp(T, t[m], nv[m], left=np.nan, right=np.nan))
        if int(float(summ[k]['survived'])) == 0: crashes.append(int(float(summ[k]['crash_step'])))
    return (np.nanmean(G, 0), np.nanmean(V, 0), len(ks), crashes) if G else (None, None, 0, [])
FS = 'results/claudecodefortest/failsafe_dh'; NOM6 = 'results/claudecodefortest/bandcap_arm005'; NOM0 = 'results_bandcap'
PANELS = [(0.76, 6, 'waypoint'), (0.76, 6, 'aggressive'), (0.76, 6, 'circle'), (0.8, 6, 'waypoint'), (0.8, 6, 'aggressive'), (0.8, 6, 'circle'), (0.8, 0, 'waypoint'), (0.8, 0, 'aggressive'), (0.8, 0, 'circle')]
T = np.arange(0, 300); fig, ax = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
for a, (dl, ws, p) in zip(ax.ravel(), PANELS):
    print('panel', dl, ws, p, flush=True)
    for D, pol, lab, ls, lw in ((FS, 'track', 'track', ':', 1.2), (NOM6 if ws == 6 else NOM0, 'dhover3', '명목 hover', '--', 1.3), (FS, 'dhover3', 'failsafe hover', '-', 2.0)):
        g, v, n, cr = series(D, dl, ws, p, policy=pol)
        if g is None: continue
        a.plot(T, g, color='tab:red', ls=ls, lw=lw, label=f'gyro {lab} (n{n})'); a.plot(T, v, color='tab:blue', ls=ls, lw=lw, label=f'vel {lab}')
        for c in cr: a.axvline(c, color='k' if pol == 'track' else 'gray', lw=0.8, ls=':')
    if ws > 0: a.axvspan(60, 290, color='c', alpha=.06)
    a.axvspan(ATK0, ATK0 + 30, color='r', alpha=.12); a.axvline(183, color='b', lw=0.8, ls='--')
    a.set_ylim(0, 4.1); a.grid(alpha=.3); a.set_title(f'{p} · δ{dl} · ws{ws}', fontsize=10); a.legend(fontsize=7, ncol=3, loc='upper left')
for a in ax[-1]: a.set_xlabel('step (10 Hz)')
fig.suptitle('공격 중 NIS — track(점선) vs 명목 hover(파선) vs failsafe hover(실선), G2+ · 클립 4.0 · gyro 빨강 / vel 파랑 · 빨간 띠 = 공격 3 s · 파란 세로선 = hover 전환(+0.3 s)')
fig.tight_layout(); fig.savefig('results/claudecodefortest/night/nis_hover_compare.png', dpi=100); print('saved', flush=True)
