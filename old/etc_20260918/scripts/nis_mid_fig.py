#!/usr/bin/env python3
"""nis_mid_fig — 중간 필터 M(Q_gyro 1e-2·R_gyro 0.1·Q_euler 5e-4·R_vel 0.1) 을 E·G2 와 겹친 대표 time-step NIS 그림 6패널.
   데이터: bandcap_arm005(ws6, track) + results_weakfloor(δ0.15). 필요한 세그먼트만 재생(세그먼트 시작에서 UKF 재초기화되므로 독립).
   출력: results/claudecodefortest/night/nis_mid_compare2.png"""
import sys, os, csv, collections, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
A = 4.36; ATK0 = 180
CFGS = {'E': dict(), 'M': dict(qg=1e-2, rg=0.1, qe=5e-4, rv=0.1), 'M2': dict(qg=5e-3, rg=0.1, qe=5e-4, rv=0.1), 'G2': dict(qg=2e-3, rg=0.2, qe=5e-4, rv=0.1)}
STY = {'E': dict(ls='-', lw=1.0, alpha=.7), 'M': dict(ls='-', lw=2.0, alpha=1.0), 'M2': dict(ls='-.', lw=1.6, alpha=1.0), 'G2': dict(ls=':', lw=1.3, alpha=.9)}
calib = load_calibration('calibration/calibration.json')
def load(D):
    d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    fresh = np.ones(len(data), bool); fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); fresh[data[:, 1] > 0.5] = True
    for r in summ: r['delta'] = round(float(r['bias']) / A, 2); r['ws'] = int(round(float(r['wind_speed'])))
    return data, dt, summ, starts, fresh
def replay_seg(data, dt, fresh, s0, s1, cfg):
    ukf = DynamicsUKF(dt=dt, calib=calib)
    for sl, v in ((slice(3, 6), cfg.get('qe')), (slice(9, 12), cfg.get('qg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.Q[i, i] = v
    for sl, v in ((slice(3, 6), cfg.get('rv')), (slice(6, 9), cfg.get('rg'))):
        if v is not None:
            for i in range(sl.start, sl.stop): ukf.R[i, i] = v
    nv = np.zeros(s1 - s0); ng = np.zeros(s1 - s0)
    for j, k in enumerate(range(s0, s1)):
        row = data[k]; z = row[4:13].copy(); u = row[13:17].copy()
        if row[1] > 0.5:
            ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]; ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u, gps_fresh=bool(fresh[k]))
        _, nv[j] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3); _, ng[j] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
    return nv, ng
def series(D, delta, ws, pattern, on, T=np.arange(0, 300)):
    """(delta, ws, pattern, track) 세그먼트 평균 곡선 {cfg: (gyro, vel)} + 추락 스텝 목록"""
    data, dt, summ, starts, fresh = CACHE[D]
    ks = [k for k in range(min(len(starts) - 1, len(summ))) if (summ[k]['delta'], summ[k]['ws'], summ[k]['pattern'], summ[k]['policy']) == (delta, ws, pattern, 'track')]
    atkf = data[:, 2] > 0.5
    offs = [np.where(atkf[starts[k]:starts[k+1]])[0][0] for k in range(len(starts) - 1) if atkf[starts[k]:starts[k+1]].any()]
    off_med = int(np.median(offs))
    out = {nm: ([], []) for nm in CFGS}; crashes = []
    for k in ks:
        s0, s1 = starts[k], starts[k+1]; a = np.where(atkf[s0:s1])[0]; o = a[0] if len(a) else off_med
        t = ATK0 + (np.arange(s1 - s0) - o) * dt * 10; m = fresh[s0:s1]
        if m.sum() < 10: continue
        for nm, cfg in CFGS.items():
            nv, ng = replay_seg(data, dt, fresh, s0, s1, cfg)
            out[nm][0].append(np.interp(T, t[m], ng[m], left=np.nan, right=np.nan)); out[nm][1].append(np.interp(T, t[m], nv[m], left=np.nan, right=np.nan))
        if int(float(summ[k]['survived'])) == 0: crashes.append(int(float(summ[k]['crash_step'])))
    return {nm: (np.nanmean(g, 0), np.nanmean(v, 0)) for nm, (g, v) in out.items() if g}, len(ks), crashes
BA = 'results/claudecodefortest/bandcap_arm005'; WF = 'results_weakfloor'
CACHE = {BA: load(BA), WF: load(WF)}
PANELS = [(BA, 0.0, 6, 'waypoint', 30, '바람만 ws6 · waypoint'), (BA, 0.0, 6, 'aggressive', 30, '바람만 ws6 · aggressive'),
          (BA, 0.25, 6, 'waypoint', 30, '약공격 δ0.25 ws6 · waypoint'), (BA, 0.25, 6, 'aggressive', 30, '약공격 δ0.25 ws6 · aggressive'),
          (WF, 0.15, 6, 'circle', 30, '약공격 하한 δ0.15 ws6 · circle'), (BA, 0.8, 6, 'aggressive', 30, '치명 δ0.80 ws6 · aggressive')]
T = np.arange(0, 300)
fig, ax = plt.subplots(3, 2, figsize=(14, 9.5), sharex=True)
for a, (D, dl, ws, p, on, title) in zip(ax.ravel(), PANELS):
    print('panel', title, flush=True)
    cur, n, crashes = series(D, dl, ws, p, on)
    for nm in ('E', 'G2', 'M2', 'M'):
        if nm not in cur: continue
        g, v = cur[nm]; st = STY[nm]
        a.plot(T, g, color='tab:red', label=f'gyro {nm}', **st); a.plot(T, v, color='tab:blue', label=f'vel {nm}', **st)
    if ws > 0: a.axvspan(60, 290, color='c', alpha=.06)
    if dl > 0: a.axvspan(ATK0, ATK0 + on, color='r', alpha=.12)
    for c in crashes: a.axvline(c, color='k', lw=0.8, ls=':')
    a.set_ylim(0, 3.1); a.grid(alpha=.3); a.set_title(f'{title} (track, n={n})', fontsize=10)
    if a is ax[0][0]: a.legend(fontsize=7, ncol=4, loc='upper left')
for a in ax[-1]: a.set_xlabel('step (10 Hz)')
fig.suptitle('time-step NIS — E(가는 실선) · G2(점선) · M=Qg1e-2·Rg0.1(굵은 실선) · M2=Qg5e-3·Rg0.1(일점쇄선), 둘 다 Qe5e-4·Rv0.1 · gyro 빨강 / vel 파랑 · 빨간 띠=공격 ON 3 s · 세로 점선=추락')
fig.tight_layout(); os.makedirs('results/claudecodefortest/night', exist_ok=True)
fig.savefig('results/claudecodefortest/night/nis_mid_compare2.png', dpi=110); print('saved', flush=True)
