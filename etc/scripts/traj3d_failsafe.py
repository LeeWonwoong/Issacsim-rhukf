#!/usr/bin/env python3
"""traj3d_failsafe — hover 전/후(명목 vs failsafe) dhover3 3D 궤적 비교. δ{0.76,0.80} × ws{0,6} × 5패턴.
   명목: δ0.80 = nomod_dh (변조 off), δ0.76 = results_bandcap(ws0)/bandcap_arm005(ws6). failsafe: failsafe_dh."""
import csv, collections, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(fp); matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name(); matplotlib.rcParams['axes.unicode_minus'] = False
A = 4.36; ATK0 = 180; R = 'results/claudecodefortest'; PATS = ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve']
def load(D):
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv')))
    for r in summ: r['delta'] = round(float(r['bias']) / A, 2); r['ws'] = int(round(float(r['wind_speed'])))
    det = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
        if r['policy'] == 'dhover3' and int(r['step']) >= 20: det[(int(r['cell_idx']), int(r['episode']))].append(r)
    for k in det: det[k].sort(key=lambda r: int(r['step']))
    return summ, det
SRC = {(0.8, 0, 'X'): f'{R}/nomod_dh', (0.8, 6, 'X'): f'{R}/nomod_dh', (0.76, 0, 'X'): 'results_bandcap', (0.76, 6, 'X'): f'{R}/bandcap_arm005'}
for k in [(0.76, 0), (0.76, 6), (0.8, 0), (0.8, 6)]: SRC[(k[0], k[1], 'O')] = f'{R}/failsafe_dh'
CACHE = {}
def get(D):
    if D not in CACHE: CACHE[D] = load(D)
    return CACHE[D]
for ws in (0, 6):
    rows = [(0.76, 'X'), (0.76, 'O'), (0.8, 'X'), (0.8, 'O')]
    fig = plt.figure(figsize=(4.3 * len(PATS), 3.7 * len(rows)))
    for ri, (dl, mode) in enumerate(rows):
        summ, det = get(SRC[(dl, ws, mode)])
        for ci, p in enumerate(PATS):
            a = fig.add_subplot(len(rows), len(PATS), ri * len(PATS) + ci + 1, projection='3d')
            cells = [((int(r['cell_idx']), int(r['episode'])), int(float(r['survived']))) for r in summ if r['delta'] == dl and r['ws'] == ws and r['policy'] == 'dhover3' and r['pattern'] == p]
            first = True; dmax = []
            for ck, surv in cells:
                rs = det.get(ck, [])
                if not rs: continue
                x = np.array([float(r['pos_x']) for r in rs]); y = np.array([float(r['pos_y']) for r in rs]); z = np.array([float(r['alt']) for r in rs])
                st = np.array([int(r['step']) for r in rs]); act = np.array([int(float(r['attack_active'])) for r in rs])
                if first and rs[0].get('ref_x', '') not in ('', 'nan'):
                    rx = np.array([float(r['ref_x']) for r in rs]); ry = np.array([float(r['ref_y']) for r in rs]); rz = np.array([float(r['ref_alt']) for r in rs]); a.plot(rx, ry, rz, 'k--', lw=0.7); first = False
                on = act == 1; pre = (st < ATK0) & ~on; post = (st >= ATK0) & ~on
                a.plot(x[pre], y[pre], z[pre], color='gray', lw=0.8); a.plot(x[on], y[on], z[on], color='r', lw=1.5); a.plot(x[post], y[post], z[post], color='orange', lw=0.9)
                i0 = np.searchsorted(st, 183)
                if i0 < len(st): a.scatter([x[i0]], [y[i0]], [z[i0]], marker='o', color='b', s=18); dmax.append(np.hypot(x[i0:] - x[i0], y[i0:] - y[i0]).max())
                if surv == 0: a.scatter([x[-1]], [y[-1]], [z[-1]], marker='x', color='k', s=50)
            a.set_title(f"{'failsafe' if mode == 'O' else '명목'} δ{dl} {p}  이탈 max {np.median(dmax) if dmax else float('nan'):.1f} m", fontsize=8)
            a.set_xlim(-12, 12); a.set_ylim(-12, 12); a.set_zlim(0, 5); a.tick_params(labelsize=6)
    fig.suptitle(f'dhover3 3D 궤적 ws{ws} — 행: 명목 hover vs failsafe hover (δ0.76, 0.80) · 회색 공격 전 · 빨강 공격 중(3 s) · 주황 후 · 파랑점 = hover 전환 시점 · ×=추락 · 축 ±12 m 고정', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{R}/night/traj3d_failsafe_ws{ws}.png', dpi=85); plt.close(fig); print('saved', ws)
