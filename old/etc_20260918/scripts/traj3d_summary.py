#!/usr/bin/env python3
"""traj3d_summary — 무풍 δ0.80: 5패턴 세로, 각 행 = [3D: 미대응 track(빨강, × 종료) + failsafe hover(파랑, ● 전환)] + [x·y·z 시간 흐름]. 데이터 failsafe_dh (변조 off)."""
import csv, collections, numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(fp); matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name(); matplotlib.rcParams['axes.unicode_minus'] = False
A = 4.36; ATK0 = 180; D = sys.argv[3] if len(sys.argv) > 3 else 'results/claudecodefortest/failsafe_dh'; PATS = ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve']
DL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8; WS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv')))
for r in summ: r['delta'] = round(float(r['bias']) / A, 2); r['ws'] = int(round(float(r['wind_speed'])))
det = collections.defaultdict(list)
for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
    if int(r['step']) >= 100: det[(int(r['cell_idx']), int(r['episode']))].append(r)
for k in det: det[k].sort(key=lambda r: int(r['step']))
def pick(p, pol):
    cs = [r for r in summ if r['delta'] == DL and r['ws'] == WS and r['policy'] == pol and r['pattern'] == p]
    if pol == 'track': cs.sort(key=lambda r: int(float(r['survived'])))   # 추락 에피 우선
    r = cs[0]; rs = det[(int(r['cell_idx']), int(r['episode']))]
    g = lambda k: np.array([float(x[k]) for x in rs])
    return dict(st=g('step'), x=g('pos_x'), y=g('pos_y'), z=g('alt'), rx=g('ref_x'), ry=g('ref_y'), rz=g('ref_alt'), atk=g('attack_active'), surv=int(float(r['survived'])), crash=int(float(r['crash_step'])), reason=r['crash_reason'])
fig = plt.figure(figsize=(15, 6.2 * len(PATS)))
gs = fig.add_gridspec(4 * len(PATS), 2, width_ratios=[1.45, 1], wspace=0.12, hspace=0.55, top=0.975, bottom=0.02, left=0.03, right=0.98)
for i, p in enumerate(PATS):
    T = pick(p, 'track'); H = pick(p, 'dhover3'); t = lambda Q: (Q['st'] - ATK0) / 10
    a = fig.add_subplot(gs[4 * i:4 * i + 4, 0], projection='3d')
    a.plot(T['rx'], T['ry'], T['rz'], color='k', ls='--', lw=1.0, label='기준 궤적')
    for Q, col, lab in ((T, 'crimson', '미대응 (track)'), (H, 'royalblue', 'failsafe hover')):
        pre = Q['st'] < ATK0; on = Q['atk'] == 1; post = (Q['st'] >= ATK0) & ~on
        a.plot(Q['x'][pre], Q['y'][pre], Q['z'][pre], color='0.6', lw=1.2)
        a.plot(Q['x'][on], Q['y'][on], Q['z'][on], color=col, lw=3.0, label=f'{lab} · 공격 중')
        a.plot(Q['x'][post], Q['y'][post], Q['z'][post], color=col, lw=1.4, alpha=.55)
    i0 = np.searchsorted(H['st'], 183); a.scatter([H['x'][i0]], [H['y'][i0]], [H['z'][i0]], color='royalblue', s=90, marker='o', zorder=5, label='hover 전환 (+0.3 s)')
    if T['surv'] == 0: a.scatter([T['x'][-1]], [T['y'][-1]], [T['z'][-1]], color='crimson', s=160, marker='x', linewidths=3, zorder=6, label=f"종료 {T['reason'][6:]} @{(T['crash']-ATK0)/10:.1f} s")
    a.set_xlabel('x [m]', labelpad=8); a.set_ylabel('y [m]', labelpad=8); a.set_zlabel('alt [m]'); a.set_zlim(0, 4); a.tick_params(labelsize=9)
    a.set_title(f'{p}  ·  δ{DL}  ·  {"무풍" if WS == 0 else f"ws{WS}"}', fontsize=15, fontweight='bold', loc='left', pad=2)
    a.legend(fontsize=9.5, loc='upper left', bbox_to_anchor=(-0.02, 0.98)); a.view_init(elev=24, azim=-55)
    eT = np.hypot(T['x'] - T['rx'], T['y'] - T['ry']); eH = np.hypot(H['x'] - H['x'][i0], H['y'] - H['y'][i0])
    panels = (('x', 'rx', 'x [m]'), ('y', 'ry', 'y [m]'), ('z', 'rz', 'alt [m]'), ('err', None, '수평 오차 [m]'))
    for j, (key, rkey, lab) in enumerate(panels):
        b = fig.add_subplot(gs[4 * i + j, 1])
        if key == 'err':
            b.plot(t(T), eT, color='crimson', lw=1.8, label='미대응: 기준궤적 오차'); b.plot(t(H)[i0:], eH[i0:], color='royalblue', lw=1.8, label='hover: 앵커 이탈')
            b.axhline(10, color='k', lw=1.0, ls='--'); b.text(-5.8, 10.4, '지오펜스 10 m (1 s 지속 → 종료)', fontsize=8)
            if T['surv'] == 0: b.scatter([(T['crash'] - ATK0) / 10], [eT[-1]], color='crimson', marker='x', s=80, linewidths=2, zorder=5)
            b.set_ylim(0, max(12, eT.max() * 1.08)); b.text(9.8, 0.5, f"hover 최대 {eH[i0:].max():.1f} m · 3 s 후 {eH[min(np.searchsorted(H['st'], 209), len(eH)-1)]:.1f} m", ha='right', va='bottom', fontsize=8.5, color='royalblue', bbox=dict(fc='white', ec='none', alpha=.85))
            if i == 0: b.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 0.92))
        else:
            b.plot(t(T), T[rkey], color='k', ls='--', lw=0.8); b.plot(t(T), T[key], color='crimson', lw=1.8, label='미대응'); b.plot(t(H), H[key], color='royalblue', lw=1.8, label='failsafe hover')
            if T['surv'] == 0: b.scatter([(T['crash'] - ATK0) / 10], [T[key][-1]], color='crimson', marker='x', s=80, linewidths=2, zorder=5)
            if key == 'z': b.set_ylim(0, 4)
            if i == 0 and j == 0: b.legend(fontsize=8, loc='upper left', ncol=2)
        b.axvspan(0, 3, color='r', alpha=.10); b.axvline(0.3, color='royalblue', lw=0.8, ls=':')
        b.set_xlim(-6, 10); b.grid(alpha=.3); b.set_ylabel(lab, fontsize=9); b.tick_params(labelsize=8)
        if j < 3: b.tick_params(labelbottom=False)
        else: b.set_xlabel('공격 온셋 기준 시간 [s]', fontsize=9)
fig.suptitle(f'무대응 vs failsafe hover — δ{DL} · {"무풍" if WS == 0 else f"ws{WS}"} · 공격 3 s (빨간 띠) · 회색 = 공격 전 · 굵은 선 = 공격 중 · 흐린 선 = 공격 후 · × = 종료(지오펜스 10 m / flip) · ● = hover 전환', fontsize=13, y=0.992)
out = f'results/claudecodefortest/night/traj3d_summary_d{DL}_ws{WS}.png'; fig.savefig(out, dpi=95); print('saved', out)
