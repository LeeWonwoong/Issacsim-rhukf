#!/usr/bin/env python3
"""fig_traj_3d (09-16, 논문 그림): circle · 무풍(ws0) · δ0.80(bias 3.488) 공격에서
   (a) 탐지 비활성(track 고정) → 표류·운용공간 이탈,  (b) 학습된 검출기(SWIRL) → 온셋 직후 호버 유지.
   원자료: cert_WIND(track) · roll_v30p_swirl(model). 점선 = 기준 궤적, 점선 육면체 = 운용공간 10×10×4 m.
   출력: night/fig_traj_3d.{png,pdf}"""
import csv, collections, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9})
R = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'
FENCE = dict(x=(-5, 5), y=(-5, 5), z=(0, 4))     # 사용자 확정 운용공간 10×10×4 m
def load(path, pat, pol, bias, ws, ep='0'):
    rows = []
    for r in csv.DictReader(open(path)):
        if r['pattern'] == pat and r['policy'] == pol and r['bias'] == bias and r['wind_speed'] == ws and r['episode'] == ep: rows.append(r)
    rows.sort(key=lambda x: int(x['step']))
    st = np.array([int(x['step']) for x in rows])
    pos = np.array([[float(x['pos_x']), float(x['pos_y']), float(x['alt'])] for x in rows])
    ref = np.array([[float(x['ref_x']), float(x['ref_y']), float(x['ref_alt'])] for x in rows])
    act = np.array([int(float(x['action'])) for x in rows])
    atk = np.array([x['attack_active'] in ('1', 'True', 'true') for x in rows])
    return st, pos, ref, act, atk, rows[-1]['crash_reason']
A = load(f'{R}/cert_WIND/sweep_detail.csv', 'circle', 'track', '3.488', '0.0')
B = load(f'{R}/roll_v30p_swirl/sweep_detail.csv', 'circle', 'model', '3.488', '0.0')
def box(ax):
    (x0, x1), (y0, y1), (z0, z1) = FENCE['x'], FENCE['y'], FENCE['z']
    E = [((x0,x1),(y0,y0),(z0,z0)), ((x0,x1),(y1,y1),(z0,z0)), ((x0,x0),(y0,y1),(z0,z0)), ((x1,x1),(y0,y1),(z0,z0)),
         ((x0,x1),(y0,y0),(z1,z1)), ((x0,x1),(y1,y1),(z1,z1)), ((x0,x0),(y0,y1),(z1,z1)), ((x1,x1),(y0,y1),(z1,z1)),
         ((x0,x0),(y0,y0),(z0,z1)), ((x1,x1),(y0,y0),(z0,z1)), ((x0,x0),(y1,y1),(z0,z1)), ((x1,x1),(y1,y1),(z0,z1))]
    for xs, ys, zs in E: ax.plot(xs, ys, zs, ls='--', lw=0.8, color='#888888', alpha=0.9, zorder=1)
lim = np.vstack([A[1][:, :2], B[1][:, :2]])
XL = (min(lim[:, 0].min(), FENCE['x'][0]) - 1.5, max(lim[:, 0].max(), FENCE['x'][1]) + 1.5)
YL = (min(lim[:, 1].min(), FENCE['y'][0]) - 1.5, max(lim[:, 1].max(), FENCE['y'][1]) + 1.5)
fig = plt.figure(figsize=(11.4, 4.9))
for i, (st, pos, ref, act, atk, reason, title) in enumerate([
        A + ('(a) 탐지 비활성 — 공격 대응 없음',), B + ('(b) 학습된 검출기(SWIRL) — 온셋 직후 호버',)]):
    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    box(ax)
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], ls=':', lw=1.6, color='#333333', label='기준 궤적', zorder=2)
    pre = ~atk
    ax.plot(pos[pre, 0], pos[pre, 1], pos[pre, 2], lw=1.6, color='#3b4a56', label='공격 전 추종', zorder=3)
    post_t = atk & (act == 0); post_h = atk & (act == 1)
    seg = lambda m, c, lab: ax.plot(np.where(m, pos[:, 0], np.nan), np.where(m, pos[:, 1], np.nan), np.where(m, pos[:, 2], np.nan), lw=2.0, color=c, label=lab, zorder=4)
    if post_t.any(): seg(post_t, '#c62828', '공격 중 track')
    if post_h.any(): seg(post_h, '#1565c0', '공격 중 hover')
    k = int(np.argmax(atk))
    ax.scatter(*pos[k], s=60, marker='*', color='#000000', depthshade=False, zorder=6, label=f'공격 온셋 (t={st[k]/10:.0f} s)')
    if post_h.any():
        h0 = int(np.argmax(post_h)); ax.scatter(*pos[h0], s=45, marker='^', color='#1565c0', depthshade=False, zorder=6, label=f'호버 전환 (+{(st[h0]-st[k])/10:.1f} s)')
    ax.scatter(*pos[-1], s=55, marker='X' if reason != 'timeout' else 'o', color='#c62828' if reason != 'timeout' else '#2e7d32',
               depthshade=False, zorder=6, label='이탈 종료' if reason != 'timeout' else '정상 종료')
    ax.plot(pos[:, 0], pos[:, 1], np.zeros_like(pos[:, 2]), lw=1.0, color='#bbbbbb', zorder=1)   # 바닥 투영(표류 크기 가독)
    ax.plot(ref[:, 0], ref[:, 1], np.zeros_like(ref[:, 2]), ls=':', lw=0.9, color='#cccccc', zorder=1)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_zlim(0, 4.5); ax.set_box_aspect((1, 1, 0.34))
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('고도 [m]')
    ax.set_title(title, fontsize=10, pad=2); ax.view_init(elev=22, azim=-58)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_._axinfo['grid'].update(color='#dddddd', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=6.8, frameon=False, bbox_to_anchor=(0.02, 0.98), handlelength=1.4, labelspacing=0.32)
fig.suptitle('circle 궤적 · 무풍 · 틸트 공격 δ0.80 — 점선 육면체 = 운용공간 10×10×4 m', fontsize=11, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ('png', 'pdf'): fig.savefig(f'{R}/night/fig_traj_3d.{ext}', dpi=200 if ext == 'png' else None, bbox_inches='tight')
print('저장: night/fig_traj_3d.png / .pdf')
print(f"(a) 스텝 {len(A[0])} 종료 {A[5]} 최대 |xy| {np.linalg.norm(A[1][:, :2], axis=1).max():.1f} m")
print(f"(b) 스텝 {len(B[0])} 종료 {B[5]} 최대 |xy| {np.linalg.norm(B[1][:, :2], axis=1).max():.1f} m 호버 스텝 {(B[3] == 1).sum()}")
