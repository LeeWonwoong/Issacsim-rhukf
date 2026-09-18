#!/usr/bin/env python3
"""traj_laps_check (09-15 저녁, 사용자 질문: 30초 안에 패턴을 몇 번 도나) — cert_WIND 무공격·무풍·track 에피소드의 ref/pos 로
   패턴별 실제 주회수·스텝당 진행을 재고 설계값과 비교. 그림 night/traj_laps_ws0.png, 표 stdout."""
import csv, collections, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams['axes.unicode_minus'] = False
R0 = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'
R, W = 3.4, 0.38                     # swrl_config flight_radius, flight_omega (SPEED_SCALE 1.0)
Rf = R * 0.70710678; Ra, Tp = 2.24, 5.6; Rs, Na = 2.0, 3; kk = 3.2 / 5.0
def fig8_len():
    t = np.linspace(0, 2 * np.pi / W, 20001); x = Rf * np.sin(W * t); y = Rf / 2 * np.sin(2 * W * t)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))
wps = np.array([[0, 0], [5, 0], [5, 5], [-5, 5], [-5, 0], [0, 0]], float) * kk
CYCLE = {'circle': (2 * np.pi * R, 2 * np.pi / W), 'figure8': (fig8_len(), 2 * np.pi / W),
         'scurve': (2 * np.pi * Rs * Na, 2 * np.pi * Rs * Na / (R * W)),
         'aggressive': (2 * np.pi * Ra, 4 * Tp), 'waypoint': (float(np.linalg.norm(np.diff(wps, axis=0), axis=1).sum()), None)}
eps = collections.defaultdict(list)
with open(f'{R0}/cert_WIND/sweep_detail.csv') as fh:
    for row in csv.DictReader(fh):
        if row['bias'] != '0.000' or row['wind_speed'] != '0.0' or row['policy'] != 'track': continue
        eps[(row['pattern'], row['cell_idx'], row['episode'])].append((int(row['step']), float(row['ref_x']), float(row['ref_y']), float(row['pos_x']), float(row['pos_y']), row['crash_reason']))
print('| 패턴 | 에피 수 | 스텝 수(중앙) | 한 사이클 길이 m | 설계 주기 s | 실측 기준점 경로 m | 실측 주회수 | 설계 30 s 주회수 | 원: 스텝당 각 rad(설계 0.038) |')
print('|---|---|---|---|---|---|---|---|---|')
pick = {}
for pat in ('waypoint', 'circle', 'figure8', 'aggressive', 'scurve'):
    ks = [k for k in eps if k[0] == pat]; L, S, AN = [], [], []
    for k in ks:
        e = sorted(eps[k]); st = np.array([r[0] for r in e]); rx = np.array([r[1] for r in e]); ry = np.array([r[2] for r in e])
        if st.max() < 280: continue
        L.append(float(np.sum(np.hypot(np.diff(rx), np.diff(ry))))); S.append(int(st.max() - st.min() + 1))
        if pat == 'circle':
            ang = np.unwrap(np.arctan2(ry, rx + R)); AN.append(float(np.median(np.diff(ang))))
        pick.setdefault(pat, e)
    cl, per = CYCLE[pat]; Lm = float(np.median(L)) if L else float('nan')
    des = (30.0 / per) if per else (30.0 * 1.3 / cl)
    an = f"{np.median(AN):.4f}" if AN else '—'
    print(f"| {pat} | {len(L)} | {int(np.median(S)) if S else 0} | {cl:.1f} | {per if per else float('nan'):.1f} | {Lm:.1f} | {Lm / cl:.2f} | {des:.2f} | {an} |")
fig, axs = plt.subplots(1, 5, figsize=(22, 4.8))
for ax, pat in zip(axs, ('waypoint', 'circle', 'figure8', 'aggressive', 'scurve')):
    e = pick.get(pat)
    if not e: ax.set_title(f'{pat}: 자료 없음'); continue
    st = np.array([r[0] for r in e]); rx = np.array([r[1] for r in e]); ry = np.array([r[2] for r in e]); px = np.array([r[3] for r in e]); py = np.array([r[4] for r in e])
    ax.plot(rx, ry, '--', color='0.55', lw=1.2, label='기준 궤적(ref)')
    sc = ax.scatter(px, py, c=st, cmap='viridis', s=6, label='실제 위치(색=스텝)')
    ax.plot(px[0], py[0], 'o', color='#d62728', ms=7, label='시작'); ax.plot(px[-1], py[-1], 's', color='#1f77b4', ms=7, label='끝')
    Lr = float(np.sum(np.hypot(np.diff(rx), np.diff(ry))))
    ax.set_title(f'{pat} — {st.max() - st.min() + 1} 스텝, 기준점 {Lr / CYCLE[pat][0]:.2f} 사이클'); ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('x [m]')
axs[0].set_ylabel('y [m]'); axs[0].legend(loc='lower left', fontsize=8); fig.colorbar(sc, ax=axs, shrink=0.8, label='RL 스텝(10 Hz)')
fig.suptitle('무공격·무풍(ws0)·track 한 에피소드의 패턴별 궤적 (cert_WIND, EP_MAX_STEPS 300, SPEED_MOD 0, SPEED_SCALE 1.0)')
out = f'{R0}/night/traj_laps_ws0.png'; fig.savefig(out, dpi=110, bbox_inches='tight'); print('그림', out)
