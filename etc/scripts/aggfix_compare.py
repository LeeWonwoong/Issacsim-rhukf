#!/usr/bin/env python3
"""aggfix_compare — aggressive 위상 수정 전(cert_WIND) / 후(aggfix_verify) 무공격·track 비교: 설정점 점프, 추종오차, 기체 경로·주회수, 평시 gyro/vel NIS.
   그림 night/aggfix_compare.png. 수정 후 자료가 없으면 수정 전만 출력."""
import csv, collections, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams['axes.unicode_minus'] = False
R0 = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'; CYC = 2 * np.pi * 2.24
def load(d):
    eps = collections.defaultdict(list); f = f'{R0}/{d}/sweep_detail.csv'
    if not os.path.exists(f): return {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r['pattern'] != 'aggressive' or r['policy'] != 'track' or float(r['bias']) != 0.0: continue
            eps[(r['wind_speed'], r['cell_idx'], r['episode'])].append((int(r['step']), float(r['ref_x']), float(r['ref_y']), float(r['pos_x']), float(r['pos_y']), float(r['nis_g_scaled']), float(r['nis_v_scaled']), r['crash_reason']))
    return {k: np.array(sorted(v), dtype=object) for k, v in eps.items()}
rows = []; picks = {}
print('| 코드 | ws | 에피 | 스텝 중앙 | 점프 >0.5 m 에피당 | 점프 최대 m | 추종오차 중앙/최대 m | 기준점 주회수 | 기체 경로 주회수 | gyro NIS 중앙/95% | vel NIS 중앙/95% | 추락 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for tag, d in (('수정 전', 'cert_WIND'), ('수정 후', 'aggfix_verify')):
    E = load(d); by = collections.defaultdict(list)
    for k, e in E.items(): by[k[0]].append(e)
    for ws in sorted(by, key=float):
        J, JM, EM, EX, RC, PC, G, V, S, CR = [], [], [], [], [], [], [], [], [], 0
        for e in by[ws]:
            st = e[:, 0].astype(int); rx, ry, px, py = (e[:, i].astype(float) for i in (1, 2, 3, 4))
            jump = np.hypot(np.diff(rx), np.diff(ry)); err = np.hypot(rx - px, ry - py)
            J.append(int((jump > 0.5).sum())); JM.append(float(jump.max()) if len(jump) else 0); EM.append(float(np.median(err))); EX.append(float(err.max()))
            RC.append(float(jump[jump <= 0.5].sum()) / CYC); PC.append(float(np.hypot(np.diff(px), np.diff(py)).sum()) / CYC)
            G.extend(e[:, 5].astype(float)); V.extend(e[:, 6].astype(float)); S.append(int(st.max() - st.min() + 1)); CR += int(any(c not in ('', 'timeout', 'none') for c in e[:, 7]))
            picks.setdefault((tag, ws), e)
        print(f"| {tag} | {ws} | {len(by[ws])} | {int(np.median(S))} | {np.mean(J):.1f} | {max(JM):.2f} | {np.median(EM):.2f} / {max(EX):.2f} | {np.mean(RC):.2f} | {np.mean(PC):.2f} | {np.median(G):.2f} / {np.percentile(G, 95):.2f} | {np.median(V):.2f} / {np.percentile(V, 95):.2f} | {CR} |")
keys = [k for k in (('수정 전', '0.0'), ('수정 후', '0.0'), ('수정 전', '10.0'), ('수정 후', '10.0')) if k in picks]
if keys:
    fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, k in zip(axs[0], keys):
        e = picks[k]; st = e[:, 0].astype(int); rx, ry, px, py = (e[:, i].astype(float) for i in (1, 2, 3, 4))
        ax.plot(rx, ry, '--', color='0.55', lw=1.1, label='기준 궤적'); sc = ax.scatter(px, py, c=st, cmap='viridis', s=6, label='실제 위치')
        ax.set_title(f'aggressive {k[0]} · ws{k[1][:-2]}'); ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('x [m]')
    axs[0][0].set_ylabel('y [m]'); axs[0][0].legend(fontsize=8, loc='lower left'); fig.colorbar(sc, ax=axs[0], shrink=0.8, label='RL 스텝')
    out = f'{R0}/night/aggfix_compare.png'; fig.savefig(out, dpi=110, bbox_inches='tight'); print('그림', out)
