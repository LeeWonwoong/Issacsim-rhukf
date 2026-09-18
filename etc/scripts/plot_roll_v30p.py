#!/usr/bin/env python3
"""plot_roll_v30p (09-15) — 학습 정책 greedy 롤아웃(roll_v30p_{learner}) 시각화.
   그림 A(학습기별·패턴별): 열 = 조건(δ0/0.05/0.10/0.80 × ws0/10), 행 = ①위에서 본 궤적(실제·기준, hover 구간 강조) ②gyro·vel NIS(공격창·바람창 음영) ③행동(track/hover)과 추종오차.
   그림 B(학습기 비교): 조건×학습기 행동 띠(에피 1, circle/aggressive).  표: 온셋 후 첫 hover 지연, 온셋 전 hover 스텝(오경보), 생존, 최대 이탈."""
import csv, glob, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f and 'Regular' in f: matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name(); break
matplotlib.rcParams['axes.unicode_minus'] = False
R = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'; OUT = f'{R}/night'
A = 4.36; ONSET, OFF, W0, W1 = 180, 210, 60, 290
LEARN = [l for l in ('swirl', 'adam', 'ukf', 'ekf') if os.path.exists(f'{R}/roll_v30p_{l}/sweep_detail.csv')]
COND = [(0.0, '0.0'), (0.218, '0.0'), (0.436, '0.0'), (3.488, '0.0'), (0.0, '10.0'), (0.218, '10.0'), (0.436, '10.0'), (3.488, '10.0')]
def lab(b, ws): return f"δ{b/A:.2f} ws{ws[:-2]}"
def load(l):
    S = list(csv.DictReader(open(f'{R}/roll_v30p_{l}/sweep_summary.csv'))); D = list(csv.DictReader(open(f'{R}/roll_v30p_{l}/sweep_detail.csv')))
    by = {}
    for x in D: by.setdefault((x['cell_idx'], x['episode']), []).append(x)
    return S, by
rows = []
for l in LEARN:
    S, by = load(l)
    for pat in ('circle', 'aggressive'):
        fig, axs = plt.subplots(3, len(COND), figsize=(3.1 * len(COND), 9.2), gridspec_kw=dict(height_ratios=[1.25, 1, 0.8]))
        for j, (b, ws) in enumerate(COND):
            cand = [r for r in S if abs(float(r['bias']) - b) < 1e-3 and r['wind_speed'] == ws and r['pattern'] == pat]
            for r in cand:
                xs = by.get((r['cell_idx'], r['episode']), [])
                if not xs: continue
                st = np.array([int(x['step']) for x in xs]); act = np.array([int(x['action']) for x in xs])
                px = np.array([float(x['pos_x']) for x in xs]); py = np.array([float(x['pos_y']) for x in xs])
                rx = np.array([float(x['ref_x']) for x in xs]); ry = np.array([float(x['ref_y']) for x in xs])
                g = np.array([float(x['nis_g_scaled']) for x in xs]); v = np.array([float(x['nis_v_scaled']) for x in xs])
                err = np.hypot(px - rx, py - ry)
                pre = act[(st >= 20) & (st < ONSET)]; post = np.where((st >= ONSET) & (act == 1))[0]
                d1 = int(st[post[0]] - ONSET) if len(post) else -1
                anc = np.where((st >= ONSET) & (act == 1))[0]; drift = float(np.max(np.hypot(px[anc] - px[anc[0]], py[anc] - py[anc[0]]))) if len(anc) else 0.0
                rows.append(dict(learner=l, pattern=pat, cond=lab(b, ws), ep=r['episode'], survived=r['survived'], reason=r['crash_reason'],
                                 delay=d1, fp_hover_pre=int(pre.sum()), hover_frac_after=float(act[st >= ONSET].mean()) if (st >= ONSET).any() else 0.0,
                                 max_err=float(err.max()), hover_drift=drift))
                if r['episode'] != min(c['episode'] for c in cand): continue
                ax = axs[0, j]; ax.plot(rx, ry, 'k--', lw=0.8, label='기준'); ax.plot(px, py, color='0.55', lw=1.0, label='실제')
                h = act == 1; ax.scatter(px[h], py[h], s=5, color='tab:red', label='hover', zorder=3)
                ax.scatter([px[st >= ONSET][0]] if (st >= ONSET).any() else [], [py[st >= ONSET][0]] if (st >= ONSET).any() else [], marker='x', color='tab:purple', s=40, zorder=4, label='공격 온셋')
                ax.set_title(f"{lab(b, ws)} ({'생존' if r['survived'] == '1' else r['crash_reason']})", fontsize=9); ax.set_aspect('equal', 'datalim'); ax.tick_params(labelsize=7)
                ax = axs[1, j]; ax.axvspan(W0, W1, color='tab:blue', alpha=0.05) if ws != '0.0' else None
                ax.axvspan(ONSET, OFF, color='tab:purple', alpha=0.12) if b > 0 else None
                ax.plot(st, g, color='tab:red', lw=0.9, label='gyro NIS'); ax.plot(st, v, color='tab:blue', lw=0.9, label='vel NIS'); ax.set_ylim(0, 4.1); ax.tick_params(labelsize=7)
                ax = axs[2, j]; ax.step(st, act, where='post', color='tab:red', lw=1.0); ax.set_ylim(-0.1, 1.15); ax.set_yticks([0, 1]); ax.set_yticklabels(['track', 'hover'], fontsize=7)
                ax.axvspan(ONSET, OFF, color='tab:purple', alpha=0.12) if b > 0 else None
                ax2 = ax.twinx(); ax2.plot(st, err, color='0.5', lw=0.8); ax2.set_ylim(0, 12); ax2.tick_params(labelsize=7); ax.set_xlabel('step (10 Hz)', fontsize=8)
                if j == len(COND) - 1: ax2.set_ylabel('|pos−ref| m', fontsize=8)
        axs[0, 0].legend(fontsize=7, loc='best'); axs[1, 0].legend(fontsize=7, loc='upper left'); axs[0, 0].set_ylabel('E [m]'); axs[1, 0].set_ylabel('scaled NIS'); axs[2, 0].set_ylabel('행동')
        fig.suptitle(f'{l.upper()} 학습 정책(isaac_v30p final, 시드 42) greedy 롤아웃 — {pat}. 보라 음영 = 공격창(180–210), 파랑 음영 = 바람창(60–290), 빨강 점 = hover', fontsize=12)
        plt.tight_layout(); fn = f'{OUT}/roll_{l}_{pat}.png'; plt.savefig(fn, dpi=95); plt.close(fig); print('saved', fn)
# 그림 B: 학습기×조건 행동 띠
if LEARN:
    for pat in ('circle', 'aggressive'):
        fig, axs = plt.subplots(len(COND), 1, figsize=(13, 1.0 * len(COND) + 1), sharex=True)
        for j, (b, ws) in enumerate(COND):
            ax = axs[j]; M = []
            for l in LEARN:
                S, by = load(l); cand = [r for r in S if abs(float(r['bias']) - b) < 1e-3 and r['wind_speed'] == ws and r['pattern'] == pat]
                cand = [c for c in cand if c['episode'] == min(x['episode'] for x in cand)] if cand else cand
                a = np.full(301, np.nan)
                if cand:
                    for x in by.get((cand[0]['cell_idx'], cand[0]['episode']), []): a[int(x['step'])] = int(x['action'])
                M.append(a)
            ax.imshow(np.array(M), aspect='auto', cmap='coolwarm', vmin=0, vmax=1, interpolation='nearest', extent=[0, 301, len(LEARN) - 0.5, -0.5])
            ax.set_yticks(range(len(LEARN))); ax.set_yticklabels(LEARN, fontsize=8); ax.set_ylabel(lab(b, ws), rotation=0, ha='right', fontsize=9, va='center')
            if b > 0: ax.axvline(ONSET, color='k', lw=1); ax.axvline(OFF, color='k', lw=1, ls=':')
            if ws != '0.0': ax.axvline(W0, color='tab:blue', lw=0.8, ls='--')
        axs[-1].set_xlabel('step — 빨강 = hover, 파랑 = track, 검은 실선/점선 = 공격 온셋/종료, 파란 점선 = 바람 시작')
        fig.suptitle(f'학습기별 행동 비교 (셀 첫 에피, {pat})', fontsize=12); plt.tight_layout(); fn = f'{OUT}/roll_actions_{pat}.png'; plt.savefig(fn, dpi=100); plt.close(fig); print('saved', fn)
with open(f'{OUT}/roll_v30p_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['learner']); w.writeheader(); [w.writerow(r) for r in rows]
print(f'rows {len(rows)} → {OUT}/roll_v30p_summary.csv')
for l in LEARN:
    for c in [lab(b, ws) for b, ws in COND]:
        rr = [r for r in rows if r['learner'] == l and r['cond'] == c]
        if rr: print(f"{l:5s} {c:13s} n={len(rr)} 생존 {sum(r['survived']=='1' for r in rr)}/{len(rr)} | 첫 hover 지연 {[r['delay'] for r in rr]} | 온셋 전 hover 스텝 {[r['fp_hover_pre'] for r in rr]} | 온셋 후 hover 비율 {np.mean([r['hover_frac_after'] for r in rr]):.2f} | 최대 이탈 {max(r['max_err'] for r in rr):.1f} m")
