#!/usr/bin/env python3
"""plot_novelty_curves (09-15 밤, 사용자 요청: 처음 보는 상황에서 SWIRL 이 더 잘하는 것을 곡선으로)
   ① v30+v30b blk50k (시드 42–51, v5d 구 풀): 에피소드 축 보상·행동 오탐·탐욕 프로브 오탐 + SW−Adam 짝 차 95% CI
   ② v33 mkv50k (시드 42–46): 폭풍 진입 시점 정렬 — 1차(처음 보는 강풍) vs 2차(재진입) 오탐·보상"""
import glob, json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 2})
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'
COL = {'SW': '#2a6fdb', 'Adam': '#e0592a', 'UKF': '#7a9a3a', 'EKF': '#9b6bb3'}
NAME = {'SW': 'SWIRL', 'Adam': 'Adam 3e-4', 'UKF': 'UKF-TD', 'EKF': 'EKF-TD'}
TCR = {4: 2.776, 9: 2.262}
def load(pre):
    t = {}
    for f in glob.glob(f'{N}/{pre}gpu_w*.json') + glob.glob(f'{N}/{pre}cpu_w*.json'):
        for k, m in json.load(open(f)).items():
            if 'hist' in m: c, a, s = k.rsplit('_', 2); t[(c, a, int(s[1:]))] = m['hist']
    return t
def arr(h, key):
    return np.array([np.nan if x.get(key) is None else float(x[key]) for x in h], float)
def roll(a, w=5):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        s = a[max(0, i - w + 1):i + 1]; s = s[~np.isnan(s)]; out[i] = s.mean() if len(s) else np.nan
    return out
def band(ax, x, M, col, lab, alpha=0.18, lw=2, ls='-'):
    mu = np.nanmean(M, 0); se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), 0))
    ax.plot(x, mu, color=col, lw=lw, ls=ls, label=lab); ax.fill_between(x, mu - se, mu + se, color=col, alpha=alpha, lw=0)
# ── ① 급변 블록
t = load('v30'); t.update(load('v30b'))
S = [s for s in range(42, 52) if ('blk50k', 'SW', s) in t and ('blk50k', 'Adam', s) in t]
x = np.arange(150)
fig, axs = plt.subplots(4, 1, figsize=(11, 12.5), sharex=True)
panels = [('reward', '에피소드 보상 (이동평균 5)', True), ('fpr', '학습 중 오탐률 (행동정책, 이동평균 5)', True), ('probe_fpr', '탐욕 프로브 오탐률 (탐험 없음, 이동평균 5)', False)]
for ax, (key, title, withkt) in zip(axs[:3], panels):
    for ag in (['UKF', 'EKF'] if withkt else []) + ['Adam', 'SW']:
        ss = [s for s in S if ('blk50k', ag, s) in t]
        if not ss: continue
        M = np.array([roll(arr(t[('blk50k', ag, s)], key)) for s in ss])
        band(ax, x, M, COL[ag], f'{NAME[ag]} (n={len(ss)})', alpha=0.10 if ag in ('UKF', 'EKF') else 0.2, lw=1.3 if ag in ('UKF', 'EKF') else 2.2)
    ax.axvspan(60, 110, color='#8a8f98', alpha=0.10, lw=0); ax.set_title(title, loc='left', fontsize=11)
axs[0].text(61, axs[0].get_ylim()[1], ' 처음 보는 강풍 ws10 (ep60–109)', va='top', fontsize=9, color='#555')
axs[0].legend(ncol=4, fontsize=9, loc='lower right')
D = np.array([roll(arr(t[('blk50k', 'SW', s)], 'reward')) - roll(arr(t[('blk50k', 'Adam', s)], 'reward')) for s in S])
mu = np.nanmean(D, 0); se = np.nanstd(D, 0, ddof=1) / np.sqrt(len(S)); cr = TCR.get(len(S) - 1, 2.26)
axs[3].plot(x, mu, color=COL['SW'], lw=2.2, label='SWIRL − Adam 보상 차 (시드 짝 평균)'); axs[3].fill_between(x, mu - cr * se, mu + cr * se, color=COL['SW'], alpha=0.2, lw=0, label='95% 신뢰구간')
axs[3].axhline(0, color='#333', lw=1); axs[3].axvspan(60, 110, color='#8a8f98', alpha=0.10, lw=0)
axs[3].set_title('SWIRL − Adam 보상 차 (0 위 = SWIRL 우세)', loc='left', fontsize=11); axs[3].legend(fontsize=9, loc='lower right'); axs[3].set_xlabel('에피소드')
fig.suptitle(f'급변 블록 blk50k: 잔잔 → 처음 보는 강풍 → 복귀 (surrogate v30+v30b, 시드 {len(S)}, 구 풀 v5d)', fontsize=13, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{N}/novelty_blk50k_curves.png', dpi=110); plt.close(fig)
pre = lambda M, a, b: float(np.nanmean(M[:, a:b]))
print('blk50k 창별 평균 (SW / Adam):')
for key in ('reward', 'fpr', 'probe_fpr'):
    A = {ag: np.array([arr(t[('blk50k', ag, s)], key) for s in S]) for ag in ('SW', 'Adam')}
    print(f'  {key:9s} 전 40–59 {pre(A["SW"],40,60):.3f}/{pre(A["Adam"],40,60):.3f} · 진입 60–69 {pre(A["SW"],60,70):.3f}/{pre(A["Adam"],60,70):.3f} · 블록 60–109 {pre(A["SW"],60,110):.3f}/{pre(A["Adam"],60,110):.3f} · 후기 110–149 {pre(A["SW"],110,150):.3f}/{pre(A["Adam"],110,150):.3f}')
# ── ② 마르코프 체제: 진입 정렬
t3 = load('v33')
def markov(sd):
    rng = np.random.default_rng(1000 + sd); segs = []; ep = 0; calm = True
    while ep < 150:
        e = min(ep + int(rng.integers(20, 41)) - 1, 149); segs.append((ep, e, calm)); ep = e + 1; calm = not calm
    return [a for a, b, c in segs if not c]
L = np.arange(-10, 20)
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for j, (key, title) in enumerate((('fpr', '학습 중 오탐률'), ('reward', '에피소드 보상'))):
    for i, which in enumerate((0, 1)):
        ax = axs[j, i]
        for ag in ('Adam', 'SW'):
            rows = []
            for s in range(42, 47):
                if ('mkv50k', ag, s) not in t3: continue
                ent = markov(s)
                if len(ent) <= which: continue
                e0 = ent[which]; a = arr(t3[('mkv50k', ag, s)], key)
                rows.append([a[e0 + k] if 0 <= e0 + k < len(a) else np.nan for k in L])
            if rows: band(ax, L, np.array(rows), COL[ag], f'{NAME[ag]} (n={len(rows)})')
        ax.axvline(0, color='#333', lw=1); ax.axvspan(0, 19, color='#8a8f98', alpha=0.10, lw=0)
        ax.set_title(f"{title} — {'1차 진입: 처음 보는 강풍' if which == 0 else '2차 진입: 이미 겪은 강풍'}", loc='left', fontsize=11)
        if j == 1: ax.set_xlabel('폭풍 진입 기준 에피소드 (0 = 진입)')
axs[0, 0].legend(fontsize=9, loc='upper right')
fig.suptitle('마르코프 교대 체제 mkv50k: 진입 시점 정렬 (surrogate v33, 시드 5, 구 풀 v5d)', fontsize=13, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{N}/novelty_mkv50k_entries.png', dpi=110); plt.close(fig)
print('그림', f'{N}/novelty_blk50k_curves.png', f'{N}/novelty_mkv50k_entries.png')
