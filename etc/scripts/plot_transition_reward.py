#!/usr/bin/env python3
"""전환 구간 보상 곡선 (09-15 23:10 사용자 요청): 잔잔 → 강풍 전환(ep60) 전후 ep40–109.
   A: 구 풀 v5d (v30+v30b, 시드 10): SWIRL vs Adam 3e-4
   B: 새 풀 v5e (사슬 S1 g90n3 + tonight blk): SWIRL vs Adam 3e-4 vs Adam 1e-3 — 끝난 시드만"""
import glob, json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'
COL = {'SW': '#2a6fdb', 'Adam3e-4': '#e0592a', 'Adam1e-3': '#b8860b'}; NM = {'SW': 'SWIRL', 'Adam3e-4': 'Adam lr 3e-4', 'Adam1e-3': 'Adam lr 1e-3'}
def roll(x, w=5): return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])
def rew(h): return np.array([float(r['reward']) for r in h], float)
old = {}
for f in glob.glob(f'{N}/v30*_w*.json'):
    for k, m in json.load(open(f)).items():
        if 'hist' in m and k.startswith('blk50k_'): c, a, s = k.rsplit('_', 2); old[(a, int(s[1:]))] = m['hist']
new = {}
for f in glob.glob(f'{N}/chain_hz/S1*_w*.json'):
    for k, m in json.load(open(f)).items():
        if m['cell'] == 'g90n3': new[(m['learner'], m['seed'])] = m['hist']
for f in glob.glob(f'{N}/tonight/T*_w*.json'):
    for k, m in json.load(open(f)).items():
        if m['cell'] == 'blk': new[(m['learner'], m['seed'])] = m['hist']
x = np.arange(40, 110)
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, D, ags, title in ((axs[0], old, ('Adam', 'SW'), 'A. 구 풀 v5d (v30+v30b)'), (axs[1], new, ('Adam1e-3', 'Adam3e-4', 'SW'), 'B. 새 풀 v5e (끝난 시드만)')):
    for ag in ags:
        key = 'Adam3e-4' if ag == 'Adam' else ag
        M = np.array([roll(rew(h))[40:110] for (a, s), h in D.items() if a == ag])
        if not len(M): continue
        mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else 0 * mu
        ax.plot(x, mu, color=COL[key], lw=2.2, label=f'{NM[key]} (n={len(M)})'); ax.fill_between(x, mu - se, mu + se, color=COL[key], alpha=0.18, lw=0)
        lo = int(np.argmin(mu[20:40])) + 60
        ax.annotate(f'{mu[lo - 40]:.0f}', (lo, mu[lo - 40]), textcoords='offset points', xytext=(4, -12), color=COL[key], fontsize=9)
    ax.axvspan(60, 110, color='#8a8f98', alpha=0.10, lw=0); ax.axvline(60, color='#333', lw=1)
    ax.set_title(title, loc='left', fontsize=11); ax.set_xlabel('에피소드 (ep60 = 강풍 시작)'); ax.legend(fontsize=9, loc='lower right')
axs[0].set_ylabel('에피소드 보상 (이동평균 5, 평균 ± 표준오차)')
fig.suptitle('잔잔 바람(무풍 60 % · 7 m/s 40 %) → 강풍 10 m/s 전환 전후 보상', fontsize=13, x=0.01, ha='left')
fig.tight_layout(); out = f'{N}/transition_reward.png'; fig.savefig(out, dpi=110); print(out)
for lab, D, ags in (('구 풀', old, ('SW', 'Adam')), ('새 풀', new, ('SW', 'Adam3e-4', 'Adam1e-3'))):
    for ag in ags:
        H = [h for (a, s), h in D.items() if a == ag]
        if H: print(f'{lab} {ag:9s} n={len(H)} 전환 전 40–59 {np.mean([rew(h)[40:60].mean() for h in H]):.1f} · 진입 60–69 {np.mean([rew(h)[60:70].mean() for h in H]):.1f} · 70–89 {np.mean([rew(h)[70:90].mean() for h in H]):.1f} · 90–109 {np.mean([rew(h)[90:110].mean() for h in H]):.1f}')
