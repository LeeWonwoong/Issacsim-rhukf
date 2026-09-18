#!/usr/bin/env python3
"""fig_final1 (09-16): 확정 무대 1시드 보상 곡선.
   무대 = 바람 점진 확대(0–59 잔잔 / 60–119 중간 섞임 / 120–199 강풍 포함) · 다중 버스트 3 · 보상 TP+1/TN+0.5/FP·FN −2.0(상수)
          · 확약 없음 · 1스텝 TD · 200판. 출력 night/fig_final1_curves.{png,pdf}"""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'; H = {}
ARM = os.environ.get('ARM', 'ns')
PAT = 'final1_*' if ARM == 'ns' else 'final1s_*'
for f in glob.glob(f'{N}/tonight/q/{PAT}.json'):
    for k, m in json.load(open(f)).items(): H[m['learner']] = m['hist']
NM = {'SW': 'SWIRL', 'UKFp03': 'UKF-TD', 'EKFp03': 'EKF-TD', 'Adam3e-4': 'Adam 3e-4', 'Adam1e-3': 'Adam 1e-3', 'SGD1e-3': 'SGD 1e-3', 'SGD1e-2': 'SGD 1e-2'}
CO = {'SW': '#1f4e9c', 'UKFp03': '#6b8e23', 'EKFp03': '#8e5fa8', 'Adam3e-4': '#d1495b', 'Adam1e-3': '#e08a95', 'SGD1e-3': '#8a8a8a', 'SGD1e-2': '#c0c0c0'}
ORDER = ['SW', 'UKFp03', 'EKFp03', 'Adam3e-4', 'Adam1e-3', 'SGD1e-3', 'SGD1e-2']
def mv(y, k=10):
    o = np.full(len(y), np.nan)
    for i in range(len(y)): o[i] = np.nanmean(y[max(0, i - k + 1):i + 1])
    return o
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={'width_ratios': [2, 1]})
ax = axes[0]
SEGL = ((0, '잔잔 (ws0)', '#f4f4f4'), (60, '중간 바람 섞임', '#eaf2fb'), (120, '강풍 포함', '#fdecea')) if ARM == 'ns' else ((0, '고정 혼합 (동일 주변분포)', '#f4f4f4'),)
for seg, lab, col in SEGL:
    end = ({0: 60, 60: 120, 120: 200} if ARM == 'ns' else {0: 200})[seg]
    ax.axvspan(seg, end, color=col, lw=0, zorder=0)
    ax.text((seg + end) / 2, 0.99, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.5, color='#555')
rows = []
for l in ORDER:
    if l not in H: continue
    y = np.array([r['reward'] for r in H[l]], float)
    ax.plot(np.arange(len(y)), mv(y), color=CO[l], lw=1.9 if l == 'SW' else 1.3, label=NM[l], zorder=3 if l == 'SW' else 2)
    rows.append((l, np.nanmean(y[:60]), np.nanmean(y[60:120]), np.nanmean(y[120:]), np.nanmean(y[60:70]) - np.nanmean(y[40:60]), np.nanmean(y[120:130]) - np.nanmean(y[100:120])))
ax.set_xlabel('학습 에피소드'); ax.set_ylabel('판당 보상 (10판 이동평균)'); ax.set_xlim(0, len(y) - 1)
ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False, loc='lower right')
ax.set_title(('비정상(점진 확대)' if ARM == 'ns' else '정상 대조(고정 혼합)') + ' · 시드 42', fontsize=10, loc='left')
ax2 = axes[1]; xs = np.arange(len(rows)); w = 0.38
ax2.bar(xs - w / 2, [r[4] for r in rows], w, color=[CO[r[0]] for r in rows], label='중간 바람 진입 낙폭')
ax2.bar(xs + w / 2, [r[5] for r in rows], w, color=[CO[r[0]] for r in rows], alpha=0.5, label='강풍 진입 낙폭')
ax2.axhline(0, color='#333', lw=0.8); ax2.set_xticks(xs); ax2.set_xticklabels([NM[r[0]] for r in rows], rotation=35, ha='right', fontsize=7.5)
ax2.set_ylabel('진입 10판 − 직전 20판'); ax2.grid(axis='y', alpha=0.25, lw=0.5); ax2.legend(fontsize=7.5, frameon=False)
ax2.set_title('전환 낙폭', fontsize=10, loc='left')
fig.suptitle('확정 무대 1시드 확인 — 바람 점진 확대 · 다중 버스트 3 · 보상 FP·FN −2.0 상수 · 확약 없음 · 1스텝 TD · 200판', fontsize=10.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for e in ('png', 'pdf'): fig.savefig(f'{N}/fig_final1_curves_{ARM}.{e}', dpi=200 if e == 'png' else None, bbox_inches='tight')
print('저장: night/fig_final1_curves.png / .pdf')
print('| 학습기 | 잔잔 0–59 | 중간 60–119 | 강풍 120–199 | 중간 진입 낙폭 | 강풍 진입 낙폭 |')
for r in rows: print(f'| {NM[r[0]]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:+.2f} | {r[5]:+.2f} |')
