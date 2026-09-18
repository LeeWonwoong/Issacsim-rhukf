#!/usr/bin/env python3
"""fig4_nis_windows (09-16, 논문 그림 4): 탐지 비활성 대본 시간창에서 압축 NIS 관측.
   2행 1열 — (a) circle, (b) aggressive. 각 패널에 δ {0, 0.1, 0.4, 0.8} 의 gyro·vel 압축 NIS, 채널별 단색 농도 구배, y 0–4 고정.
   구간: 기저 0–60 · 바람만 60–150 · 겹침 150–180 · 공격만 180–260 · 기저 260–300.
   원자료: results/claudecodefortest/fig4_nis/sweep_detail.csv (capture_fig4_nis.sh). 출력 night/fig4_nis_windows.{png,pdf}"""
import csv, collections, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9})
R = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'
SRC = sys.argv[1] if len(sys.argv) > 1 else f'{R}/fig4_nis/sweep_detail.csv'
DELTA = [('0.000', 0.0), ('0.436', 0.1), ('1.744', 0.4), ('3.488', 0.8)]   # bias[N·m] → δ = bias/4.36
SEG = [(0, 60, '기저'), (60, 150, '바람만'), (150, 180, '겹침'), (180, 260, '공격만'), (260, 300, '기저')]
GY = plt.cm.Oranges(np.linspace(0.42, 0.95, len(DELTA))); VE = plt.cm.Blues(np.linspace(0.42, 0.95, len(DELTA)))
D = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(SRC)):
    D[(r['pattern'], r['bias'])][int(r['step'])].append((float(r['nis_g_scaled']), float(r['nis_v_scaled'])))
fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
for ax, pat, lab in ((axes[0], 'circle', '(a) circle'), (axes[1], 'aggressive', '(b) aggressive')):
    for a, b, name in SEG:
        ax.axvspan(a, b, color='#f2f2f2' if name == '기저' else ('#e8f1fb' if name == '바람만' else ('#fdebe7' if name == '공격만' else '#efe6f7')), lw=0, zorder=0)
        ax.text((a + b) / 2, 3.88, name, ha='center', va='top', fontsize=7.5, color='#555555')
    for i, (bias, d) in enumerate(DELTA):
        st = sorted(D[(pat, bias)])
        if not st: continue
        g = np.array([np.mean([v[0] for v in D[(pat, bias)][s]]) for s in st])
        v = np.array([np.mean([v[1] for v in D[(pat, bias)][s]]) for s in st])
        k = 5; sm = lambda y: np.convolve(y, np.ones(k) / k, 'same')
        ax.plot(st, sm(g), color=GY[i], lw=1.5, label=f'gyro δ={d:.1f}')
        ax.plot(st, sm(v), color=VE[i], lw=1.5, ls='-', label=f'vel δ={d:.1f}')
    ax.set_ylim(0, 4); ax.set_xlim(0, 300); ax.set_ylabel('압축 NIS')
    ax.set_title(lab, fontsize=10, loc='left', pad=2)
    ax.grid(alpha=0.25, lw=0.5); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes[1].set_xlabel('스텝 (10 Hz, 30 s)')
h, l = axes[0].get_legend_handles_labels()
axes[0].legend(h, l, ncol=4, fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.02))
fig.suptitle('탐지 비활성 대본 창 — 강풍 티어 ws10, 틸트 공격 δ 스윕', fontsize=11, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ('png', 'pdf'): fig.savefig(f'{R}/night/fig4_nis_windows.{ext}', dpi=200 if ext == 'png' else None, bbox_inches='tight')
print('저장: night/fig4_nis_windows.png / .pdf')
