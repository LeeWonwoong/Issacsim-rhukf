#!/usr/bin/env python3
"""scripts/plot_compare.py — 설정 묶음별 학습곡선(시드 평균 ± 최소/최대): 보상(생존 뺀 cost/c)·loss(최고점 정규화)·공격 에피 F1.

    python3 scripts/plot_compare.py <출력.png> "<이름>=<폴더 glob>" ["<이름>=<폴더 glob>" ...]
예) python3 scripts/plot_compare.py out.png "SWIRL=results/.../night2/a_sActB256_batch256_s4?" "Adam=results/.../night2/a_aB256_batch256_s4?"
"""
import glob
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NotoSerifCJK' in f:
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
COL = ['#2f62e6', '#d1483a', '#1f9d63', '#b9791a', '#6a3fd6', '#0e8f8f']


def smooth(x, w=10):
    m = ~np.isnan(x); k = np.ones(w)
    num = np.convolve(np.where(m, x, 0.0), k, mode='valid'); den = np.convolve(m.astype(float), k, mode='valid')
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def main():
    out = sys.argv[1]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, spec in enumerate(sys.argv[2:]):
        name, pat = spec.rsplit('=', 1)
        C, L, F = [], [], []
        for d in sorted(glob.glob(pat)):
            try:
                H = json.load(open(d + '/hist.json'))['hist']
            except FileNotFoundError:
                continue
            c = float(yaml.safe_load(open(d + '/config.yaml'))['reward'].get('scale', 1))
            C.append(smooth(np.array([h['reward_cost'] for h in H]) / c))
            l = np.array([h['loss'] for h in H]); L.append(smooth(l / l.max()))
            F.append(smooth(np.array([h['f1'] if h['has_atk'] else np.nan for h in H], float), 20))
        if not C:
            continue
        col = COL[i % len(COL)]
        for a, Y in zip(ax, (C, L, F)):
            Y = np.array(Y); m = np.nanmean(Y, 0); x = np.arange(len(m))
            a.plot(x, m, color=col, lw=2, label=f'{name} (n={len(Y)})')
            a.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=col, alpha=0.15, lw=0)
    ax[0].set_title('보상 (cost ÷ c, 10ep 평균)'); ax[1].set_title('loss (런별 최고점=1, 10ep 평균)'); ax[2].set_title('공격 에피소드 F1 (20ep 평균)')
    for a in ax:
        a.set_xlabel('에피소드'); a.grid(alpha=0.3)
    ax[0].legend(loc='lower right', fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main()
