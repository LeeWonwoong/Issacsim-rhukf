#!/usr/bin/env python3
"""plot_pattern_timeseries.py — 패턴별 한 주행의 timestep 시계열 (2026-08-13)

    ~/isaacsim/python.sh plot_pattern_timeseries.py <outdir> [-o out.png]

무엇을 그리나 (패턴 하나당 한 행, 4열)
    ① 속도      |v_xy| 와 v_z          — 설계 접선속도·클램프 상한 대비 어디에 있나
    ② 각속도    |ω|                     — 기동 강도(자세 여기). NIS 를 만드는 실제 원천
    ③ 추종오차  gt_err                  — 순간 오차와 누적 RMSE
    ④ NIS       vel / gyro (raw)        — 탐지 feature. 실기 실측 범위를 음영으로 겹쳐 표시

왜 한 화면인가
    "속도를 올리면 NIS 가 오르지 않나" 는 질문은 **같은 시간축 위에 겹쳐 봐야** 답이 된다.
    속도가 최대인 구간과 NIS 가 최대인 구간이 어긋난다는 것이 08-12 실기의 관찰이었다.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ('Noto Sans CJK KR', 'NanumGothic', 'Noto Sans CJK JP'):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

REAL_NIS_V = (0.18, 0.41)      # 실기 08-12 평시 p50 범위
REAL_NIS_G = (0.60, 1.17)
V_LIMIT = 2.0                  # MPC_XY_VEL_MAX (2026-08-13: 2.5 로 올렸다가 2.0 복귀)
V_DESIGN = 1.40                # R 2.8 × ω 0.5


def load(path):
    per = defaultdict(lambda: defaultdict(list))
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                if float(r.get('bias', 0)) != 0.0:
                    continue                      # 평시(무공격) 주행만
                # ★ deadline 캡처는 정책이 5종(track/dhover*)인데 **track 만 패턴을 실제로 비행**한다.
                #   dhover* 는 도중에 호버로 전환하므로 궤적 시계열로 쓰면 안 된다(2026-08-13 실측).
                if r.get('policy', 'track') != 'track':
                    continue
                k = (r['pattern'], r['policy'], r['episode'])
                for c in ('step', 'speed_xy', 'speed_z', 'omega_norm',
                          'gt_err', 'nis_v_raw', 'nis_g_raw', 'alt'):
                    per[k][c].append(float(r[c]))
            except (ValueError, KeyError):
                continue
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    src = os.path.join(a.outdir, 'sweep_detail.csv')
    per = load(src)
    if not per:
        raise SystemExit(f'[!] {src} 에 평시 주행이 없다')

    # 패턴별로 가장 긴 track 주행 하나씩
    best = {}
    for (pat, pol, ep), d in per.items():
        if len(d['step']) > len(best.get(pat, {}).get('step', [])):
            best[pat] = d
    order = [p for p in ('hover', 'waypoint', 'circle', 'figure8', 'aggressive') if p in best]
    order += [p for p in best if p not in order]

    n = len(order)
    fig, axes = plt.subplots(n, 4, figsize=(19, 2.5 * n), squeeze=False)
    fig.suptitle('패턴별 한 주행 시계열 — 속도 · 각속도 · 추종오차 · NIS   '
                 f'(MPC_XY_VEL_MAX {V_LIMIT} m/s, 설계 접선 {V_DESIGN} m/s)',
                 fontsize=14, fontweight='bold', y=0.995)

    summary = []
    for i, pat in enumerate(order):
        d = best[pat]
        t = np.array(d['step']) * 0.1                       # 10Hz → 초
        vxy, vz = np.array(d['speed_xy']), np.array(d['speed_z'])
        om = np.array(d['omega_norm'])
        err = np.array(d['gt_err'])
        nv, ng = np.array(d['nis_v_raw']), np.array(d['nis_g_raw'])
        rmse = np.sqrt(np.cumsum(err ** 2) / np.arange(1, err.size + 1))

        ax = axes[i][0]
        ax.plot(t, vxy, lw=1.2, color='#1D4ED8', label='|v_xy|')
        ax.plot(t, vz, lw=0.9, color='#93C5FD', label='v_z')
        ax.axhline(V_LIMIT, ls='--', lw=1, color='#DC2626', label=f'상한 {V_LIMIT}')
        ax.axhline(V_DESIGN, ls=':', lw=1, color='#059669', label=f'설계 {V_DESIGN}')
        ax.set_ylabel(f'{pat}\n속도 [m/s]', fontsize=10, fontweight='bold')
        ax.set_ylim(min(-0.5, vz.min() - 0.2), max(V_LIMIT * 1.15, vxy.max() * 1.15))
        if i == 0:
            ax.legend(fontsize=7, loc='upper right', ncol=2)

        ax = axes[i][1]
        ax.plot(t, om, lw=1.1, color='#7C3AED')
        ax.set_ylabel('|ω| [rad/s]', fontsize=9)
        ax.axhline(np.median(om), ls=':', lw=1, color='#4B5563')

        ax = axes[i][2]
        ax.plot(t, err, lw=1.0, color='#B45309', label='순간오차')
        ax.plot(t, rmse, lw=1.8, color='#DC2626', label='누적 RMSE')
        ax.set_ylabel('추종오차 [m]', fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

        ax = axes[i][3]
        ax.axhspan(*REAL_NIS_V, color='#1D4ED8', alpha=0.10)
        ax.axhspan(*REAL_NIS_G, color='#059669', alpha=0.10)
        ax.plot(t, nv, lw=1.0, color='#1D4ED8', label='vel NIS')
        ax.plot(t, ng, lw=1.0, color='#059669', label='gyro NIS')
        ax.set_ylabel('NIS (raw)', fontsize=9)
        ax.set_yscale('symlog', linthresh=1.0)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
            ax.set_title('음영 = 실기 08-12 평시 범위', fontsize=8, color='#6B7280')

        for j in range(4):
            axes[i][j].grid(alpha=0.25, lw=0.5)
            axes[i][j].tick_params(labelsize=8)
            if i == n - 1:
                axes[i][j].set_xlabel('시간 [s]  (10 Hz timestep)', fontsize=9)

        clamp = 100.0 * np.mean(vxy > 0.97 * V_LIMIT)
        summary.append((pat, len(t), np.median(vxy), vxy.max(), clamp, np.median(om),
                        rmse[-1], np.median(nv), np.median(ng)))

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = a.out or os.path.join(a.outdir, 'pattern_timeseries.png')
    fig.savefig(out, dpi=130)
    print(f'저장: {out}')

    print(f"\n{'패턴':12s} {'n':>5s} {'v_med':>7s} {'v_max':>7s} {'클램프%':>8s} "
          f"{'|ω|_med':>8s} {'RMSE':>7s} {'NIS_v':>7s} {'NIS_g':>7s}")
    print('-' * 78)
    for r in summary:
        print(f'{r[0]:12s} {r[1]:5d} {r[2]:7.2f} {r[3]:7.2f} {r[4]:8.1f} '
              f'{r[5]:8.3f} {r[6]:7.3f} {r[7]:7.2f} {r[8]:7.2f}')
    print(f"\n실기 08-12 평시 NIS p50: vel {REAL_NIS_V[0]}~{REAL_NIS_V[1]} · "
          f"gyro {REAL_NIS_G[0]}~{REAL_NIS_G[1]}")
    print("\n※ 2026-08-13 궤적 수정 후: 전 패턴이 **기하**(R×ω≈1.40)에 지배된다.")
    print("   구 구현은 aggressive phase 경계 2.24/4.48 m 순간이동, waypoint 레그당 4.0s 고정이라")
    print("   속도가 상한에 지배됐다 → 스파이크·클램프의 원인이었다.")


if __name__ == '__main__':
    main()
