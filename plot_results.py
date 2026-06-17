"""
plot_results.py — 학습 메트릭 plot (단일 / RHUKF vs Adam 비교 통합)
====================================================================
online_rl_main.py가 에피소드마다 기록하는 metrics_<agent>.csv 를 읽어
reward / loss / F1 / FP율 / 검출지연 / 추락율 / TD첨도 를 한 figure로.

사용:
  단일:   python plot_results.py results_rhukf/metrics_rhukf.csv
  비교:   python plot_results.py results_rhukf/metrics_rhukf.csv results_adam/metrics_adam.csv
          → 같은 패널에 두 에이전트를 겹쳐 그림 (agent 컬럼으로 자동 구분/라벨)
  옵션:   --outdir <폴더(기본 .)>  --smooth <롤링창(기본 20)>  --out <파일명>

CSV 컬럼: episode,agent,reward,loss,steps,tp,fp,fn,tn,precision,recall,f1,
          fp_rate,det_delay,crashed,td_exkurt,epsilon
"""
import sys, os, csv, argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {'rhukf': '#1f77b4', 'adam': '#d62728'}   # RHUKF=파랑, Adam=빨강
DEFAULT_C = ['#2ca02c', '#9467bd', '#ff7f0e']


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, None
    agent = rows[0].get('agent', os.path.basename(path))
    def col(name, cast=float):
        out = []
        for r in rows:
            try:
                out.append(cast(r[name]))
            except (KeyError, ValueError):
                out.append(np.nan)
        return np.array(out, dtype=float)
    data = {k: col(k) for k in
            ['episode', 'reward', 'loss', 'f1', 'fp_rate', 'det_delay',
             'crashed', 'td_exkurt', 'recall', 'precision']}
    return agent, data


def roll(y, w):
    """NaN 무시 롤링 평균."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < 2:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def plot_panel(ax, runs, key, title, ylabel, smooth, valid_filter=None):
    for agent, data in runs:
        x = data['episode']
        y = data[key].copy()
        if valid_filter is not None:
            mask = valid_filter(y)
            y = np.where(mask, y, np.nan)
        c = COLORS.get(agent, DEFAULT_C[hash(agent) % len(DEFAULT_C)])
        ax.plot(x, y, color=c, alpha=0.18, lw=0.8)                 # raw (옅게)
        ax.plot(x, roll(y, smooth), color=c, lw=2.0, label=agent)  # 롤링 평균
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('episode'); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25); ax.legend(fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+', help='metrics_<agent>.csv (1개=단일, 2개=비교)')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--smooth', type=int, default=20, help='롤링 평균 창')
    ap.add_argument('--out', default=None, help='출력 파일명(기본 자동)')
    args = ap.parse_args()

    runs = []
    for p in args.csvs:
        agent, data = load(p)
        if data is not None:
            runs.append((agent, data))
            print(f'[load] {p}  agent={agent}  episodes={len(data["episode"])}')
    if not runs:
        print('읽을 데이터 없음'); return

    mode = 'compare' if len(runs) > 1 else runs[0][0]
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    sm = args.smooth

    plot_panel(axes[0], runs, 'reward', 'Episode Reward', 'reward', sm)
    plot_panel(axes[1], runs, 'loss', 'Training Loss', 'loss', sm)
    plot_panel(axes[2], runs, 'f1', 'Detection F1', 'F1', sm)
    plot_panel(axes[3], runs, 'precision', 'Precision', 'precision', sm)
    plot_panel(axes[4], runs, 'recall', 'Recall', 'recall', sm)
    plot_panel(axes[5], runs, 'fp_rate', 'False-Positive Rate', 'FP rate', sm)
    plot_panel(axes[6], runs, 'det_delay', 'Detection Delay', 'steps', sm,
               valid_filter=lambda y: y >= 0)   # -1(미검출/무공격) 제외
    plot_panel(axes[7], runs, 'crashed', 'Crash Rate', 'crash', sm)
    plot_panel(axes[8], runs, 'td_exkurt', 'TD-error Excess Kurtosis (heavy-tail)', 'exkurt', sm)

    title = ('RHUKF vs Adam — Training Comparison' if mode == 'compare'
             else f'{mode.upper()} — Training Metrics')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or (f'compare_rhukf_adam.png' if mode == 'compare'
                       else f'metrics_{mode}.png')
    out = os.path.join(args.outdir, out)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'[saved] {out}')

    # 최종 구간(마지막 20%) 요약 표 출력
    print('\n=== 최종 구간(마지막 20%) 평균 ===')
    print(f'{"agent":>8} | {"reward":>8} {"loss":>7} {"F1":>6} {"FPrate":>7} '
          f'{"delay":>6} {"crash":>6} {"exkurt":>7}')
    for agent, d in runs:
        n = len(d['episode']); k = max(1, n // 5)
        def tail(key, vf=None):
            y = d[key][-k:]
            if vf is not None:
                y = y[vf(y)]
            y = y[~np.isnan(y)]
            return y.mean() if len(y) else float('nan')
        print(f'{agent:>8} | {tail("reward"):8.1f} {tail("loss"):7.3f} '
              f'{tail("f1"):6.3f} {tail("fp_rate"):7.3f} '
              f'{tail("det_delay", lambda y: y>=0):6.1f} {tail("crashed"):6.2f} '
              f'{tail("td_exkurt"):7.2f}')


if __name__ == '__main__':
    main()
