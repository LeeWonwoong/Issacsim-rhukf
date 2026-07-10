"""
plot_fine_staircase.py — ramp {0.0,0.1,0.3} × bias × policy 생존율 계단그림
====================================================================
논문 figure 후보: 3조건(결과성/탐지가능성/대응가능성) 겹침을 한 눈에.
  - 좌: 판별 정책(track=결과성 하한, hover=구제 상한, dhover3=탐지지연3 구제)
        생존율 vs bias, ramp별 3-패널 계단.
  - 우: dhover3 생존율 vs bias를 ramp 3곡선 오버레이 (ramp 의존성 서사).

사용: python3 plot_fine_staircase.py [out.png]
"""
import sys, os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 한글 라벨 렌더링 (Noto Sans CJK KR). 없으면 무시.
from matplotlib import font_manager
for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=_fp).get_name()
        except Exception:
            pass
        break
plt.rcParams['axes.unicode_minus'] = False

RAMPS = [('0.0', 'results_torque_r00_fine'),
         ('0.1', 'results_torque_r01_fine'),
         ('0.3', 'results_torque_r03_fine')]
POLS = ['track', 'dhover1', 'dhover2', 'dhover3', 'hover']
COL = {'track': '#d62728', 'dhover1': '#ff7f0e', 'dhover2': '#9467bd',
       'dhover3': '#1f77b4', 'hover': '#2ca02c'}


def surv_by_bias(path):
    """return {policy: {bias: survrate}}, sorted bias list."""
    if not os.path.exists(path):
        return None, []
    rows = list(csv.DictReader(open(path)))
    acc = defaultdict(list)
    for r in rows:
        acc[(float(r['bias']), r['policy'])].append(int(r['survived']))
    biases = sorted({float(r['bias']) for r in rows if float(r['bias']) > 0})
    out = defaultdict(dict)
    for (b, pol), v in acc.items():
        if b > 0:
            out[pol][b] = float(np.mean(v))
    return out, biases


FINAL = ('final', 'results_torque_final', (1.30, 1.32))  # (라벨, dir, 동결밴드[lo,hi])


def main():
    outpng = sys.argv[1] if len(sys.argv) > 1 else 'ramp_staircase.png'
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.6), sharey=True)

    data = {}
    for ramp, d in RAMPS:
        data[ramp], _ = surv_by_bias(os.path.join(d, 'sweep_summary.csv'))

    # 좌 3패널: ramp별 정책 계단
    for ax, (ramp, d) in zip(axes[:3], RAMPS):
        sd = data[ramp]
        if not sd:
            ax.set_title(f'ramp={ramp}s  (데이터 없음)')
            ax.set_xlabel('torque bias (Nm)')
            continue
        for pol in POLS:
            if pol not in sd:
                continue
            bs = sorted(sd[pol])
            ys = [sd[pol][b] for b in bs]
            ax.step(bs, ys, where='mid', marker='o', ms=4,
                    color=COL[pol], label=pol, lw=1.8)
        ax.axhspan(-0.02, 0.5, color='gray', alpha=0.07)  # 추락 우세 영역
        ax.set_title(f'ramp={ramp}s')
        ax.set_xlabel('torque bias (Nm)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('생존율 (ramp패널=8ep, FINAL=20ep)')
    axes[0].legend(fontsize=8, loc='center left')

    # 우 패널: dhover3 생존율 ramp 오버레이 (+ track/hover 얇게 참조)
    ax = axes[3]
    rc = {'0.0': '#1b9e77', '0.1': '#d95f02', '0.3': '#7570b3'}
    for ramp, d in RAMPS:
        sd = data[ramp]
        if not sd or 'dhover3' not in sd:
            continue
        bs = sorted(sd['dhover3'])
        ys = [sd['dhover3'][b] for b in bs]
        ax.step(bs, ys, where='mid', marker='s', ms=5, lw=2.2,
                color=rc[ramp], label=f'ramp={ramp}s')
    ax.axhline(0.5, color='k', ls=':', lw=1)
    ax.axhspan(-0.02, 0.5, color='gray', alpha=0.07)
    ax.set_title('d=3 구제 생존율 vs bias (ramp 의존성)')
    ax.set_xlabel('torque bias (Nm)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # 5번째 패널: 최종 20ep 스윕(동결) — 정책 계단 + 결과성 밴드 음영
    axf = axes[4]
    flabel, fdir, (blo, bhi) = FINAL
    fsd, _ = surv_by_bias(os.path.join(fdir, 'sweep_summary.csv'))
    if fsd:
        axf.axvspan(blo, bhi, color='#2ca02c', alpha=0.12, label=f'동결밴드 [{blo},{bhi}]')
        for pol in POLS:
            if pol not in fsd:
                continue
            bs = sorted(fsd[pol])
            ys = [fsd[pol][b] for b in bs]
            axf.step(bs, ys, where='mid', marker='o', ms=4,
                     color=COL[pol], label=pol, lw=1.8)
        axf.axhspan(-0.02, 0.5, color='gray', alpha=0.07)
        axf.set_title(f'FINAL {flabel} (20ep, ramp0.0) — 동결밴드 [{blo}, {bhi}]Nm')
    else:
        axf.set_title(f'FINAL {flabel} (데이터 없음)')
    axf.set_xlabel('torque bias (Nm)')
    axf.set_ylim(-0.05, 1.05)
    axf.grid(alpha=0.25)
    axf.legend(fontsize=8, loc='center left')

    fig.suptitle('torque fine sweep — ramp × bias × policy 생존율 계단 '
                 '(track=결과성 하한, hover=구제 상한, dhover3=탐지지연3 구제)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpng, dpi=130)
    print(f'saved {outpng}')

    # 콘솔 판독 요약
    print('\n=== 판독 요약 (survival, d=3 vs track vs hover) ===')
    for ramp, d in RAMPS:
        sd = data[ramp]
        print(f'\n[ramp={ramp}s]  ({d})')
        if not sd:
            print('  데이터 없음'); continue
        biases = sorted({b for pol in sd for b in sd[pol]})
        print(f"  {'bias':>6} | {'track':>6} {'dh1':>5} {'dh2':>5} {'dh3':>5} {'hover':>6}")
        for b in biases:
            def g(p): return f"{sd[p][b]:.2f}" if p in sd and b in sd[p] else "  — "
            print(f"  {b:6.3f} | {g('track'):>6} {g('dhover1'):>5} "
                  f"{g('dhover2'):>5} {g('dhover3'):>5} {g('hover'):>6}")


if __name__ == '__main__':
    main()
