"""
plot_deadline.py — ramp별 bias × d=3 생존율 계단 그림 + dhover crash-lag 진단
집계 CSV만 사용(numpy/matplotlib). Isaac 불필요.
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOLDERS = [
    ('torque r0.1',  'results_torque_r01', 'torque'),
    ('torque r0.3',  'results_torque_r03', 'torque'),
    ('combined r0.3','results_pilot4',     'combined'),
]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def surv_by_bias(summ, policy):
    d = defaultdict(list)
    for r in summ:
        if r['policy'] == policy:
            d[float(r['bias'])].append(int(r['survived']))
    return {b: float(np.mean(v)) for b, v in d.items()}

# ── 1. d=3 계단 곡선 + hover/track 참조선 ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
lag_report = []
for ax, (name, folder, mode) in zip(axes, FOLDERS):
    summ = load(os.path.join(folder, 'sweep_summary.csv'))
    d3    = surv_by_bias(summ, 'dhover3')
    d1    = surv_by_bias(summ, 'dhover1')
    hov   = surv_by_bias(summ, 'hover')
    trk   = surv_by_bias(summ, 'track')
    biases = sorted(b for b in set(list(d3)+list(hov)+list(trk)) if b > 0)
    x = biases
    ax.step([b for b in x], [hov.get(b, np.nan) for b in x], where='mid',
            color='tab:green', lw=2, label='hover (fallback ceiling)')
    ax.step(x, [trk.get(b, np.nan) for b in x], where='mid',
            color='tab:red', lw=2, label='track (no response)')
    ax.step(x, [d3.get(b, np.nan) for b in x], where='mid',
            color='tab:blue', lw=2.5, marker='o', label='dhover d=3 (detect@3→hover)')
    ax.step(x, [d1.get(b, np.nan) for b in x], where='mid',
            color='tab:cyan', lw=1.2, ls='--', alpha=.7, label='dhover d=1 (best case)')
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_title(f'{name}\n(mode={mode})', fontsize=11)
    ax.set_xlabel('bias scale [Nm]')
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=.25)
axes[0].set_ylabel('survival rate')
axes[0].legend(fontsize=8, loc='center right')
fig.suptitle('Response-deadline staircase: survival vs attack bias, per ramp  '
             '(Cond-3: does the d=3 curve meet the hover ceiling at some bias?)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('deadline_staircase.png', dpi=130)
print('saved deadline_staircase.png')

# ── 2. pilot4 dhover crash-lag 진단 (band bias) ──────────────────
det = load('results_pilot4/sweep_detail.csv')
summ4 = load('results_pilot4/sweep_summary.csv')
# 전환 스텝 d 이후 몇 스텝 만에 죽었나
print("\n=== pilot4 dhover crash-lag (전환스텝 대비 추락지연) ===")
print("bias  policy   n_crash  mean_crashstep  mean_lag(=crash - d)  reasons")
for b in [1.33, 1.36, 1.38, 1.40, 1.44]:
    for pol in ['dhover2', 'dhover3']:
        d = int(pol[6:])
        rows = [r for r in summ4 if abs(float(r['bias'])-b) < 1e-6 and r['policy'] == pol]
        crashed = [r for r in rows if int(r['survived']) == 0]
        if not crashed:
            continue
        cs = [int(r['crash_step']) for r in crashed]
        reasons = defaultdict(int)
        for r in crashed:
            reasons[r['crash_reason']] += 1
        lag = np.mean(cs) - d
        print(f"{b:.2f}  {pol:8} {len(crashed):5}    {np.mean(cs):8.1f}       "
              f"{lag:8.1f}          {dict(reasons)}")
