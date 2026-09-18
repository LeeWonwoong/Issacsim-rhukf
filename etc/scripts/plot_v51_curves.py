#!/usr/bin/env python3
# env5.1 학습곡선 — reward / F1 / loss, 이동평균 20에피
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['SWIRL_P0.002_R1.5','SWIRL_P0.01_R1.5','SWIRL_P0.03_R1.5','SWIRL_P0.03_R1',
         'SWIRL_P0.1_R1.5','Adam_mse_3e-4','Adam_mse_1e-3']
H = {0.002:0.10, 0.01:0.23, 0.03:0.39, 0.1:0.72}
COL = {'SWIRL_P0.002_R1.5':'#9ecae1','SWIRL_P0.01_R1.5':'#4292c6','SWIRL_P0.03_R1.5':'#08519c',
       'SWIRL_P0.03_R1':'#6a51a3','SWIRL_P0.1_R1.5':'#41ab5d',
       'Adam_mse_3e-4':'#d94801','Adam_mse_1e-3':'#fb6a4a'}
def ma(x, w=20):
    x = np.asarray(x, float)
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

D = {}
for f in glob.glob('results_v51curve/*.json'):
    d = json.load(open(f)); D[d['name']] = d

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for nm in ORDER:
    if nm not in D: continue
    h = D[nm]['hist']; c = COL[nm]
    lab = nm.replace('SWIRL_','SWIRL ').replace('Adam_mse_','Adam MSE ').replace('_',' ')
    if nm.startswith('SWIRL'):
        p = float(nm.split('_')[1][1:]); lab += f'  (h={H.get(p, 0):.2f})'
    ls = '--' if nm.startswith('Adam') else '-'
    rew = [e['reward'] for e in h]
    ax[0].plot(range(19, 19+len(ma(rew))), ma(rew), color=c, ls=ls, lw=1.8, label=lab)
    atk = [(e['ep'], e['f1']) for e in h if e['has_atk']]
    if atk:
        xs = [a[0] for a in atk]; ys = ma([a[1] for a in atk])
        ax[1].plot(xs[19:19+len(ys)], ys, color=c, ls=ls, lw=1.8)
    lo = [e['loss'] for e in h]
    ax[2].plot(range(19, 19+len(ma(lo))), ma(lo), color=c, ls=ls, lw=1.8)
for a, t, yl in zip(ax, ['Episode reward (MA20)', 'F1 on attack episodes (MA20)', 'TD loss (MA20)'],
                    ['reward', 'F1', 'loss']):
    a.set_title(t, fontsize=12); a.set_xlabel('episode'); a.set_ylabel(yl); a.grid(alpha=.3)
ax[2].set_yscale('log')
ax[0].legend(fontsize=8.5, loc='lower right')
fig.suptitle('surrogate env5.1 (Isaac-matched: gyro d\'=2.09, rho(5)=+0.807)  |  160 ep, seed 42, 300 steps/ep',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('v51_learning_curves.png', dpi=130)
print('saved: v51_learning_curves.png')

print(f"\n{'config':22s} {'h':>5} | {'rwd(1-40)':>10} {'rwd(41-80)':>11} {'rwd(81-120)':>12} {'rwd(121-160)':>13} | {'F1 last40':>10}")
for nm in ORDER:
    if nm not in D: continue
    h = D[nm]['hist']; rew = np.array([e['reward'] for e in h])
    p = float(nm.split('_')[1][1:]) if nm.startswith('SWIRL') else 0
    f1a = [e['f1'] for e in h[-40:] if e['has_atk']]
    print(f"{nm:22s} {H.get(p,0):5.2f} | {rew[:40].mean():10.1f} {rew[40:80].mean():11.1f} "
          f"{rew[80:120].mean():12.1f} {rew[120:].mean():13.1f} | {np.mean(f1a):10.3f}")
