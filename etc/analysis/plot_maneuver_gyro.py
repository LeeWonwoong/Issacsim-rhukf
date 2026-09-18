"""패턴별 benign 기동에서 gyro가 튀나? — omega(방향전환)는 변하는데 gyro NIS는? """
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from collections import defaultdict

_FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_FP).get_name()
plt.rcParams['axes.unicode_minus'] = False

def comp(x):
    return np.minimum(np.log1p(np.sqrt(np.maximum(x, 0.0))), 3.0)
def winmax(v, w=4):
    return np.array([v[max(0, i-w+1):i+1].max() for i in range(len(v))])

rows = list(csv.DictReader(open('etc/results/results_windthreshold/sweep_detail.csv')))
ep = defaultdict(list)
for r in rows:
    try:
        ep[(r['pattern'], float(r['wind_speed']), float(r['bias']), r['episode'])].append(
            (int(r['step']), float(r['nis_v_raw']), float(r['nis_g_raw']),
             float(r.get('omega_norm', 0)), int(r['attack_active'])))
    except (ValueError, KeyError):
        pass

def pick(pat, ws, bias, want_atk):
    cands = [(k, v) for k, v in ep.items() if k[0] == pat and abs(k[1]-ws) < 0.1 and abs(k[2]-bias) < 0.01]
    if want_atk:
        cands = [(k, v) for k, v in cands if any(x[4] == 1 for x in v)]
    if not cands:
        return None
    k, v = max(cands, key=lambda kv: len(kv[1])); v.sort(key=lambda x: x[0]); return v

pats = ['aggressive', 'circle', 'figure8', 'waypoint']
fig, axes = plt.subplots(len(pats), 2, figsize=(15, 11))
fig.suptitle('패턴별 기동에서 gyro가 튀나? (windthreshold, ws9)\n'
             '왼쪽=benign(공격X), 오른쪽=공격. 빨강=gyro압축NIS(굵=윈도우), 파랑=vel, 회색점선=omega(방향전환 세기)',
             fontsize=12, fontweight='bold')
for i, pat in enumerate(pats):
    for j, (cond, bias, want) in enumerate([('benign 기동', 0.0, False), ('공격', 3.05, True)]):
        ax = axes[i, j]
        s = pick(pat, 9.0, bias, want)
        if s is None:
            ax.set_title(f'{pat} — {cond} 없음'); continue
        t = np.arange(len(s))
        ng = comp(np.array([x[2] for x in s])); nv = comp(np.array([x[1] for x in s]))
        om = np.array([x[3] for x in s]); atk = np.array([x[4] for x in s])
        # omega 정규화(보기용, 0~3 스케일)
        oms = 3.0 * (om / (om.max() + 1e-9))
        if atk.max() > 0:
            ina = False
            for kk in range(len(atk)):
                if atk[kk] and not ina:
                    st = kk; ina = True
                elif not atk[kk] and ina:
                    ax.axvspan(st, kk, color='#d1483a', alpha=0.12); ina = False
            if ina:
                ax.axvspan(st, len(atk), color='#d1483a', alpha=0.12)
        ax.plot(t, oms, color='gray', lw=0.8, ls=':', alpha=0.7, label='omega(방향전환)')
        ax.plot(t, comp(np.array([x[1] for x in s])), color='#2f62e6', lw=0.6, alpha=0.4)
        ax.plot(t, winmax(nv), color='#2f62e6', lw=1.8, label='vel')
        ax.plot(t, ng, color='#d1483a', lw=0.6, alpha=0.4)
        ax.plot(t, winmax(ng), color='#d1483a', lw=2.2, label='gyro')
        ax.set_title(f'{pat} · {cond}', fontsize=10)
        ax.set_ylim(0, 3.1); ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(loc='upper left', fontsize=8)
        if j == 0:
            ax.set_ylabel('압축 NIS [0,3]')
axes[-1, 0].set_xlabel('스텝'); axes[-1, 1].set_xlabel('스텝')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('maneuver_gyro_check.png', dpi=125, bbox_inches='tight')
print('saved maneuver_gyro_check.png')
