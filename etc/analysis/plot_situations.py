"""상황별 대표 에피소드의 관측값(압축 NIS) 시계열 — d′ 대신 눈으로 보는 버전.
   공격구간=빨간음영, 바람=하늘색음영. 한글=Noto Sans CJK."""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 한글 폰트
_FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_FP).get_name()
plt.rcParams['axes.unicode_minus'] = False

OUT = sys.argv[1] if len(sys.argv) > 1 else 'situations_timeseries.png'

def comp(x):
    return np.minimum(np.log1p(np.sqrt(np.maximum(x, 0.0))), 3.0)
def winmax(v, w=4):
    return np.array([v[max(0, i-w+1):i+1].max() for i in range(len(v))])

def load(ws):
    p = f'results_62_ws{ws}/sweep_detail.csv'
    if not os.path.exists(p):
        return {}
    from collections import defaultdict
    ep = defaultdict(list)
    for r in csv.DictReader(open(p)):
        try:
            ep[(float(r['bias']), r['pattern'], r['episode'], r['policy'])].append(
                (int(r['step']), float(r['nis_v_raw']), float(r['nis_g_raw']), int(r['attack_active'])))
        except (ValueError, KeyError):
            pass
    return ep

def pick(ep, bias, want_attack):
    cands = [(k, v) for k, v in ep.items() if abs(k[0]-bias) < 0.01 and 'track' in k[3] and k[1] == 'aggressive']
    if want_attack:
        cands = [(k, v) for k, v in cands if any(x[3] == 1 for x in v)]
    if not cands:
        return None
    k, v = max(cands, key=lambda kv: len(kv[1]))
    v.sort(key=lambda x: x[0])
    return v

# 상황: (제목, ws, bias, 바람있음)
situations = [
    ('① 평시 (무풍·무공격)', 0, 0.0, False),
    ('② 바람만 (ws12·무공격)', 12, 0.0, True),
    ('③ 공격만 (무풍·틸트 δ0.6)', 0, 2.6, True),
    ('④ 공격+강풍 (ws12·틸트)', 12, 2.6, True),
]
eps = {w: load(w) for w in [0, 5, 9, 12, 15]}

fig, axes = plt.subplots(4, 1, figsize=(13, 12))
fig.suptitle('상황별 관측값(압축 NIS = log(1+√NIS) clip3) 시계열\n'
             '얇은선=단일스텝, 굵은선=슬라이딩윈도우(4) · 빨강=gyro(공격채널) 파랑=vel(바람채널) · '
             '빨간음영=공격, 하늘음영=바람',
             fontsize=12, fontweight='bold')

for ax, (name, ws, bias, has_wind) in zip(axes, situations):
    ep = eps.get(ws, {})
    s = pick(ep, bias, want_attack=(bias > 0))
    if s is None:
        ax.set_title(f'{name} — 데이터 없음'); ax.grid(alpha=0.3); continue
    t = np.arange(len(s))
    ng = comp(np.array([x[2] for x in s])); nv = comp(np.array([x[1] for x in s]))
    atk = np.array([x[3] for x in s])
    ngw, nvw = winmax(ng), winmax(nv)
    # 바람 음영 (전체 — sweep 셀은 바람 상시)
    if has_wind:
        ax.axvspan(0, len(t), color='#5aa9e6', alpha=0.10, zorder=0)
    # 공격 음영
    if atk.max() > 0:
        inatk = False
        for i in range(len(atk)):
            if atk[i] and not inatk:
                st = i; inatk = True
            elif not atk[i] and inatk:
                ax.axvspan(st, i, color='#d1483a', alpha=0.14, zorder=0); inatk = False
        if inatk:
            ax.axvspan(st, len(atk), color='#d1483a', alpha=0.14, zorder=0)
    ax.plot(t, ng, color='#d1483a', lw=0.7, alpha=0.45)
    ax.plot(t, ngw, color='#d1483a', lw=2.2, label='gyro (공격채널)')
    ax.plot(t, nv, color='#2f62e6', lw=0.7, alpha=0.45)
    ax.plot(t, nvw, color='#2f62e6', lw=2.2, label='vel (바람채널)')
    ax.set_title(name, fontsize=11)
    ax.set_ylabel('압축 NIS [0,3]'); ax.set_ylim(0, 3.1); ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('스텝 (에피소드 내)')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'saved {OUT}')
