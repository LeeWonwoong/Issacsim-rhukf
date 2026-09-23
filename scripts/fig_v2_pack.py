#!/usr/bin/env python3
"""v2 무대 그림 두 장 (기존 로그만 읽음, 학습·GPU 없음).
  A) surrogate 10시드 4학습기 학습곡선: 학습중 보상 cost/c · 공격 에피 F1 · greedy 프로브 F1 + 최종 greedy F1 막대
  B) Isaac v2 시드42 온라인 플레이(같은 시나리오 짝): 클래스별 첫 persistent 공격 에피(ep≥140) — δ_eff · gyro 관측 · 행동 · 기울기
"""
import os, glob, json, yaml, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
R = 'results/claudecodefortest/'
OUT = os.environ.get('OUT', R + 'figs')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# 참조 팔레트 1–4번 슬롯(순서 고정) + 선 모양 이중 부호화
L = {'SWIRL':  dict(c='#2a78d6', ls='-',  lw=2.2, grp=('final_v4', 'a_sF_buffer50000')),
     'Adam':   dict(c='#eb6834', ls=':',  lw=2.0, grp=('final_v4', 'a_aF_buffer50000')),
     'EKF-TD': dict(c='#1baf7a', ls='-.', lw=1.8, grp=('n1_ktd_std', 'a_kS_typeekf')),
     'UKF-TD': dict(c='#eda100', ls='--', lw=1.8, grp=('n1_ktd_std', 'a_kS_typeukf'))}
ORDER = ['SWIRL', 'UKF-TD', 'EKF-TD', 'Adam']

def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b3')
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)

def mavg(x, w):
    x = np.asarray(x, float); out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1):i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg): out[i] = seg.mean()
    return out

# ---------------- A) surrogate 10시드 ----------------
curves, finals = {}, {}
for nm, d in L.items():
    sub, pre = d['grp']; runs = sorted(glob.glob(f'{R}{sub}/{pre}_s*/'))
    C, F1, PR, EV = [], [], [], []
    for r in runs:
        h = json.load(open(r + 'hist.json'))['hist']; sc = float(yaml.safe_load(open(r + 'config.yaml'))['reward'].get('scale', 1.0))
        n = 200; cost = np.full(n, np.nan); f1 = np.full(n, np.nan); pf = np.full(n, np.nan)
        for e in h:
            k = int(e['ep'])
            if k >= n: continue
            cost[k] = e['reward_cost'] / sc
            if e.get('has_atk'): f1[k] = e['f1']
            if e.get('probe_f1') is not None: pf[k] = e['probe_f1']
        C.append(mavg(cost, 10)); F1.append(mavg(f1, 20)); PR.append(mavg(pf, 10))
        ev = json.load(open(r + 'eval.json')); EV.append(ev['f1'])
    curves[nm] = (np.array(C), np.array(F1), np.array(PR)); finals[nm] = np.array(EV)
    print(f'{nm:7s} 런 {len(runs)}  greedy F1 {np.mean(EV):.3f}±{np.std(EV, ddof=1):.3f}  학습중 cost/c(ep100–199) {np.nanmean(np.array(C)[:, 100:200]):+.2f}')

fig = plt.figure(figsize=(15.5, 4.6), dpi=150)
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.78], wspace=0.28)
x = np.arange(200)
titles = ['학습 중 보상 cost/c (10ep 이동평균)', '공격 에피소드 F1 (학습 중, 20ep 이동평균)', 'greedy 프로브 F1 (탐험 없이, 10ep 이동평균)']
for j in range(3):
    ax = fig.add_subplot(gs[0, j]); style(ax)
    for nm in ORDER:
        d = L[nm]; Y = curves[nm][j]
        if np.all(np.isnan(Y)): continue
        m = np.nanmean(Y, 0); lo, hi = np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)
        ax.fill_between(x, lo, hi, color=d['c'], alpha=0.10, lw=0)
        ax.plot(x, m, color=d['c'], ls=d['ls'], lw=d['lw'], label=nm)
    ax.set_title(titles[j], fontsize=10.5, color=INK, loc='left'); ax.set_xlabel('에피소드', color=INK2, fontsize=9)
    ax.set_xlim(15, 199)
    if j == 0: ax.set_ylim(-4.5, 0.6); ax.axhline(0, color='#9a9994', lw=0.8)
    else: ax.set_ylim(0.25, 0.95)
    if j == 0: ax.legend(loc='lower right', fontsize=8.5, frameon=False, handlelength=2.6)
    # 오른쪽 끝 직접 라벨(최종 10ep 평균)
    if j == 0:
        ends = {nm: np.nanmean(curves[nm][0][:, 190:200]) for nm in ORDER}
        for nm, v in ends.items(): pass
ax = fig.add_subplot(gs[0, 3]); style(ax)
for i, nm in enumerate(ORDER):
    v = finals[nm]; d = L[nm]
    ax.bar(i, v.mean(), width=0.62, color=d['c'], alpha=0.9, edgecolor='white', linewidth=2)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v, s=9, color=INK, alpha=0.55, zorder=3)
    ax.text(i, v.mean() + 0.045, f'{v.mean():.3f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, fontsize=9, color=INK2); ax.set_ylim(0.6, 0.96)
ax.set_title('최종 greedy F1 (100ep, 점=시드)', fontsize=10.5, color=INK, loc='left')
fig.suptitle('v2 무대 · surrogate 10시드 · 칼만-TD = 표준판(act=active·P 지속·N1) · 띠 = 사분위(25–75%)', fontsize=11, color=INK, y=1.02)
fa = os.path.join(OUT, 'v2_4learners_curves.png'); fig.savefig(fa, bbox_inches='tight', facecolor='white'); plt.close(fig); print('저장', fa)

# ---------------- B) Isaac v2 온라인 플레이 ----------------
IS = R + 'isaac_v2/'
def load(run, e):
    z = np.load(f'{IS}{run}/steps/ep{e:04d}.npz', allow_pickle=True); cs = [str(c) for c in z['cols']]; ix = {k: i for i, k in enumerate(cs)}
    Rw = z['rows']; g = lambda k: Rw[:, ix[k]]
    return dict(z=z, d=g('delta_eff'), a=g('atk_flag') > 0.5, h=g('prev_action') > 0.5, go=g('g_obs'), vo=g('v_obs'),
                tilt=np.degrees(np.hypot(g('roll'), g('pitch'))), alt=g('alt'))
pick = {}
for f in sorted(glob.glob(IS + 'swirl_s42/steps/ep*.npz')):
    e = int(os.path.basename(f)[2:6])
    if e < 140 or not os.path.exists(f'{IS}adam_s42/steps/ep{e:04d}.npz'): continue
    k = str(np.load(f, allow_pickle=True)['kind'])
    if k in ('weak_persist', 'trans_persist', 'strong_persist') and k not in pick: pick[k] = e
eps = [(k, pick[k]) for k in ('weak_persist', 'trans_persist', 'strong_persist') if k in pick]
print('선정(클래스별 ep≥140 첫 persistent):', eps)
fig, axs = plt.subplots(4, len(eps), figsize=(5.3 * len(eps), 9.2), dpi=150, sharex='col',
                        gridspec_kw=dict(height_ratios=[0.8, 1.1, 0.7, 1.0], hspace=0.12, wspace=0.16))
for j, (k, e) in enumerate(eps):
    S, A = load('swirl_s42', e), load('adam_s42', e); n = min(len(S['d']), len(A['d'])); t = np.arange(n)
    z = S['z']; win = S['a'][:n]
    st = np.where(win & ~np.r_[False, win[:-1]])[0]; en = np.where(win & ~np.r_[win[1:], False])[0]
    lo_, hi_ = max(0, st[0] - 40), min(n, en[-1] + 60)
    for i in range(4):
        ax = axs[i, j]; style(ax)
        for s0, s1 in zip(st, en): ax.axvspan(s0 - 0.5, s1 + 0.5, color='#f3d9d3', alpha=0.55, lw=0)
    ax = axs[0, j]; ax.step(t, S['d'][:n], where='post', color=INK, lw=1.6); ax.set_ylim(-0.03, 0.85)
    tp_s = int((S['a'][:n] & S['h'][:n]).sum()); tp_a = int((A['a'][:n] & A['h'][:n]).sum()); na = int(S['a'][:n].sum())
    ax.set_title(f'ep{e} · {k} · {z["pattern"]} · 풍속 {float(z["wind_speed"]):.1f} m/s\n공격 중 hover: SWIRL {tp_s}/{na} · Adam {tp_a}/{na}', fontsize=10, color=INK, loc='left')
    ax = axs[1, j]
    ax.plot(t, S['go'][:n], color=L['SWIRL']['c'], lw=1.5, label='SWIRL 정책 하 관측'); ax.plot(t, A['go'][:n], color=L['Adam']['c'], lw=1.3, ls=':', label='Adam 정책 하 관측')
    ax.set_ylim(0, 1.0)
    ax = axs[2, j]
    ax.fill_between(t, 1.15, 1.15 + 0.7 * S['h'][:n], step='post', color=L['SWIRL']['c'], lw=0)
    ax.fill_between(t, 0.15, 0.15 + 0.7 * A['h'][:n], step='post', color=L['Adam']['c'], lw=0)
    ax.set_ylim(0, 2.0); ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Adam', 'SWIRL'], fontsize=9, color=INK2); ax.grid(False)
    ax = axs[3, j]
    ax.plot(t, S['tilt'][:n], color=L['SWIRL']['c'], lw=1.5); ax.plot(t, A['tilt'][:n], color=L['Adam']['c'], lw=1.3, ls=':')
    ax.set_xlim(lo_, hi_); ax.set_xlabel('스텝 (10 Hz)', color=INK2, fontsize=9)
    if j == 0:
        axs[0, 0].set_ylabel('δ_eff (공격 세기)', color=INK2, fontsize=9); axs[1, 0].set_ylabel('gyro 관측 (압축 0–1)', color=INK2, fontsize=9)
        axs[2, 0].set_ylabel('행동 (채움 = hover)', color=INK2, fontsize=9); axs[3, 0].set_ylabel('기울기 |roll,pitch| [deg]', color=INK2, fontsize=9)
        axs[1, 0].legend(loc='upper left', fontsize=8.5, frameon=False)
fig.suptitle('Isaac v2 · 시드42 · 같은 시나리오 짝(학습 중 온라인 플레이, ε≈0.01) · 붉은 띠 = 공격 구간 · 에피 선정 규칙 = 클래스별 ep≥140 첫 persistent',
             fontsize=10.5, color=INK, y=0.955)
fb = os.path.join(OUT, 'v2_isaac_online_play.png'); fig.savefig(fb, bbox_inches='tight', facecolor='white'); plt.close(fig); print('저장', fb)
