#!/usr/bin/env python3
"""fig_paper (09-16): 논문 그림 3~6. 모두 기존 결과로 그린다.
   3 학습곡선(per5, 확약 1 vs 5) · 4 무대별 짝 차이 forest · 5 긴 지평 메커니즘(공분산 누적) · 6 Isaac 확인."""
import json, glob, os, csv, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
R = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'; N = f'{R}/night'
H = {}
for f in glob.glob(f'{N}/tonight/q/*.json') + glob.glob(f'{N}/tonight/[TUA]*_w*.json') + glob.glob(f'{N}/chain_hz/S1*_w*.json'):
    for k, m in json.load(open(f)).items():
        c = 'blk' if m['cell'] == 'g90n3' else (m['cell'][1:] if m['cell'].startswith('x') else m['cell'])
        if m['cell'] == 'g90n3' and m['learner'] not in ('SW', 'Adam3e-4', 'Adam1e-3'): continue
        H[(c, m['learner'], m['seed'])] = m['hist']
NM = {'SW': 'SWIRL', 'UKFp03': 'UKF-TD', 'EKFp03': 'EKF-TD', 'Adam3e-4ams': 'Adam 3e-4 (AMSGrad)', 'Adam1e-3ams': 'Adam 1e-3 (AMSGrad, 튜닝)', 'Adam3e-4': 'Adam 3e-4 (기본값)', 'Adam1e-3': 'Adam 1e-3 (기본값, 튜닝)'}
CO = {'SW': '#1f4e9c', 'UKFp03': '#6b8e23', 'EKFp03': '#8e5fa8', 'Adam3e-4ams': '#d1495b', 'Adam1e-3ams': '#c8860d', 'Adam3e-4': '#e08a95', 'Adam1e-3': '#e0b86a'}
S5 = [42, 43, 44, 45, 46]
def series(cell, l, key='reward', n=150):
    xs = [H[(cell, l, s)] for s in S5 if (cell, l, s) in H]
    if not xs: return None
    a = np.array([[h[i].get(key, np.nan) if i < len(h) else np.nan for i in range(n)] for h in xs], float)
    return a
def mv(y, k=20):
    out = np.full_like(y, np.nan, float)
    for i in range(len(y)):
        lo = max(0, i - k + 1); out[i] = np.nanmean(y[lo:i + 1])
    return out
# ── 그림 3: 학습곡선 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
for ax, cell, title in ((axes[0], 'per5_D1', '(a) 확약 없음 (매 스텝 결정, 표준 QCD)'), (axes[1], 'per5', '(b) 확약 5스텝 (기존)')):
    for l in ('SW', 'UKFp03', 'EKFp03', 'Adam3e-4ams'):
        a = series(cell, l)
        if a is None: continue
        m = np.array([mv(r) for r in a]); mu = np.nanmean(m, 0); sd = np.nanstd(m, 0)
        x = np.arange(len(mu))
        ax.plot(x, mu, color=CO[l], lw=1.8, label=f'{NM[l]} (n={a.shape[0]})')
        ax.fill_between(x, mu - sd, mu + sd, color=CO[l], alpha=0.13, lw=0)
    ax.set_xlabel('학습 에피소드'); ax.set_title(title, fontsize=10, loc='left')
    ax.grid(alpha=0.25, lw=0.5); ax.set_xlim(0, 149)
axes[0].set_ylabel('판당 보상 (20판 이동평균)'); axes[0].legend(fontsize=7.6, frameon=False, loc='lower right')
axes[0].set_ylim(60, 160)
fig.suptitle('그림 3 · 학습곡선 — 조건 지속 5판 무대, 시드 42–46 (띠 = 시드 표준편차)', fontsize=11, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f'{N}/fig3_learning_curves.png', dpi=200, bbox_inches='tight'); fig.savefig(f'{N}/fig3_learning_curves.pdf', bbox_inches='tight'); plt.close(fig)
# ── 그림 4: 무대별 짝 차이 forest ────────────────────────────
STAGES = [('per5_D1', '조건 지속 5판 · 확약 없음', 'reward', 30, 150), ('per5', '조건 지속 5판 · 확약 5', 'reward', 30, 150),
          ('per20', '조건 지속 20판 · 확약 5', 'reward', 30, 150), ('mix', '조건 고정 (평상시)', 'reward', 30, 150),
          ('blk', '첫 노출 급변 · 블록', 'reward', 60, 110), ('gblk', '돌풍 + 급변 · 블록', 'reward', 60, 110),
          ('dual300', '300판 후반 (250–299)', 'reward', 250, 300)]
COMPS = ['Adam3e-4ams', 'UKFp03', 'EKFp03']
def wm(h, key, a, b, atk=False):
    v = [float(h[i][key]) for i in range(a, min(b, len(h)))
         if h[i].get(key) is not None and not np.isnan(h[i][key]) and (not atk or h[i].get('has_atk'))]
    return float(np.mean(v)) if v else np.nan
rows = []
for cell, lab, key, a, b in STAGES:
    for comp in COMPS:
        d = [wm(H[(cell, 'SW', s)], key, a, b) - wm(H[(cell, comp, s)], key, a, b) for s in S5 if (cell, 'SW', s) in H and (cell, comp, s) in H]
        d = [x for x in d if not np.isnan(x)]
        if len(d) < 2: continue
        n = len(d); mu = float(np.mean(d)); se = float(np.std(d, ddof=1) / np.sqrt(n))
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
        rows.append((lab, comp, mu, tcrit * se, n, sum(1 for x in d if x > 0)))
fig, ax = plt.subplots(figsize=(8.6, 6.4))
ys = []; y = 0; labels = []
for cell, lab, key, a, b in STAGES:
    rs = [r for r in rows if r[0] == lab]
    if not rs: continue
    for j, (lab_, comp, mu, ci, n, pos) in enumerate(rs):
        ax.errorbar(mu, y, xerr=ci, fmt='o', ms=5.5, color=CO[comp], ecolor=CO[comp], elinewidth=1.4, capsize=3)
        ax.text(mu, y + 0.26, f'{pos}/{n}', ha='center', fontsize=6.6, color=CO[comp])
        ys.append(y); labels.append(f'{lab}  vs {NM[comp]}'); y += 1
    y += 0.6
ax.axvline(0, color='#333333', lw=1.0, ls='--')
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7.8); ax.invert_yaxis()
ax.set_xlabel('SWIRL − 비교군 (판당 보상, 무대별 1차 창)'); ax.grid(axis='x', alpha=0.25, lw=0.5)
h = [plt.Line2D([], [], color=CO[c], marker='o', ls='', label=NM[c]) for c in COMPS]
ax.legend(handles=h, fontsize=8, frameon=False, loc='lower right')
ax.set_title('그림 4 · 무대별 시드 짝 차이 (막대 = 95% 구간, 숫자 = SWIRL 우세 시드 수)', fontsize=10.5, loc='left')
fig.tight_layout(); fig.savefig(f'{N}/fig4_forest.png', dpi=200, bbox_inches='tight'); fig.savefig(f'{N}/fig4_forest.pdf', bbox_inches='tight'); plt.close(fig)
# ── 그림 5: 긴 지평 메커니즘 ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
for l in ('SW', 'UKFp03', 'EKFp03'):
    hs = [H[('dual300', l, s)] for s in (42, 43) if ('dual300', l, s) in H]
    if not hs: continue
    a = np.array([[h[i].get('pmax', np.nan) for i in range(300)] for h in hs], float)
    mu = np.nanmean(a, 0); ax.plot(np.arange(300), np.maximum(mu, 1e-3), color=CO[l], lw=1.7, label=f'{NM[l]} (n={len(hs)})')
ax.set_yscale('log'); ax.set_xlabel('학습 에피소드'); ax.set_ylabel('파라미터 공분산 최댓값 (log)')
ax.axvspan(60, 110, color='#888888', alpha=0.10, lw=0); ax.grid(alpha=0.25, lw=0.5)
ax.set_title('(a) 공분산 누적 — 무한 기억 대 유한 창', fontsize=10, loc='left'); ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ls = ['SW', 'UKFp03', 'EKFp03', 'Adam3e-4ams']; w = 0.35
for i, (key, off, lab) in enumerate((('f1', -w / 2, '후반 F1 250–299 (공격 판)'), ('reward', w / 2, '후반 보상/200'))):
    vals = []
    for l in ls:
        v = [wm(H[('dual300', l, s)], key, 250, 300, atk=(key == 'f1')) for s in (42, 43) if ('dual300', l, s) in H]
        vals.append(np.nanmean(v) / (1 if key == 'f1' else 200))
    ax.bar(np.arange(len(ls)) + off, vals, width=w, color=[CO[l] for l in ls], alpha=1.0 if key == 'f1' else 0.45,
           edgecolor='none', label=lab)
ax.set_xticks(np.arange(len(ls))); ax.set_xticklabels([NM[l] for l in ls], fontsize=8)
ax.set_ylim(0, 1.0); ax.grid(axis='y', alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False)
ax.set_title('(b) 후반 성능 (시드 2, 방향)', fontsize=10, loc='left')
fig.suptitle('그림 5 · 300판 학습 — 칼만-TD 공분산 누적과 후반 성능', fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'{N}/fig5_longhorizon.png', dpi=200, bbox_inches='tight'); fig.savefig(f'{N}/fig5_longhorizon.pdf', bbox_inches='tight'); plt.close(fig)
# ── 그림 6: Isaac 확인 ──────────────────────────────────────
def isaac(d):
    p = glob.glob(f'{R}/{d}/metrics_*.csv')
    if not p: return None
    rows = list(csv.DictReader(open(p[0])))
    out = {}
    for lab, lo, hi in (('진입 60–69', 60, 70), ('블록 60–109', 60, 110), ('후기 110+', 110, 200)):
        sel = [r for r in rows if lo <= int(r['episode']) < hi]
        na = [r for r in sel if float(r['bias_scale']) == 0]
        at = [r for r in sel if float(r['bias_scale']) > 0]
        out[lab] = dict(fp=np.mean([float(r['fp']) for r in na]) if na else np.nan,
                        f1=np.mean([float(r['f1']) for r in at]) if at else np.nan,
                        delay=np.mean([float(r['det_delay']) for r in at if float(r['det_delay']) >= 0]) if at else np.nan,
                        reward=np.mean([float(r['reward']) for r in sel]) if sel else np.nan)
    return out
PAIRS = [('시드 42', 'isaac_t2_swirl_s42', 'isaac_t_adam_s42'), ('시드 43', 'isaac_t2_swirl_s43', 'isaac_t_adam_s43')]
MET = [('진입 60–69', 'fp', '무공격 오경보 건수/판'), ('블록 60–109', 'fp', '무공격 오경보 건수/판'),
       ('블록 60–109', 'delay', '탐지 지연 [스텝]'), ('블록 60–109', 'f1', '공격 F1')]
fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4))
for ax, (win, key, lab) in zip(axes, MET):
    xs = np.arange(len(PAIRS)); w = 0.36
    sw = [isaac(a)[win][key] for _, a, _ in PAIRS]; ad = [isaac(b)[win][key] for _, _, b in PAIRS]
    ax.bar(xs - w / 2, sw, w, color=CO['SW'], label='SWIRL')
    ax.bar(xs + w / 2, ad, w, color=CO['Adam3e-4ams'], label='Adam 3e-4 (기본값)')
    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in PAIRS], fontsize=8)
    ax.set_title(f'{win}\n{lab}', fontsize=8.6, loc='left'); ax.grid(axis='y', alpha=0.25, lw=0.5)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle('그림 6 · Isaac Sim 확인 — 첫 노출 급변 블록, 시드 2 (갱신 10 ms < 스텝 예산 40 ms · 지연은 미탐지 판 제외)', fontsize=10.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f'{N}/fig6_isaac.png', dpi=200, bbox_inches='tight'); fig.savefig(f'{N}/fig6_isaac.pdf', bbox_inches='tight'); plt.close(fig)
print('저장: fig3_learning_curves / fig4_forest / fig5_longhorizon / fig6_isaac (.png/.pdf)')
for cell, lab, key, a, b in STAGES:
    got = [l for l in ['SW'] + COMPS if any((cell, l, s) in H for s in S5)]
    print(f'  {lab}: {got}')
