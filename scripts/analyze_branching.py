#!/usr/bin/env python3
"""scripts/analyze_branching.py — burst vs persistent 관측 분기 판정 (설계 검증 Step 1).

    python3 scripts/analyze_branching.py <캡처 결과폴더>     (폴더/capture/ep*.npz — online_rl_main 캡처 모드 산출)
출력: <폴더>/BRANCHING.md · branching.json · branching_traj.png · branching_auc.png

무엇을 보나 — 정책이 실제로 받는 관측
  프레임 = [ε̃_vel, ε̃_gyro] (ObsSpec 로 압축·정규화된 값),  창 = 최근 4 프레임(행동 칸 제외 8 차원)
  오프셋 k = 창 마지막 스텝 − 온셋.  k=3 → 창 = 온셋 t0..t3 (prefix 구간)

판정 (결과를 보기 전에 고정)
  겹침   OVERLAP  : AUC_burst↔persist(k=3) ≤ 0.70           — prefix 창만으로는 형태를 못 가린다
  분기   DIVERGE  : k ∈ [4, 15] 에서 AUC ≥ 0.85 가 3 스텝 연속 — 이후 창에서는 가린다 (첫 k = '분기 시점')
  가시성 VISIBLE  : AUC_attack↔none(k=3) — prefix 창에서 공격 자체는 보이나 (정보용; 약공격은 낮을 수 있음)
  설계 성립 = 등급별 OVERLAP ∧ DIVERGE.  (강 등급은 persistent 가 성장하지 않아 분기가 약할 수 있음 — 정보용)
AUC: 짝 단위 leave-one-pair-out 교차검증 Fisher 판별(공분산 릿지). 짝 = (d0·성장률·온셋·방향·패턴·풍속) 공유.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

K_MIN, K_MAX = -2, 30


def load(d):
    eps = []
    for f in sorted(glob.glob(os.path.join(d, 'capture', 'ep*.npz'))):
        z = np.load(f, allow_pickle=False)
        cols = [str(c) for c in z['cols']]; R = z['rows']
        if R.ndim != 2 or len(R) == 0:
            continue
        C = {c: R[:, i] for i, c in enumerate(cols)}
        eps.append(dict(file=os.path.basename(f), pair=str(z['pair']), grade=str(z['grade']), kind=str(z['kind']),
                        onset=int(z['onset']), step=C['step'].astype(int), v=C['v_obs'], g=C['g_obs'],
                        vr=C['nis_v_raw'], gr=C['nis_g_raw'], delta=C['delta'], reason=str(z['reason'])))
    return eps


def window(ep, k):
    """오프셋 k 에서 끝나는 4 프레임 창 [v,g]×4 (없으면 None)."""
    idx = {s: i for i, s in enumerate(ep['step'])}
    t = ep['onset'] + k
    rows = [idx.get(t - 3 + j) for j in range(4)]
    if any(r is None for r in rows):
        return None
    return np.array([[ep['v'][r], ep['g'][r]] for r in rows]).ravel()


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def fisher_lopo(X, y, groups, ridge=1e-2):
    """leave-one-group-out Fisher 판별 점수 → AUC."""
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups)
    scores = np.full(len(y), np.nan)
    for gname in np.unique(groups):
        te = groups == gname; tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        m1, m0 = Z[tr & (y == 1)].mean(0), Z[tr & (y == 0)].mean(0)
        S = np.cov(Z[tr].T) + ridge * np.eye(Z.shape[1])
        w = np.linalg.solve(S, m1 - m0)
        scores[te] = Z[te] @ w
    ok = np.isfinite(scores)
    return auc(scores[ok & (y == 1)], scores[ok & (y == 0)])


def analyze(eps):
    out = {}
    grades = sorted({e['grade'] for e in eps if e['grade'] != 'none'})
    nones = [e for e in eps if e['kind'] == 'none']
    for gr in grades:
        E = [e for e in eps if e['grade'] == gr]
        pairs = sorted({e['pair'] for e in E})
        res = dict(pairs=len(pairs), k=[], auc_bp=[], auc_an=[], n=[])
        for k in range(K_MIN, K_MAX + 1):
            X, y, G = [], [], []
            for e in E:
                w = window(e, k)
                if w is not None:
                    X.append(w); y.append(1 if e['kind'] == 'persistent' else 0); G.append(e['pair'])
            a_bp = fisher_lopo(X, y, G) if len(set(y)) == 2 and len(set(G)) >= 4 else np.nan
            Xa = [window(e, k) for e in E if e['kind'] == 'burst']; Xn = [window(e, k) for e in nones]
            Xa = [w for w in Xa if w is not None]; Xn = [w for w in Xn if w is not None]
            a_an = np.nan
            if len(Xa) >= 3 and len(Xn) >= 3:   # 공격(burst — prefix 는 두 형태 동일) vs 무공격, 무공격은 에피소드별 그룹
                a_an = fisher_lopo(Xa + Xn, [1] * len(Xa) + [0] * len(Xn),
                                   [f'a{i}' for i in range(len(Xa))] + [f'n{i}' for i in range(len(Xn))])
            res['k'].append(k); res['auc_bp'].append(a_bp); res['auc_an'].append(a_an); res['n'].append(len(y))
        K = np.array(res['k']); A = np.array(res['auc_bp'])
        a3 = float(A[K == 3][0]) if (K == 3).any() else np.nan
        div_k = None
        for k0 in range(4, 16):
            seg = [A[K == k0 + j][0] for j in range(3) if (K == k0 + j).any()]
            if len(seg) == 3 and all(np.isfinite(seg)) and min(seg) >= 0.85:
                div_k = k0; break
        res.update(auc_k3=a3, overlap=bool(np.isfinite(a3) and a3 <= 0.70), diverge_k=div_k, diverge=div_k is not None,
                   visible_k3=float(np.array(res['auc_an'])[K == 3][0]) if (K == 3).any() else np.nan,
                   holds=bool(np.isfinite(a3) and a3 <= 0.70 and div_k is not None),
                   crashes={kind: sum(1 for e in E if e['kind'] == kind and e['reason'] in ('crash_drift', 'crash_altitude', 'crash_flip'))
                            for kind in ('burst', 'persistent')})
        out[gr] = res
    return out


def traj_stats(eps, gr, kind, key):
    E = [e for e in eps if e['grade'] == gr and e['kind'] == kind] if kind != 'none' else [e for e in eps if e['kind'] == 'none']
    ks = np.arange(K_MIN - 3, K_MAX + 1)
    M = np.full((len(E), len(ks)), np.nan)
    for i, e in enumerate(E):
        idx = {s: j for j, s in enumerate(e['step'])}
        for j, k in enumerate(ks):
            r = idx.get(e['onset'] + k)
            if r is not None:
                M[i, j] = e[key][r]
    return ks, np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)


def plots(d, eps, res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'axes.unicode_minus': False})
    SURF, INK2, GRID, AXIS = '#fcfcfb', '#52514e', '#e1e0d9', '#c3c2b7'
    COL = {'burst': '#2a78d6', 'persistent': '#eb6834', 'none': '#898781'}
    grades = list(res)
    fig, ax = plt.subplots(2, len(grades), figsize=(4.6 * len(grades), 6.4), squeeze=False, facecolor=SURF, sharex=True)
    for j, gr in enumerate(grades):
        for i, key in enumerate(['g', 'v']):
            a = ax[i][j]
            for kind in ('none', 'burst', 'persistent'):
                ks, med, lo, hi = traj_stats(eps, gr, kind, key)
                a.plot(ks, med, color=COL[kind], lw=2, label={'none': '무공격', 'burst': 'burst', 'persistent': 'persistent'}[kind])
                a.fill_between(ks, lo, hi, color=COL[kind], alpha=0.15, lw=0)
            a.axvspan(0, 3, color='#e1e0d9', alpha=0.5, lw=0)
            a.set_title(f'{gr} — {"ε̃_gyro" if key == "g" else "ε̃_vel"} (중앙값·IQR)', fontsize=10, loc='left')
            a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6)
            for s in ('top', 'right'): a.spines[s].set_visible(False)
            for s in ('left', 'bottom'): a.spines[s].set_color(AXIS)
            a.tick_params(colors=INK2, labelsize=8)
        ax[1][j].set_xlabel('온셋 기준 스텝 k (회색 = prefix t0–t3)', color=INK2)
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(d, 'branching_traj.png'), dpi=120, facecolor=SURF); plt.close(fig)
    fig, a = plt.subplots(figsize=(7.5, 3.8), facecolor=SURF)
    ramp = ['#86b6ef', '#2a78d6', '#104281']
    for c, gr in zip(ramp, grades):
        a.plot(res[gr]['k'], res[gr]['auc_bp'], color=c, lw=2, label=f'{gr}: burst↔persistent')
    a.axhline(0.85, color=INK2, lw=1, ls=(0, (4, 3))); a.axhline(0.70, color=INK2, lw=1, ls=(0, (1, 2)))
    a.axvspan(0, 3, color='#e1e0d9', alpha=0.5, lw=0); a.set_ylim(0.3, 1.02)
    a.set_xlabel('창 끝 오프셋 k', color=INK2); a.set_ylabel('교차검증 AUC', color=INK2)
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6); a.legend(frameon=False, fontsize=8)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, 'branching_auc.png'), dpi=120, facecolor=SURF); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir')
    a = ap.parse_args()
    eps = load(a.dir)
    if not eps:
        print('캡처 없음'); return
    res = analyze(eps)
    L = ['# burst vs persistent 관측 분기 판정', '', f'에피소드 {len(eps)} (무공격 {sum(e["kind"] == "none" for e in eps)})', '',
         '| 등급 | 짝 | AUC(k=3, prefix 창) | 겹침 ≤0.70 | 분기 시점 k (AUC≥0.85 3연속) | 공격 가시성 AUC(k=3) | 추락 burst/persist | 설계 성립 |',
         '|---|---|---|---|---|---|---|---|']
    for gr, r in res.items():
        L.append(f"| {gr} | {r['pairs']} | {r['auc_k3']:.2f} | {'O' if r['overlap'] else '·'} | {r['diverge_k'] if r['diverge'] else '없음'} | "
                 f"{r['visible_k3']:.2f} | {r['crashes']['burst']}/{r['crashes']['persistent']} | {'**성립**' if r['holds'] else '·'} |")
    open(os.path.join(a.dir, 'BRANCHING.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(os.path.join(a.dir, 'branching.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))
    try:
        plots(a.dir, eps, res)
    except Exception as e:
        print('그림 실패:', e)


if __name__ == '__main__':
    main()
