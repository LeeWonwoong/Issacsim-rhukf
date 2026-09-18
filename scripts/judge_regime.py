#!/usr/bin/env python3
"""scripts/judge_regime.py — "reward 상승 ∧ loss 하락" 판정 (CartPole/LL 레짐 격자, 2026-09-18 사전 고정 기준).

    python3 scripts/judge_regime.py <결과폴더> [--gamma 0.97]
입력: 새 형식(런 폴더: config.yaml + hist.json) 또는 구 형식(cp_regime_scan 의 <name>.json) — 둘 다 읽는다.
출력: <결과폴더>/REPORT.md · gates.json · curves_<stage>_<tu>.png · loss_ratio_vs_c2R.png

판정 (결과를 보기 전에 고정; L1 은 착수 전 스모크에서 Q 상승기 loss 상승을 보고 '초반 최고점' 기준으로 수정)
  L1 loss 초반최고점(ep0–59, 5판 이동평균) / loss(ep150–199) ≥ 2
  L2 Spearman(ep, loss | 최고점 → 끝) < −0.3
  L3 loss(175–199) ≤ 1.2 · loss(125–149)          (07-17 '느린 수렴 착시' 방지)
  R1 reward(150–199) > reward(0–9) ∧ Spearman(ep, reward) > +0.3
  R2 F1(150–199, 공격 에피) ≥ 0.80 ∧ FPR(150–199) ≤ 0.05
  Q  Qmax(150–199) ≤ 3 · V_exp,  V_exp = scale · r_tn / (1 − γ)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import yaml


def _rank(x): return np.argsort(np.argsort(x)).astype(float)


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1]) if m.sum() >= 5 else np.nan


def seg(h, k, a, b, atk=False):
    v = [r.get(k) for r in h[a:b] if (not atk or r.get('has_atk'))]
    v = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else np.nan


def load_runs(d):
    """[(name, meta, hist)] — meta: stage, agent, tu, c, R."""
    out = []
    for cy in sorted(glob.glob(os.path.join(d, '*', 'config.yaml'))):
        rd = os.path.dirname(cy); hp = os.path.join(rd, 'hist.json')
        if not os.path.exists(hp):
            continue
        c = yaml.safe_load(open(cy)); name = os.path.basename(rd); ag = c['agent']
        meta = dict(stage=name.split('_')[0], agent=('adam' if ag.get('type') == 'adam' else ag.get('type', 'swirl')),
                    tu=('ll' if int(ag.get('update_interval', 1)) == 4 else 'cp'), c=float(c['reward'].get('scale', 1.0)),
                    R=float(ag['swirl'].get('R', np.nan)) if ag.get('type') != 'adam' else np.nan,
                    r_tn=float(c['reward'].get('r_tn', 0.5)), gamma=float(ag.get('gamma', 0.97)))
        out.append((name, meta, json.load(open(hp))['hist']))
    for jp in sorted(glob.glob(os.path.join(d, '*_*.json'))):   # 구 cp_regime_scan 형식
        if jp.endswith('gates.json'):
            continue
        r = json.load(open(jp)); e = r.get('env', {})
        meta = dict(stage=r['name'].split('_')[0], agent=('adam' if r.get('agent') == 'adam' else 'swirl'),
                    tu=('ll' if e.get('RHUKF_UI') == '4' else 'cp'), c=float(e.get('REWARD_SCALE', 1.0)),
                    R=float(e.get('RHUKF_R', 'nan')), r_tn=0.5, gamma=0.97)
        out.append((r['name'], meta, r['hist']))
    return out


def gate(name, m, h, gamma=None):
    g0 = gamma if gamma is not None else m['gamma']
    ep = np.arange(len(h)); L = np.array([r['loss'] for r in h], float); Rw = np.array([r['reward'] for r in h], float)
    L_l = seg(h, 'loss', 150, 200)
    Ls = np.convolve(np.nan_to_num(L), np.ones(5) / 5, mode='same'); pk = int(np.argmax(Ls[:60])); L_pk = float(Ls[pk])
    V = m['c'] * m['r_tn'] / (1 - g0)
    g = dict(name=name, **m, n_ep=len(h), L_peak=L_pk, L_peak_ep=pk, L_late=L_l, L_ratio=(L_pk / L_l if L_l > 0 else np.nan),
             L_ratio_e=(seg(h, 'loss', 2, 12) / L_l if L_l > 0 else np.nan),
             L_rho=spear(ep[pk:], L[pk:]), L_tail=seg(h, 'loss', 175, 200) / seg(h, 'loss', 125, 150),
             R_early=seg(h, 'reward', 0, 10), R_late=seg(h, 'reward', 150, 200), R_rho=spear(ep, Rw),
             F1_late=seg(h, 'f1', 150, 200, True), FPR_late=seg(h, 'fpr', 150, 200), crash=int(sum(r.get('crashed', 0) for r in h)),
             qmax_late=seg(h, 'qmax', 150, 200), V_exp=V, kgain_late=seg(h, 'kgain', 150, 200),
             nisf_early=seg(h, 'nisf', 2, 12), nisf_late=seg(h, 'nisf', 150, 200), adapt_late=seg(h, 'adapt', 150, 200),
             aflip_late=seg(h, 'aflip', 150, 200), tvar_early=seg(h, 'tvar', 2, 12),
             probe_f1_late=seg(h, 'probe_f1', 150, 200))
    g['c2R'] = m['c'] ** 2 / m['R'] if np.isfinite(m['R']) else np.nan
    g['L1'] = bool(g['L_ratio'] >= 2); g['L2'] = bool(g['L_rho'] < -0.3); g['L3'] = bool(g['L_tail'] <= 1.2)
    g['R1'] = bool(g['R_late'] > g['R_early'] and g['R_rho'] > 0.3)
    g['R2'] = bool(g['F1_late'] >= 0.80 and g['FPR_late'] <= 0.05)
    g['Q'] = bool(np.isfinite(g['qmax_late']) and g['qmax_late'] <= 3 * V)
    g['PASS'] = all(g[k] for k in ('L1', 'L2', 'L3', 'R1', 'R2', 'Q'))
    return g


def report(d, runs, G):
    ok = lambda b: 'O' if b else '·'
    lines = ['# 레짐 격자 판정 — reward 상승 ∧ loss 하락', '', f'갱신 {time.strftime("%m-%d %H:%M")} · 런 {len(G)}', '',
             '| stage | 학습기 | τ/ui | c | R | c²/R | L 최고점@ep/후반 (비) | ρ(L) | L꼬리 | R 초→후 | ρ(R) | F1 | FPR | 추락 | Qmax/V | K | NIS 초→후 | huber팽창 | flip | L1 L2 L3 R1 R2 Q | 합격 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g in sorted(G, key=lambda g: (g['stage'], g['agent'], g['tu'], g['c2R'] if np.isfinite(g['c2R']) else g['c'])):
        lines.append(f"| {g['stage']} | {g['agent']} | {g['tu'] if g['agent'] != 'adam' else '-'} | {g['c']:g} | {g['R']:g} | {g['c2R']:.3g} | "
                     f"{g['L_peak']:.3g}@{g['L_peak_ep']}/{g['L_late']:.3g} ({g['L_ratio']:.2f}) | {g['L_rho']:+.2f} | {g['L_tail']:.2f} | "
                     f"{g['R_early']:.1f}→{g['R_late']:.1f} | {g['R_rho']:+.2f} | {g['F1_late']:.3f} | {g['FPR_late']:.3f} | {g['crash']} | "
                     f"{g['qmax_late'] / g['V_exp']:.2f} | {g['kgain_late']:.3g} | {g['nisf_early']:.2g}→{g['nisf_late']:.2g} | {g['adapt_late']:.2f} | "
                     f"{g['aflip_late']:.3f} | {ok(g['L1'])} {ok(g['L2'])} {ok(g['L3'])} {ok(g['R1'])} {ok(g['R2'])} {ok(g['Q'])} | {'**PASS**' if g['PASS'] else ''} |")
    open(os.path.join(d, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump(G, open(os.path.join(d, 'gates.json'), 'w'), indent=1, default=float)
    print('\n'.join(lines))


def plot(d, runs, G):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    sm = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode='valid') if len(x) >= k else np.asarray(x)
    gd = {g['name']: g for g in G}
    by = {}
    for name, m, h in runs:
        by[(m['stage'], m['agent'], m['tu'], m['c'], m['R'] if np.isfinite(m['R']) else None)] = (name, h)
    stages = sorted({k[0] for k in by}); cs = sorted({k[3] for k in by}); Rs = sorted({k[4] for k in by if k[4] is not None})
    for st in stages:
        for tu in ('cp', 'll'):
            keys = [k for k in by if k[0] == st and k[2] == tu and k[1] != 'adam']
            if not keys:
                continue
            fig, ax = plt.subplots(len(cs), len(Rs), figsize=(3.2 * len(Rs), 2.4 * len(cs)), squeeze=False)
            for i, c in enumerate(cs):
                for k, R in enumerate(Rs):
                    a = ax[i][k]; hit = [kk for kk in keys if kk[3] == c and kk[4] == R]
                    if not hit:
                        a.set_axis_off(); continue
                    name, h = by[hit[0]]
                    L = np.array([r['loss'] for r in h]); Rw = np.array([r['reward'] for r in h])
                    if (L > 0).any():
                        a.plot(sm(L), color='C3', lw=1); a.set_yscale('log')
                    a.tick_params(labelsize=6)
                    b = a.twinx(); b.plot(sm(Rw), color='C0', lw=1); b.tick_params(labelsize=6)
                    ad = [v for kk, v in by.items() if kk[0] == st and kk[1] == 'adam' and kk[3] == c]
                    if ad:
                        b.plot(sm(np.array([r['reward'] for r in ad[0][1]])), color='C0', lw=0.8, ls=':', alpha=0.7)
                    g = gd[name]
                    a.set_title(f"c{c:g} R{R:g} c²/R {g['c2R']:.2g} {'PASS' if g['PASS'] else ''}", fontsize=7,
                                color='green' if g['PASS'] else 'black')
            fig.suptitle(f'{st} · SWIRL {tu} — 빨강 loss(log, 좌) · 파랑 reward(우) · 점선 Adam reward', fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(d, f'curves_{st}_{tu}.png'), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, len(stages), figsize=(5 * len(stages), 3.6), squeeze=False)
    for a, st in zip(ax[0], stages):
        for g in G:
            if g['agent'] == 'adam' or g['stage'] != st:
                continue
            a.scatter(g['c2R'], g['L_ratio'], marker='o' if g['tu'] == 'cp' else 's', s=28,
                      c='green' if g['PASS'] else ('C1' if g['L1'] else 'C7'))
        a.set_xscale('log'); a.set_yscale('log'); a.axhline(2, ls='--', c='k', lw=0.7)
        a.set_xlabel('c²/R'); a.set_ylabel('loss 초반최고점/후반'); a.set_title(f'{st} (○ τ0.005/ui1 · □ τ0.02/ui4, 초록=합격)', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(d, 'loss_ratio_vs_c2R.png'), dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir')
    ap.add_argument('--gamma', type=float, default=None, help='구 형식 결과의 γ (기본 0.97)')
    a = ap.parse_args()
    runs = load_runs(a.dir)
    if not runs:
        print('결과 없음'); return
    G = [gate(n, m, h, a.gamma) for n, m, h in runs]
    report(a.dir, runs, G)
    try:
        plot(a.dir, runs, G)
    except Exception as e:
        print('그림 실패:', e)


if __name__ == '__main__':
    main()
