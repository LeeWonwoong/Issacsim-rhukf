#!/usr/bin/env python3
"""scripts/grid_report.py — 격자 결과 한 장 요약: 학습기×축 셀별 지표 + 같은 축의 SWIRL−Adam(또는 지정 짝) 시드 짝비교.

    python3 scripts/grid_report.py results/claudecodefortest/n13_pdelta_ut [--pair s a] [--base <다른 격자 dir>:<학습기 접두>]

셀 이름 규약(grid.py): a_<learner>_<axis1><val1>_<axis2><val2>_s<seed>. 지표:
  greedy(eval.json): F1 · FA에피 · 사건탐지 · 지연 | 학습중(hist ep100–199): F1 · cost/c · 약공격 recall · Qmax/V
  프로브(hist probe_f1, probe_every 5): ≥0.8/≥0.85 도달 ep(MA3) · ep150+ 평균
"""
import glob
import json
import os
import re
import sys

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:                       # scipy 없으면 p 생략
    wilcoxon = None

import yaml


def cell_scale(d):
    """셀 config.yaml 에서 (reward.scale, alive, γ) → cost/c 정규화·Q/V 기준(alive·c/(1−γ))."""
    c = yaml.safe_load(open(os.path.join(d, 'config.yaml')))
    r = c.get('reward', {}) or {}; a = c.get('agent', {}) or {}
    return float(r.get('scale', 1.0)), float(r.get('alive', 0.5)), float(a.get('gamma', 0.97))


def load_cell(d):
    H = json.load(open(os.path.join(d, 'hist.json')))['hist']
    ev = json.load(open(os.path.join(d, 'eval.json'))) if os.path.exists(os.path.join(d, 'eval.json')) else {}
    c, alive, gam = cell_scale(d); q_alive = alive * c / (1 - gam)
    late = H[100:200]
    r = dict(f1=ev.get('f1', np.nan), fa=ev.get('fa_episode_rate', np.nan), ed=ev.get('event_det', np.nan), delay=ev.get('delay', np.nan),
             tf1=np.nanmean([h['f1'] if h['has_atk'] else np.nan for h in late]),
             cost=np.mean([h['reward_cost'] for h in late]) / c,
             wr=np.nanmean([h['wrec'] for h in late if h['wrec'] == h['wrec']]) if any(h['wrec'] == h['wrec'] for h in late) else np.nan,
             qv=np.mean([h['qmax'] for h in H[150:200]]) / q_alive)
    pf = np.array([(h['ep'], h['probe_f1']) for h in H if h.get('probe_f1') is not None and h['ep'] % 5 == 0 and h['ep'] > 0])
    if len(pf) >= 4:
        m = np.convolve(pf[:, 1], np.ones(3) / 3, 'same')
        for thr in (0.8, 0.85):
            i = np.flatnonzero(m >= thr); r[f'r{int(thr*100)}'] = pf[i[0], 0] if len(i) else 999
        r['late'] = pf[pf[:, 0] >= 150, 1].mean()
    else:
        r['r80'] = r['r85'] = r['late'] = np.nan
    return r


def main():
    argv = sys.argv[1:]; root = argv[0]
    pair = ('s', 'a')
    if '--pair' in argv: i = argv.index('--pair'); pair = (argv[i + 1], argv[i + 2])
    cells = {}
    for d in sorted(glob.glob(os.path.join(root, 'a_*_s[0-9]*'))):
        if not os.path.exists(os.path.join(d, 'hist.json')): continue
        m = re.match(r'a_([A-Za-z]+)_(.*)_s(\d+)$', os.path.basename(d))
        if not m: continue
        learner, axes, seed = m.groups()
        cells.setdefault((learner, axes), {})[seed] = load_cell(d)
    cols = [('f1', 'greedy F1'), ('fa', 'FA에피'), ('ed', '사건'), ('delay', '지연'), ('tf1', '학습중F1'), ('cost', 'cost/c'), ('wr', '약'), ('qv', 'Q/V'), ('r80', '≥.8ep'), ('r85', '≥.85ep'), ('late', '프로브150+')]
    print(f"{'학습기':6s} {'축':34s} n  " + ' '.join(f'{n:>9s}' for _, n in cols))
    for (L, ax), v in sorted(cells.items()):
        vals = []
        for k, _ in cols:
            x = np.array([c[k] for c in v.values()], float)
            vals.append(np.nanmedian(x) if k in ('r80', 'r85') else np.nanmean(x))
        print(f'{L:6s} {ax:34s} {len(v):2d} ' + ' '.join(f'{x:9.3f}' if k not in ('r80', 'r85') else f'{x:9.0f}' for (k, _), x in zip(cols, vals)))
    # 짝비교
    A = {ax: v for (L, ax), v in cells.items() if L == pair[0]}; B = {ax: v for (L, ax), v in cells.items() if L == pair[1]}
    common = sorted(set(A) & set(B))
    if common:
        print(f'\n짝비교 {pair[0]}−{pair[1]} (같은 축·같은 시드; ≥.8ep 는 음수가 {pair[0]} 빠름):')
        for ax in common:
            seeds = sorted(set(A[ax]) & set(B[ax])); out = []
            for k, nm in (('f1', 'greedyF1'), ('cost', 'cost/c'), ('tf1', '학습중F1'), ('wr', '약'), ('fa', 'FA'), ('late', '프로브150+'), ('r80', '≥.8ep')):
                dd = np.array([A[ax][s][k] - B[ax][s][k] for s in seeds], float); dd = dd[~np.isnan(dd)]
                if not len(dd): continue
                win = (dd < 0).sum() if k in ('fa', 'r80') else (dd > 0).sum()
                p = wilcoxon(dd).pvalue if (wilcoxon and np.any(dd)) else float('nan')
                out.append(f'{nm} {dd.mean():+.3f} ({win}/{len(dd)}, p={p:.3f})')
            print(f'  {ax:34s} ' + ' | '.join(out))


if __name__ == '__main__':
    main()
