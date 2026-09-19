#!/usr/bin/env python3
"""scripts/leaderboard.py — 새 무대 전 런을 설정(시드 제외)별로 묶은 순위표. 판정 규칙 = 워크플로 wf_2b05db31.

    python3 scripts/leaderboard.py [결과 폴더 ...]   (기본: results/claudecodefortest/{newenv_Rregime,newenv_PQ_ll,night*})

행 = 설정 하나(시드 평균 ± 표준편차, n=시드 수). 후반 = ep100–199 학습 에피소드(공격 에피 F1·recall, 전체 FPR),
cost/c = 생존 보상 뺀 보상 ÷ 배율(후반 평균), ρL = Spearman(ep, loss), L후/최 = 후반 loss/최고점, Q/V = qmax/(alive·c/(1−γ)).
Δ짝 = 같은 공유 노브(eps·εz·buffer·c·γ·batch) Adam 과 같은 시드끼리 F1 차의 평균(시드 수).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml
from scipy.stats import spearmanr

SHARED = [('reward', 'scale'), ('agent', 'gamma'), ('agent', 'batch'), ('agent', 'buffer'), ('agent', 'n_step'),
          ('agent', 'eps', 'decay'), ('agent', 'eps', 'hover_p'), ('agent', 'eps', 'z_mu'), ('agent', 'eps', 'z_cap'), ('agent', 'hidden'), ('scenario', 'attack', 'family'), ('reward', 'mode')]
SWIRL = [('agent', 'swirl', 'R'), ('agent', 'swirl', 'p_delta'), ('agent', 'swirl', 'q'), ('agent', 'swirl', 'N'),
         ('agent', 'swirl', 'huber_c'), ('agent', 'tau'), ('agent', 'update_interval'), ('agent', 'swirl', 'argmax'), ('agent', 'swirl', 'act')]
SHORT = {'scale': 'c', 'gamma': 'γ', 'batch': 'B', 'buffer': 'buf', 'n_step': 'n', 'decay': 'εdec', 'hover_p': 'hp',
         'z_mu': 'z', 'z_cap': 'zcap', 'R': 'R', 'p_delta': 'pΔ', 'q': 'q', 'N': 'N', 'huber_c': 'hub', 'tau': 'τ',
         'update_interval': 'ui', 'argmax': 'arg', 'act': 'act', 'hidden': 'net', 'family': 'atk', 'mode': 'rw'}


def get(d, path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load(d):
    H = json.load(open(os.path.join(d, 'hist.json')))['hist']
    C = yaml.safe_load(open(os.path.join(d, 'config.yaml')))
    typ = C['agent']['type']; seed = C['run']['seed']
    sh = tuple((','.join(map(str, v)) if isinstance(v, list) else v) for v in (get(C, p) for p in SHARED))
    sw = tuple(get(C, p) for p in SWIRL) if typ != 'adam' else ()
    c = float(C['reward'].get('scale', 1)); g = float(C['agent'].get('gamma', 0.97)); al = float(C['reward'].get('alive', 0.5))
    late = H[100:]; atk = [h for h in late if h.get('has_atk')]
    L = np.array([h['loss'] for h in H])
    rc = defaultdict(list)
    for h in late:
        for k, v in (h.get('rec_by_cls') or {}).items(): rc[k].append(v)
    return dict(typ=typ, seed=seed, sh=sh, sw=sw, dir=d,
                f1=float(np.mean([h['f1'] for h in atk])), rec=float(np.mean([h['rec'] for h in atk])),
                fpr=float(np.mean([h['fpr'] for h in late])), cost=float(np.mean([h['reward_cost'] for h in late])) / c,
                rhoL=float(spearmanr(np.arange(len(L)), L)[0]), Lratio=float(L[len(L) * 3 // 4:].mean() / L.max()),
                qv=float(np.mean([h['qmax'] for h in late])) / (al * c / (1 - g)),
                weak=float(np.mean(rc.get('weak_persist', [np.nan]) + rc.get('weak_burst', [np.nan]))) if rc else np.nan)


def label(names, vals, base):
    return ' '.join(f'{SHORT[n[-1]]}{v}' for n, v, b in zip(names, vals, base) if v != b) or '기준'


def main():
    dirs = sys.argv[1:] or sorted(glob.glob('results/claudecodefortest/newenv_Rregime') + glob.glob('results/claudecodefortest/newenv_PQ_ll')
                                  + glob.glob('results/claudecodefortest/night*') + glob.glob('results/claudecodefortest/final*'))
    runs = [load(os.path.dirname(f)) for d in dirs for f in glob.glob(os.path.join(d, '*/hist.json'))]
    base_sh = (0.6, 0.97, 128, 50000, 3, 4000, 0.1, None, None, '16,16', 'profile', 'cost')
    base_sw = (0.03, 0.01, 0.001, 5, 5.0, 0.02, 4, 'spas', None)
    G = defaultdict(list)
    seen = set()
    for r in runs:                                     # 같은 설정·같은 시드 중복(결정론적 재실행)은 한 번만
        k = (r['typ'], r['sh'], r['sw'], r['seed'])
        if k in seen: continue
        seen.add(k); G[(r['typ'], r['sh'], r['sw'])].append(r)
    adam = {(r['sh'], r['seed']): r for r in runs if r['typ'] == 'adam'}
    rows = []
    for (typ, sh, sw), rs in G.items():
        m = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
        d = [r['f1'] - adam[(sh, r['seed'])]['f1'] for r in rs if (sh, r['seed']) in adam] if typ != 'adam' else []
        rows.append(dict(typ=typ, lab=label(SHARED, sh, base_sh) + (' | ' + label(SWIRL, sw, base_sw) if typ != 'adam' else ''),
                         n=len(rs), seeds=sorted(r['seed'] for r in rs), f1=m('f1'), rec=m('rec'), fpr=m('fpr'), cost=m('cost'),
                         rhoL=m('rhoL'), Lr=m('Lratio'), qv=m('qv'), weak=m('weak'), d=(float(np.mean(d)), len(d)) if d else None))
    rows.sort(key=lambda x: -x['f1'][0])
    print(f"{'학습기':6s} {'설정(기준 대비 바뀐 것)':44s} {'n':>2s} {'F1':>11s} {'recall':>6s} {'FPR':>6s} {'cost/c':>7s} {'약recall':>7s} {'ρL':>6s} {'L후/최':>6s} {'Q/V':>5s} {'Δ짝(Adam)':>10s}")
    for x in rows:
        dd = f"{x['d'][0]:+.3f}({x['d'][1]})" if x['d'] else ''
        print(f"{x['typ']:6s} {x['lab'][:44]:44s} {x['n']:2d} {x['f1'][0]:.3f}±{x['f1'][1]:.3f} {x['rec'][0]:6.3f} {x['fpr'][0]:6.4f} {x['cost'][0]:7.2f} {x['weak'][0]:7.3f} {x['rhoL'][0]:+6.2f} {x['Lr'][0]:6.2f} {x['qv'][0]:5.2f} {dd:>10s}")


if __name__ == '__main__':
    main()
