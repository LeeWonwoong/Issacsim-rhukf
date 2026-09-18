#!/usr/bin/env python3
"""scripts/compare_runs.py — 새 무대 판정표(워크플로 wf_2b05db31 규칙): probe 대신 학습 에피소드 기준.

    python3 scripts/compare_runs.py <런 폴더> [<런 폴더> ...]

열: 후반(ep100–199) 공격 에피 평균 F1·recall·FPR, 클래스별 recall, FVU(loss/tvar) 구간 평균, Qmax/V (V=alive·c/(1−γ)).
같은 시드 = 같은 공격 일정(짝비교). probe(시드 고정 4ep)는 시드 간 비교에 쓰지 않는다.
"""
import json
import os
import sys

import numpy as np
import yaml


def row(d):
    H = json.load(open(os.path.join(d, 'hist.json')))['hist']
    C = yaml.safe_load(open(os.path.join(d, 'config.yaml')))
    c = float(C['reward'].get('scale', 1)); g = float(C['agent'].get('gamma', 0.97)); al = float(C['reward'].get('alive', 0.5))
    V = al * c / (1 - g)
    late = H[100:]
    atk = [h for h in late if h.get('has_atk')]
    f = lambda k, hs: float(np.nanmean([h[k] for h in hs])) if hs else float('nan')
    rc = {}
    for h in late:
        for k, v in (h.get('rec_by_cls') or {}).items():
            rc.setdefault(k, []).append(v)
    fvu = lambda a, b: float(np.nanmean([h['loss'] / h['tvar'] for h in H[a:b] if h.get('tvar')]))
    return dict(run=os.path.basename(d.rstrip('/')), f1=f('f1', atk), rec=f('rec', atk), fpr=f('fpr', late),
                qv=f('qmax', late) / V, fvu=(fvu(5, 30), fvu(30, 100), fvu(100, 200)),
                cls={k: float(np.mean(v)) for k, v in rc.items()})


def main():
    K = ['strong_persist', 'trans_persist', 'weak_persist', 'strong_burst', 'trans_burst', 'weak_burst']
    print(f"{'런':40s} {'F1':>5s} {'rec':>5s} {'FPR':>6s} {'Q/V':>5s}  FVU(5-30/30-100/100-200)  " + ' '.join(k.replace('_persist', 'P').replace('_burst', 'B') for k in K))
    for d in sys.argv[1:]:
        if not os.path.exists(os.path.join(d, 'hist.json')):
            continue
        r = row(d)
        print(f"{r['run']:40s} {r['f1']:5.2f} {r['rec']:5.2f} {r['fpr']:6.3f} {r['qv']:5.2f}  {r['fvu'][0]:.2f}/{r['fvu'][1]:.2f}/{r['fvu'][2]:.2f}          "
              + ' '.join(f"{r['cls'].get(k, float('nan')):7.2f}" for k in K))


if __name__ == '__main__':
    main()
