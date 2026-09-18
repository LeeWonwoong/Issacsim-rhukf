#!/usr/bin/env python3
"""agg_b64 — 배치 64 vs 128 (같은 셀·시드, 에피소드 앞부분 공통 창). 사용: B64_CELL=<ext 셀> python3 agg_b64.py → night/B64_RESULT.md
   배치 128 원본 후보: ext 셀 'x<cell>', tonight/S1 셀 '<cell>' (blk 는 S1 'g90n3' 포함). SWIRL 은 SW*(128) ↔ SW*-N{2N}(64)."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf/etc/scripts'); os.environ.setdefault('CHAIN_DIR', '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/chain_hz')
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'
cell = os.environ.get('B64_CELL', 'per20'); SWs = os.environ.get('B64_SW', os.environ.get('EXT_SW', 'SW')); nep = int(os.environ.get('B64_NEP', '110'))
H = {}
for f in glob.glob(f'{N}/tonight/q/*.json') + glob.glob(f'{N}/tonight/[TUA]*_w*.json') + glob.glob(f'{N}/chain_hz/S1*_w*.json'):
    for k, m in json.load(open(f)).items():
        c = 'blk' if m['cell'] == 'g90n3' else m['cell']; H[(c, m['learner'], m['seed'])] = m['hist']
import chain_hz as CH
nstar = int(CH.learner_env(SWs, c='g90n3')['RHUKF_N']); sw64 = f'{SWs}-N{2 * nstar}'
pairs = [(SWs, sw64, 'SWIRL'), ('UKFp03', 'UKFp03', 'UKF-TD'), ('EKFp03', 'EKFp03', 'EKF-TD'), ('UKFp1', 'UKFp1', 'UKF-TD P₀0.1'), ('EKFp1', 'EKFp1', 'EKF-TD P₀0.1'), ('Adam3e-4', 'Adam3e-4', 'Adam 3e-4'), (os.environ.get('EXT_ADAM', 'Adam1e-3'), os.environ.get('EXT_ADAM', 'Adam1e-3'), 'Adam 튜닝')]
def wm(h, key, a, b, atk=False):
    v = [float(x[key]) for x in h[a:b] if x.get(key) is not None and (not atk or x.get('has_atk'))]; v = [x for x in v if not np.isnan(x)]; return float(np.mean(v)) if v else np.nan
W = [('보상 0–%d' % nep, lambda h: wm(h, 'reward', 0, nep)), ('후반 보상 %d–%d' % (nep - 30, nep), lambda h: wm(h, 'reward', nep - 30, nep)), ('오탐 20–%d' % nep, lambda h: wm(h, 'fpr', 20, nep)),
     ('후반 공격 F1', lambda h: wm(h, 'f1', nep - 30, nep, True)), ('진입 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70))]
L = [f'# 배치 64 vs 128 — 셀 {cell}, 에피소드 앞 {nep}개 공통 창 (SWIRL 창 N {nstar}→{2 * nstar}, 창 표본 B·N 동일)', '', '| 학습기 | 시드 | ' + ' | '.join(f'{w[0]} 128 → 64' for w in W) + ' |', '|---|---|' + '---|' * len(W)]
for l128, l64, nm in pairs:
    for sd in sorted({k[2] for k in H if k[0] == f'b64{cell}' and k[1] == l64}):
        h64 = H.get((f'b64{cell}', l64, sd)); h128 = H.get((f'x{cell}', l128, sd)) or H.get((cell, l128, sd))
        if h64 is None or h128 is None: L.append(f'| {nm} | {sd} | 배치 128 대조 없음 |' + ' |' * (len(W) - 1)); continue
        L.append(f'| {nm} | {sd} | ' + ' | '.join(f'{fn(h128):.3f} → {fn(h64):.3f}' for _, fn in W) + ' |')
L += ['', '- 시드 2개 탐색 수준. 판정은 방향만(배치 64 에서 SWIRL−Adam·SWIRL−칼만-TD 격차가 커지는지).']
open(f'{N}/B64_RESULT.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
