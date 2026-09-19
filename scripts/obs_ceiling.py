#!/usr/bin/env python3
"""scripts/obs_ceiling.py — 관측(12D 창)만으로 도달 가능한 탐지 상한: Isaac 실측 행에 지도학습 분류기(MLP) → 에피소드 분할 검증.

    python3 scripts/obs_ceiling.py

정책(SWIRL/Adam)의 recall 이 낮은 게 학습기 탓인지, 관측 자체의 한계(aliasing·지연·약공격)인지 가른다.
라벨 = atk_flag(보상 라벨과 동일, Isaac 규약 δ[t−1]). 클래스 = 그 스텝을 덮는 사건의 cls.
"""
import glob
import json
import os

import numpy as np
import torch

R = 'results/claudecodefortest/'
X, y, cls, ep = [], [], [], []
k = 0
for d in ['capture_pool', 'capture_pool_b', 'capture_branching', 'isaac_adam_probe']:
    for f in sorted(glob.glob(R + d + '/capture/ep*.npz') + glob.glob(R + d + '/steps/ep*.npz')):
        z = np.load(f, allow_pickle=False); cols = [str(c) for c in z['cols']]; Rw = z['rows']
        if len(Rw) < 8: continue
        S = Rw[:, [cols.index(f's{i}') for i in range(12)]]; a = Rw[:, cols.index('atk_flag')].astype(int); st = Rw[:, cols.index('step')].astype(int)
        ev = json.loads(str(z['events'])) if 'events' in z.files and len(str(z['events'])) > 2 else []
        if not ev and 'onset' in z.files and float(z['onset']) > 0:   # 분기 캡처: 단일 사건(grade_kind), 길이는 delta_eff>0 구간
            de = Rw[:, cols.index('delta_eff')]; idx = np.flatnonzero(de > 0)
            if len(idx): ev = [dict(start=int(st[idx[0]]) - 1, end=int(st[idx[-1]]), cls=f"{z['grade']}_{z['kind']}")]
        c = np.array(['none'] * len(Rw), dtype=object)
        for e in ev:
            m = (st - 1 >= e['start']) & (st - 1 < e['end']); c[m] = e['cls']
        X.append(S); y.append(a); cls.append(c); ep.append(np.full(len(Rw), k)); k += 1
X = np.concatenate(X).astype(np.float32); y = np.concatenate(y); cls = np.concatenate(cls); ep = np.concatenate(ep)
print(f'행 {len(X)} · 에피소드 {k} · 공격 행 {y.sum()} ({y.mean():.3f})')
rng = np.random.default_rng(0); test_ep = rng.random(k) < 0.25; te = test_ep[ep]; tr = ~te

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(12, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), 1e-3)
Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32)
for it in range(3000):
    i = torch.randint(0, len(Xt), (1024,))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[i]).squeeze(1), yt[i])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    p = torch.sigmoid(net(torch.tensor(X[te])).squeeze(1)).numpy()
yte, cte = y[te], cls[te]
# AUC
order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
auc = (ranks[yte == 1].sum() - yte.sum() * (yte.sum() + 1) / 2) / (yte.sum() * (len(yte) - yte.sum()))
print(f'검증 행 {te.sum()} · AUC {auc:.3f}')
K = ['strong_persist', 'trans_persist', 'weak_persist', 'strong_burst', 'trans_burst', 'weak_burst']
print(f"{'문턱':28s} {'FPR':>6s} {'recall':>6s} {'F1':>5s}  " + ' '.join(f"{x.replace('_persist','P').replace('_burst','B'):>8s}" for x in K))
def report(lab, thr):
    pred = p >= thr
    fpr = pred[yte == 0].mean(); rec = pred[yte == 1].mean(); prec = yte[pred].mean() if pred.any() else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"{lab:28s} {fpr:6.4f} {rec:6.3f} {f1:5.2f}  " + ' '.join(f"{pred[(cte == x) & (yte == 1)].mean():8.3f}" for x in K))
for fpr_t in (0.007, 0.02, 0.05):
    thr = np.quantile(p[yte == 0], 1 - fpr_t); report(f'FPR={fpr_t} 맞춤 (thr {thr:.2f})', thr)
report('P(공격)>0.7 (보상 손익분기 근사)', 0.7)
report('P(공격)>0.5', 0.5)
# 지연 제한 오라클: 사건 길이 dur, 탐지 지연 k 스텝이면 recall=(dur−k)/dur
print('\n지연 제한 오라클(관측 무관, 사건 길이만): 지연 k 스텝일 때 스텝 recall')
durs = {}
for d in ['capture_pool', 'capture_pool_b', 'isaac_adam_probe']:
    for f in glob.glob(R + d + '/capture/ep*.npz') + glob.glob(R + d + '/steps/ep*.npz'):
        z = np.load(f, allow_pickle=False)
        for e in (json.loads(str(z['events'])) if 'events' in z.files and len(str(z['events'])) > 2 else []):
            durs.setdefault(e['cls'], []).append(e['end'] - e['start'])
for kk in (1, 2, 3):
    print(f"  k={kk}: " + ' '.join(f"{x.replace('_persist','P').replace('_burst','B')} {np.mean([max(0, dd - kk) / dd for dd in durs[x]]):.2f}" for x in K))
