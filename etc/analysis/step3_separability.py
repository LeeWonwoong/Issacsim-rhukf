#!/usr/bin/env python3
"""
step3_separability.py — 관측 동결 관문: 5-클래스 분리도 진단
=================================================================
sweep detail CSV의 per-step [nis_v_scaled(log0.5), nis_g_scaled(log1p), action]을
윈도우4(12차원)로 묶어 RL 관측공간에서 5클래스가 분리되는지 정량화.

클래스(소스):
  정상비행 = w0 hover        (무풍·무공격 정지비행, action=1)
  급기동   = w0 track        (aggressive 패턴 기동, action=0)
  호버전환 = w0 dhover3 전환창 (track→hover 스위치 과도, action 0→1)
  공격     = combined_final track attack_active=1, bias∈[1.34,1.40] (action=0)
  외란     = w7 track        (수정 turbulence ws7, 무공격, action=0)

핵심: 급기동/공격/외란은 모두 action=0 → action이 못 가름 → NIS 채널이 갈라야 함(=aliasing 핵심).
측정: 5클래스 LDA 정확도+혼동행렬 / 쌍별 마할라노비스 d' / action ablation / Bayes error 하한.
사용: python3 step3_separability.py
"""
import csv, sys
import numpy as np
from collections import defaultdict

W0   = 'results_wind_v2_w0/sweep_detail.csv'
W7   = 'results_wind_v2_w7/sweep_detail.csv'
ATK  = 'results_combined_final/sweep_detail.csv'
BAND = {'1.340', '1.360', '1.380', '1.400'}
WIN  = 4
SEED = 0

CLASSES = ['정상비행', '급기동', '호버전환', '공격', '외란']


def load_rows(path):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append(dict(
                cell=r['cell_idx'], ep=r['episode'], step=int(r['step']),
                pol=r['policy'], atk=r['attack_active'], bias=r['bias'],
                nv=float(r['nis_v_scaled']), ng=float(r['nis_g_scaled']),
                act=float(r['action'])))
        except (ValueError, KeyError):
            pass
    return rows


def episodes(rows):
    """(cell,ep)별로 그룹, step 정렬."""
    g = defaultdict(list)
    for r in rows:
        g[(r['cell'], r['ep'])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r['step'])
    return g


def windows(ep_rows):
    """연속 step 4개 → 12차원 [nv,ng,act]×4. 에피소드 경계/불연속 건너뜀."""
    out = []
    for rows in ep_rows.values():
        for i in range(len(rows) - WIN + 1):
            seg = rows[i:i + WIN]
            if seg[-1]['step'] - seg[0]['step'] != WIN - 1:
                continue
            v = []
            for s in seg:
                v += [s['nv'], s['ng'], s['act']]
            out.append(v)
    return np.array(out, float)


def build_classes():
    w0 = load_rows(W0); w7 = load_rows(W7); atk = load_rows(ATK)
    data = {}
    # 정상비행 = w0 hover
    data['정상비행'] = windows(episodes([r for r in w0 if r['pol'] == 'hover']))
    # 급기동 = w0 track
    data['급기동'] = windows(episodes([r for r in w0 if r['pol'] == 'track']))
    # 호버전환 = w0 dhover3 전환창: 에피소드별 첫 action==1 스텝(switch) 기준 [switch-1, switch+12]
    trans = []
    for (c, e), rows in episodes([r for r in w0 if r['pol'] == 'dhover3']).items():
        sw = next((r['step'] for r in rows if r['act'] > 0.5), None)
        if sw is None:
            continue
        trans += [r for r in rows if sw - 2 <= r['step'] <= sw + 30]  # 전환 과도창 확대
    data['호버전환'] = windows(episodes(trans))
    # 공격 = combined_final track attack_active=1 밴드
    data['공격'] = windows(episodes(
        [r for r in atk if r['pol'] == 'track' and r['atk'] == '1' and r['bias'] in BAND]))
    # 외란 = w7 track (turbulence on, 무공격)
    data['외란'] = windows(episodes([r for r in w7 if r['pol'] == 'track']))
    return data


# ── LDA (공유 공분산 가우시안 판별) ──
def lda_fit(Xtr, ytr, K):
    d = Xtr.shape[1]
    means = np.zeros((K, d)); S = np.zeros((d, d)); nk = np.zeros(K)
    for c in range(K):
        Xc = Xtr[ytr == c]; nk[c] = len(Xc)
        means[c] = Xc.mean(0)
        S += (Xc - means[c]).T @ (Xc - means[c])
    S /= (len(Xtr) - K)
    S += 1e-6 * np.eye(d)
    Sinv = np.linalg.pinv(S)
    priors = nk / nk.sum()
    return means, Sinv, priors


def lda_predict(X, means, Sinv, priors):
    K = len(means)
    scores = np.zeros((len(X), K))
    for c in range(K):
        diff = X - means[c]
        scores[:, c] = -0.5 * np.einsum('ij,jk,ik->i', diff, Sinv, diff) + np.log(priors[c] + 1e-12)
    return scores.argmax(1)


def maha_dprime(Xa, Xb):
    """두 클래스 평균의 마할라노비스 거리(pooled cov) = 다변량 d'."""
    if len(Xa) < 20 or len(Xb) < 20:
        return 0.0
    S = 0.5 * (np.cov(Xa, rowvar=False) + np.cov(Xb, rowvar=False)) + 1e-6 * np.eye(Xa.shape[1])
    diff = Xa.mean(0) - Xb.mean(0)
    return float(np.sqrt(diff @ np.linalg.solve(S, diff)))


def gauss_bayes_err(dprime):
    """등분산 가우시안 가정 두 클래스 Bayes error = Φ(-d'/2)."""
    from math import erf, sqrt
    return 0.5 * (1 - erf((dprime / 2) / sqrt(2)))


def split(data, K, frac=0.7):
    rng = np.random.default_rng(SEED)
    Xtr, ytr, Xte, yte = [], [], [], []
    for c, name in enumerate(CLASSES):
        X = data[name]
        idx = rng.permutation(len(X)); n = int(len(X) * frac)
        Xtr.append(X[idx[:n]]); ytr += [c] * n
        Xte.append(X[idx[n:]]); yte += [c] * (len(X) - n)
    return (np.vstack(Xtr), np.array(ytr), np.vstack(Xte), np.array(yte))


def run(data, use_action=True, tag=''):
    K = len(CLASSES)
    if not use_action:
        # action 채널(인덱스 2,5,8,11) 제거 → 8차원
        keep = [i for i in range(12) if i % 3 != 2]
        data = {k: v[:, keep] for k, v in data.items()}
    Xtr, ytr, Xte, yte = split(data, K)
    means, Sinv, priors = lda_fit(Xtr, ytr, K)
    pred = lda_predict(Xte, means, Sinv, priors)
    acc = (pred == yte).mean()
    dim = Xtr.shape[1]
    print(f'\n{"="*66}\n[LDA {tag}] {dim}차원, train {len(Xtr)} / test {len(Xte)}  전체정확도 = {acc:.3f}')
    # 혼동행렬 (row=true, col=pred) 정규화
    print('  혼동행렬(행=실제, 열=예측, 행정규화):')
    print('           ' + ''.join(f'{c[:4]:>8}' for c in CLASSES))
    for c in range(K):
        m = yte == c
        row = [(pred[m] == j).mean() if m.sum() else 0 for j in range(K)]
        print(f'    {CLASSES[c]:<6}' + ''.join(f'{x:8.2f}' for x in row) + f'   (n={m.sum()})')
    return acc


def main():
    data = build_classes()
    print('클래스별 윈도우 수:')
    for c in CLASSES:
        print(f'  {c:<6} {len(data[c]):6d}')
    if any(len(data[c]) < 30 for c in CLASSES):
        print('⚠ 일부 클래스 윈도우 부족(<30) — 표본 주의')

    # (1) 5클래스 LDA — action 포함/제거
    acc_full = run(data, use_action=True,  tag='action포함(12D)')
    acc_noa  = run(data, use_action=False, tag='action제거(8D)')

    # (2) 쌍별 마할라노비스 d' + Bayes error 하한 — NIS-only(8D)로 계산.
    #     이유: action은 고정정책에서 클래스내 상수(hover=1/track=0) → 공분산 특이화로
    #     hover포함 쌍 d'가 발산(수치 아티팩트). 실질 aliasing은 NIS 채널이 결정하므로
    #     NIS-only가 진짜 분리도. (action은 hover를 자명 분리 — 아래 (3) 참조.)
    keep = [i for i in range(12) if i % 3 != 2]
    dN = {k: v[:, keep] for k, v in data.items()}
    print(f'\n{"="*66}\n[쌍별 다변량 d\' / Bayes error 하한] (NIS-only 8D, action제외)')
    print(f'{"쌍":<20}{"d′":>8}{"Bayes_err":>12}')
    key_pairs = [('급기동','공격'), ('외란','공격'), ('정상비행','공격'),
                 ('호버전환','공격'), ('급기동','외란'), ('정상비행','급기동'),
                 ('호버전환','급기동'), ('정상비행','외란')]
    for a, b in key_pairs:
        dp = maha_dprime(dN[a], dN[b])
        star = '  ★핵심(급기동/외란 vs 공격)' if (a, b) in [('급기동','공격'), ('외란','공격')] else ''
        print(f'{a+" vs "+b:<20}{dp:8.2f}{gauss_bayes_err(dp):12.3f}{star}')

    # (3) action 채널 진단: 클래스별 평균 action (고정정책 아티팩트 확인)
    print(f'\n{"="*66}\n[action 채널] 클래스별 평균 action (윈도우 내 4스텝 평균)')
    for c in CLASSES:
        acts = data[c][:, [2,5,8,11]].mean()
        print(f'  {c:<6} mean_action = {acts:.3f}')
    print('  → track기반(급기동/공격/외란)은 action≈0으로 동일 → action이 이 셋을 못 가름(NIS가 갈라야).')

    print(f'\n{"="*66}\n[요약] action포함 acc={acc_full:.3f} / 제거 acc={acc_noa:.3f} '
          f'(차이={acc_full-acc_noa:+.3f}=action기여, 주로 hover분리)')


if __name__ == '__main__':
    main()
