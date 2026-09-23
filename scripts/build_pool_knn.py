#!/usr/bin/env python3
"""scripts/build_pool_knn.py — Isaac 캡처·학습 스텝 기록 → k-NN 재생용 원시 NIS 풀 (surrogate model=knn).

    python3 scripts/build_pool_knn.py <출력.npz> <폴더> [<폴더> ...]
    폴더 = online_rl_main 의 capture/ 또는 steps/ 를 가진 결과 폴더 (여러 개)

행마다 조건 특징(정책이 모르는 참값 포함) + 원시 NIS 를 저장한다:
  d0, d1, d3, d6   이 스텝 NIS 가 반영하는 세기 δ_eff(t) 와 1·3·6 스텝 전 값 (위상·지연 표현)
  since_end        직전 공격 사건이 끝난 뒤 경과 스텝 (공격 중 0, 공격 이력 없음 99)
  act, dwell       직전 실행 행동 a_(t−1) 과 그 행동 유지 스텝(전환 직후 0)
  ws               풍속
  dlast            현재/직전 공격 사건의 최대 세기 (vel 은 지연이 길어 공격이 끝난 뒤에도 사건 크기에 따라 응답이 다르다)
  nis_v, nis_g     원시 NIS (ε = rᵀS⁻¹r/n_z)
  ep               에피소드 번호(풀 안 고유) — 시간 상관 추정·검증 분할용
Isaac 규약: 주입은 스텝 끝 발행 → 스텝 t NIS 는 δ[t−1] 을 반영 = delta_eff 열(캡처 기록).
"""
import glob
import json
import os
import sys

import numpy as np

PATS = ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve']   # ★09-22: 패턴 인덱스(surrogate _feat 와 동일 순서)


def episode_rows(z):
    cols = [str(c) for c in z['cols']]; R = z['rows']
    C = {k: R[:, i] for i, k in enumerate(cols)}
    de = C['delta_eff']; a = C['prev_action'].astype(int); n = len(de); st = C['step'].astype(int)
    # 스텝이 연속이라는 가정 확인(리셋 없이 한 에피소드)
    if n < 8 or np.any(np.diff(st) != 1):
        return None
    lag = lambda k: np.r_[np.zeros(k), de[:-k]] if k else de
    since = np.full(n, 99.0); last = None; dlast = np.zeros(n); cur = 0.0
    for i in range(n):
        if de[i] > 0:
            cur = de[i] if (i == 0 or de[i - 1] == 0) else max(cur, de[i])   # 사건 시작이면 새로, 아니면 최대 갱신
            last = i; since[i] = 0
        elif last is not None:
            since[i] = i - last
        dlast[i] = cur                                   # 현재/직전 사건의 최대 세기 (vel 의 긴 지연 응답 조건)
    dwell = np.zeros(n)
    for i in range(1, n):
        dwell[i] = dwell[i - 1] + 1 if a[i] == a[i - 1] else 0
    ws = np.full(n, float(z['wind_speed']))
    pat = np.full(n, float(PATS.index(str(z['pattern'])) if ('pattern' in z.files and str(z['pattern']) in PATS) else -1))   # ★09-22 v3: 기동 패턴 조건
    return np.c_[de, lag(1), lag(3), lag(6), since, a, dwell, ws, dlast, C['nis_v_raw'], C['nis_g_raw'], pat]


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    blocks, eps, srcs = [], [], []
    k = 0
    for d in dirs:
        for sub in ('capture', 'steps'):
            for f in sorted(glob.glob(os.path.join(d, sub, 'ep*.npz'))):
                X = episode_rows(np.load(f, allow_pickle=False))
                if X is None:
                    continue
                blocks.append(X); eps.append(np.full(len(X), k)); srcs.append(f); k += 1
    X = np.concatenate(blocks); ep = np.concatenate(eps)
    names = ['d0', 'd1', 'd3', 'd6', 'since_end', 'act', 'dwell', 'ws', 'dlast', 'nis_v', 'nis_g', 'pat']
    # 시간 상관(코퓰러 ρ): 조건을 만족하는 연속 두 스텝 쌍에서 log NIS 의 lag-1 상관
    def rho(mask_fn, col):
        xs, ys = [], []
        m = mask_fn(X); same = np.r_[ep[1:] == ep[:-1], False]
        idx = np.flatnonzero(m & np.r_[m[1:], False] & same)
        if len(idx) < 50:
            return 0.5
        lx = np.log(X[:, col] + 1e-6)
        return float(np.corrcoef(lx[idx], lx[idx + 1])[0, 1])
    clean = lambda A: (A[:, 0] == 0) & (A[:, 4] == 99) & (A[:, 5] == 0)
    atk = lambda A: A[:, 0] > 0
    rh = dict(rho_g_cln=rho(clean, 10), rho_v_cln=rho(clean, 9), rho_g_atk=rho(atk, 10), rho_v_atk=rho(atk, 9))
    np.savez(out, format='knn_v3', X=X, names=np.array(names), ep=ep, sources=np.array(srcs), rho=json.dumps(rh))
    import collections; pc = collections.Counter(X[:, 11].astype(int))
    print(f'{out}: 행 {len(X)} · 에피소드 {k} · 공격 행 {(X[:, 0] > 0).sum()} · hover 행 {(X[:, 5] == 1).sum()} · 패턴 행수 {dict(sorted(pc.items()))} · 시간상관 {rh}')


if __name__ == '__main__':
    main()
