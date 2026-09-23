#!/usr/bin/env python3
"""scripts/check_pbrs_run.py — P2′ 성형(reward.shape_tilt>0) Isaac 스모크 런 판독 (★09-24)

    python3 scripts/check_pbrs_run.py <런폴더> [--gamma 0.9]   (γ 생략 시 config.yaml 의 reward.shape_gamma / agent.gamma)

합격 기준(전부 만족):
  ① steps 열 끝에 reward_train·theta_eff·phi 가 있고, 에피마다 F = reward_train − reward 가
     F[0] = 0, F[t] = γ·φ[t] − φ[t−1] (|오차| < 1e-6) — 성형 항등식이 실제 로그에서 성립
  ② 동결 규칙: prev_action 전환 행부터 shape_freeze(기본 3) 행 밖에서는 theta_eff = √(roll²+pitch²)
  ③ 추락(terminal) 행은 φ = 0
  ④ metrics: reward_train − reward = shape_F (반올림 0.002 안)
  ⑤ θ 크기: 평시 track 평균 0.08–0.20 rad (Isaac v2clean 0.121), 공격 행 평균 > 평시 track
  참고 출력: |F| 중앙·p99, 에피당 Σr^G 와 Σ학습보상.
"""
import argparse
import csv
import glob
import os

import numpy as np


def _gamma(run, g):
    if g is not None:
        return g
    try:
        import yaml
        with open(os.path.join(run, 'config.yaml')) as f:
            c = yaml.safe_load(f)
        r = c.get('reward', {}) or {}
        return float(r.get('shape_gamma') or c.get('agent', {}).get('gamma'))
    except Exception as e:
        raise SystemExit(f'γ 를 못 읽었다({e}) — --gamma 로 주시오')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('--gamma', type=float, default=None); ap.add_argument('--freeze', type=int, default=3)
    a = ap.parse_args()
    g = _gamma(a.run, a.gamma)
    files = sorted(glob.glob(os.path.join(a.run, 'steps', 'ep*.npz')))
    if not files:
        raise SystemExit('steps 없음 (log.steps=true 로 돌릴 것)')
    ok = dict(cols=True, ident=True, freeze=True, term=True)
    Fs = []; th_ct = []; th_at = []; worst = 0.0
    for f in files:
        z = np.load(f, allow_pickle=True); cols = [str(c) for c in z['cols']]; R = z['rows']; c = {n: i for i, n in enumerate(cols)}
        if not {'reward_train', 'theta_eff', 'phi'} <= set(cols):
            ok['cols'] = False; continue
        if np.any(np.diff(R[:, c['step']]) < 0):          # HARD 리셋 앞붙음 파일은 항등식 대상 아님
            continue
        F = R[:, c['reward_train']] - R[:, c['reward']]; phi = R[:, c['phi']]
        err = np.r_[abs(F[0]), np.abs(F[1:] - (g * phi[1:] - phi[:-1]))] if len(F) > 1 else np.abs(F)
        worst = max(worst, float(err.max())); ok['ident'] &= bool(err.max() < 1e-6)
        pa = R[:, c['prev_action']]; th = np.hypot(R[:, c['roll']], R[:, c['pitch']])
        free = np.ones(len(R), bool)
        for s in np.flatnonzero(np.r_[False, pa[1:] != pa[:-1]]):
            free[s:s + a.freeze] = False
        ok['freeze'] &= bool(np.allclose(R[free, c['theta_eff']], th[free], atol=1e-9))
        if str(z['reason']).startswith('crash') and R[-1, c['done']] > 0:
            ok['term'] &= bool(phi[-1] == 0.0)
        Fs.extend(F[1:].tolist())
        atk = R[:, c['atk_flag']] > 0.5
        th_ct.extend(th[~atk & (pa == 0)].tolist()); th_at.extend(th[atk].tolist())
    met_ok = True; eps = []
    for mf in glob.glob(os.path.join(a.run, 'metrics_*.csv')):
        with open(mf) as fh:
            for row in csv.DictReader(fh):
                if 'reward_train' not in row:
                    met_ok = False; continue
                d = float(row['reward_train']) - float(row['reward']) - float(row['shape_F'])
                met_ok &= abs(d) < 2e-3; eps.append((float(row['reward']), float(row['reward_train'])))
    ct = float(np.mean(th_ct)) if th_ct else float('nan'); at = float(np.mean(th_at)) if th_at else float('nan')
    th_ok = 0.08 <= ct <= 0.20 and (not th_at or at > ct)
    Fa = np.abs(np.array(Fs)) if Fs else np.zeros(1)
    print(f'[P2′ 판독] {a.run} · γ={g} · 에피 {len(files)}')
    print(f'  ① 열·항등식 {"OK" if ok["cols"] and ok["ident"] else "FAIL"} (최대 오차 {worst:.2e})  ② 동결 {"OK" if ok["freeze"] else "FAIL"}  '
          f'③ terminal φ=0 {"OK" if ok["term"] else "FAIL"}  ④ metrics {"OK" if met_ok and eps else "FAIL"}  '
          f'⑤ θ 평시track {ct:.3f} · 공격 {at:.3f} {"OK" if th_ok else "FAIL"}')
    print(f'  |F| 중앙 {np.median(Fa):.3f} · p99 {np.percentile(Fa, 99):.3f} · 에피 Σr^G 평균 {np.mean([e[0] for e in eps]) if eps else float("nan"):.1f} '
          f'· Σ학습보상 평균 {np.mean([e[1] for e in eps]) if eps else float("nan"):.1f}')
    allok = ok['cols'] and ok['ident'] and ok['freeze'] and ok['term'] and met_ok and bool(eps) and th_ok
    print('판정:', '합격' if allok else '불합격')
    raise SystemExit(0 if allok else 1)


if __name__ == '__main__':
    main()
