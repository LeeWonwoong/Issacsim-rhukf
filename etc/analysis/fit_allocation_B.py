#!/usr/bin/env python3
"""fit_allocation_B.py — 배분행렬 B 실측 (2026-08-13)

목적
  모터 i 침해(EADR 위협모델)를 wrench 주입으로 재현하려면, 모터→wrench 배분행렬 B 가 필요하다.
      wrench = B · u_motor        (B: 4×4, 열 i = 모터 i 가 만드는 wrench 방향)
  모터 i 공격 δ_wrench = B[:,i] · a_i  →  이 비율이 곧 EADR 식 커플링이다.

방법
  rotor_log 는 PX4 가 요청한 정규화 wrench c[0](cmd_thr_z, cmd_tq_x/y/z) 와
  실제 적용된 로터 각속도 w0~w3 를 함께 기록한다. 비포화 구간에서
      c[0] ≈ B · T_norm,   T_i = k·w_i² (정규화)
  를 최소자승으로 풀어 B 를 회귀한다. 회귀계수가 곧 기하(배분)다.
"""
import glob
import sys

import numpy as np

K = 8.54858e-6   # rotor_constant (run_sim config_multirotor)


def load(path):
    d = np.load(path)
    a, cols = d['data'], str(d['cols']).split(',')
    ix = {c: i for i, c in enumerate(cols)}
    thr_z = a[:, ix['cmd_thr_z']]                       # body-z 추력 (호버 ≈ -0.33)
    tq = a[:, [ix['cmd_tq_x'], ix['cmd_tq_y'], ix['cmd_tq_z']]]
    w = a[:, [ix['w0'], ix['w1'], ix['w2'], ix['w3']]]
    return thr_z, tq, w


def main():
    files = sys.argv[1:] or sorted(glob.glob('results_p0813_base/rotor_log.npz'))
    THR, TQ, W = [], [], []
    for f in files:
        thr, tq, w = load(f)
        THR.append(thr); TQ.append(tq); W.append(w)
    thr_z = np.concatenate(THR); tq = np.concatenate(TQ); w = np.concatenate(W)

    # 비행 중(로터가 실제로 도는) 구간만
    Tm = K * w**2                                       # 모터별 추력 [N]
    fly = (Tm.sum(axis=1) > 0.3 * np.median(Tm.sum(axis=1)[Tm.sum(axis=1) > 0]))
    Tm, thr_z, tq = Tm[fly], thr_z[fly], tq[fly]
    # 정규화 모터 추력 (총추력으로 나눠 배분 비율만) — 절편 포함해 오프셋 흡수
    X = np.column_stack([Tm, np.ones(len(Tm))])        # [T1..T4, 1]
    Y = np.column_stack([thr_z, tq[:, 0], tq[:, 1], tq[:, 2]])   # [Fz, τx, τy, τz]

    # 각 wrench 성분을 모터추력에 회귀 → B[j,:] = j 성분의 모터별 계수
    B = np.zeros((4, 4)); B0 = np.zeros(4); R2 = np.zeros(4)
    names = ['Fz', 'tau_x(roll)', 'tau_y(pitch)', 'tau_z(yaw)']
    for j in range(4):
        coef, *_ = np.linalg.lstsq(X, Y[:, j], rcond=None)
        B[j, :] = coef[:4]; B0[j] = coef[4]
        resid = Y[:, j] - X @ coef
        R2[j] = 1 - (resid**2).sum() / ((Y[:, j] - Y[:, j].mean())**2).sum()

    print(f"표본 {len(Tm)} (비행구간)  ·  k={K:.3e}")
    print(f"\n배분행렬 B  (wrench = B · T_motor[N], 열=모터)")
    print(f"  {'':14s} {'M1':>10s} {'M2':>10s} {'M3':>10s} {'M4':>10s} {'절편':>9s} {'R²':>6s}")
    for j in range(4):
        print(f"  {names[j]:14s} " + ' '.join(f'{B[j,i]:10.4f}' for i in range(4)) +
              f" {B0[j]:9.4f} {R2[j]:6.3f}")

    # ── 모터 i 공격의 wrench 커플링 방향(정규화: |δτ_roll|=1 로 스케일) ──
    print(f"\n모터 i 침해 → wrench 커플링 (EADR 식). 각 열을 roll 성분 크기로 정규화:")
    print(f"  {'모터':>6s} {'δFz':>9s} {'δτx(roll)':>11s} {'δτy(pitch)':>11s} {'δτz(yaw)':>10s}  해석")
    for i in range(4):
        col = B[:, i].copy()
        s = abs(col[1]) if abs(col[1]) > 1e-9 else 1.0
        c = col / s
        thr_frac = col[0] / abs(B[0, :]).sum()          # 이 모터의 추력 기여 비중
        note = 'thrust 지배' if abs(c[0]) > abs(c[1]) else 'torque 지배'
        print(f"  {'M'+str(i+1):>6s} {c[0]:9.3f} {c[1]:11.3f} {c[2]:11.3f} {c[3]:10.3f}  {note}")

    # ── 저장 (attack 구현이 읽을 수 있게) ──
    np.savez('allocation_B.npz', B=B, B0=B0, R2=R2, k=K, cols='Fz,tau_x,tau_y,tau_z')
    print(f"\n저장: allocation_B.npz")
    print(f"\n검산 — 순수 roll(밤샘 비결과성)은 4모터 균형이라 δFz≈0 → 안 떨어짐이 맞다.")
    print(f"  모터 단일 침해는 열마다 δFz≠0 (thrust 성분 포함) → 결과성 생김.")


if __name__ == '__main__':
    main()
