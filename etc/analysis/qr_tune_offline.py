#!/usr/bin/env python3
"""qr_tune_offline.py — zu_log(z,u) 를 재구동해 Q·R 격자를 오프라인 튜닝 (2026-08-13)

원리
  UKF 는 순수 관측자다. 한 번 캡처한 (z, u) 시퀀스를 저장해두면, Q·R 만 바꿔가며
  **필터를 오프라인으로 다시 돌려** NIS 를 재계산할 수 있다. 재비행 없이 수십 조합을 훑는다.

무엇을 재나 (사용자 튜닝 원칙)
  필터에 "고집"을 준다 = 공격받은 센서를 흡수하지 않게 K 를 낮춘다(R↑·Q↓).
  단 기동·바람에서도 추정이 발산하면 안 되고, 공격이 끝나면 복귀해야 한다.
  → 세 지표를 동시에 본다:
    ① d′(공격 vs 평시)   높을수록 좋다 — 탐지 분리도
    ② persistence        공격 지속 중 NIS 가 유지되나 (흡수 안 되나)
                         = 공격 후반부 NIS / 공격 직후 NIS.  ≈1 이면 안 흡수, ≪1 이면 흡수됨
    ③ benign_floor       평시(기동 포함) NIS 바닥.  낮아야 오탐이 적다

zu_log 열: [ep, was_init, atk_on, action, z(9), u(4), euler(3), scale, delay]  = 22
  z = [GPS pos NED(3), GPS vel NED(3), gyro(3)],  u = to_physical_u 출력(4)

사용
  ~/isaacsim/python.sh qr_tune_offline.py <zu_log.npz> [-o outdir]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration  # noqa: E402

# ── Q·R 격자 (gyro·vel 채널) ─────────────────────────────────────────
#   roll 공격은 토크→gyro 채널에 실린다. gyro 를 주 격자로, vel 은 보조 확인.
R_GYRO = [0.2, 0.5, 1.0, 2.0]
Q_GYRO = [5e-3, 1e-3, 5e-4, 1e-4]
# vel 은 현행 고정(R_vel 0.3, Q_vel 5e-3) — roll 공격에 vel 기여가 작아 gyro 우선 튜닝.


def rebuild_ukf(calib, r_gyro, q_gyro):
    u = DynamicsUKF(dt=0.02, calib=calib)
    # gyro 블록만 격자값으로 덮어쓴다(vel/pos 는 현행 유지)
    for i in (6, 7, 8):
        u.R[i, i] = r_gyro
        u.Q[i + 3, i + 3] = q_gyro     # 상태벡터에서 gyro 는 9:12, Q 도 같은 인덱스
    return u


def replay(Z, U, FRESH, EP, ATK, calib, r_gyro, q_gyro):
    """전체 zu 를 에피소드별로 UKF 재구동. gyro NIS(raw)를 구간별로 분리 반환.

    ★ step 공격의 잔차는 **펄스**다(t=1~2 스파이크 → PX4 보상으로 침묵 → 후기 재상승).
      그래서 공격 구간 전체 평균은 신호를 희석한다. 구간을 나눠 본다:
        onset    = 공격 시작 +1~5 fresh스텝  (펄스 — RL 윈도우가 잡는 것)
        sustain  = 공격 +6~30 fresh스텝      (persistence — 흡수 여부)
        benign   = 공격 전 전체
      또 에피소드별 **공격 구간 peak** 도 모은다(윈도우 탐지 대리지표)."""
    benign, onset, sustain, atk_all, peaks = [], [], [], [], []
    pers = {}
    for a, b in EP:                              # EP = 에피소드 (start,end) 구간 리스트
        z, u, fr, atk = Z[a:b], U[a:b], FRESH[a:b], ATK[a:b]
        if z.shape[0] < 20:
            continue
        f = rebuild_ukf(calib, r_gyro, q_gyro)
        f.x[0:3] = z[0, 0:3]; f.x[6:9] = z[0, 3:6]; f.x[9:12] = z[0, 6:9]
        atk_start = None
        ep_atk_nis = []
        for k in range(z.shape[0]):
            res, Pzz = f.step(z[k], u[k], gps_fresh=bool(fr[k]))
            g, _ = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)
            if not fr[k]:
                continue                       # 정책이 보는 fresh 스텝만
            if atk[k] > 0.5:
                atk_all.append(g); ep_atk_nis.append(g)
                if atk_start is None:
                    atk_start = k
                d = sum(1 for j in range(atk_start, k+1) if fr[j])   # fresh 기준 경과
                if 1 <= d <= 5:
                    onset.append(g)
                elif 6 <= d <= 30:
                    sustain.append(g)
                if d <= 30:
                    pers.setdefault(d, []).append(g)
            else:
                benign.append(g)
        if ep_atk_nis:
            peaks.append(np.percentile(ep_atk_nis, 95))
    return (np.array(benign), np.array(onset), np.array(sustain),
            np.array(atk_all), np.array(peaks), pers)


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return 0.0
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zu')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.dirname(a.zu) or '.'

    d = np.load(a.zu)
    arr = d['data'] if 'data' in d else d[d.files[0]]
    RESET, ATK = arr[:, 1], arr[:, 2]
    Z, U = arr[:, 4:13], arr[:, 13:17]
    # fresh: GPS(z[0:6]) 가 바뀐 스텝. zu_log 는 50Hz 라 GPS 10Hz → 5스텝마다 갱신.
    dz = np.abs(np.diff(Z[:, 0:6], axis=0)).sum(axis=1)
    FRESH = np.concatenate([[True], dz > 1e-9])
    calib = load_calibration()

    # ★ 에피소드 세그먼트는 reset(was_init) 플래그로 나눈다.
    #   sweep 은 episode 컬럼을 안 올려 전부 0 → 안 나누면 셀 경계(드론 원점 리셋)에서
    #   z 가 순간이동해 거대한 가짜 innovation 이 생긴다(2026-08-13 실측 버그).
    starts = list(np.where(RESET > 0.5)[0])
    if not starts or starts[0] != 0:
        starts = [0] + starts
    EP = [(starts[i], starts[i+1] if i+1 < len(starts) else arr.shape[0])
          for i in range(len(starts))]

    n_atk = int((ATK > 0.5).sum()); n_ben = int((ATK <= 0.5).sum())
    print(f"zu_log: {arr.shape[0]} 스텝 · {len(EP)} 에피소드(reset기준) · "
          f"공격 {n_atk} / 평시 {n_ben} 스텝  fresh {int(FRESH.sum())}")
    if n_atk < 50 or n_ben < 50:
        print("⚠ 공격 또는 평시 표본이 부족하다 — 캡처를 확인할 것")

    # 평시 오탐 바닥 제약: p95 가 이 값을 넘으면 후보에서 뺀다(고집이 과해 오탐↑).
    FLOOR_MAX = 2.0
    results = []
    print(f"\n{'R_gy':>5s} {'Q_gy':>7s} | {'평시p50':>7s} {'평시p95':>7s} | "
          f"{'d′온셋':>7s} {'d′peak':>7s} {'d′평균':>7s} | {'지속비':>6s} {'채택':>4s}")
    print('-' * 74)
    grid = list(itertools.product(R_GYRO, Q_GYRO))
    for r_g, q_g in grid:
        ben, ons, sus, atk, peaks, pers = replay(Z, U, FRESH, EP, ATK, calib, r_g, q_g)
        d_onset = dprime(ons, ben)
        d_peak = dprime(peaks, ben) if peaks.size >= 3 else 0.0
        d_mean = dprime(atk, ben)
        # persistence: sustain 평균 / onset 평균 (≈1 이면 흡수 안 됨, ≪1 이면 흡수)
        persist = (sus.mean() / ons.mean()) if (ons.size and ons.mean() > 1e-6 and sus.size) else np.nan
        b_p95 = float(np.percentile(ben, 95))
        ok = b_p95 <= FLOOR_MAX
        row = dict(r_gyro=r_g, q_gyro=q_g, benign_p50=float(np.median(ben)), benign_p95=b_p95,
                   d_onset=float(d_onset), d_peak=float(d_peak), d_mean=float(d_mean),
                   persist=float(persist) if np.isfinite(persist) else None,
                   floor_ok=bool(ok), n_benign=int(ben.size), n_onset=int(ons.size), n_attack=int(atk.size))
        results.append(row)
        print(f"{r_g:5.2f} {q_g:7.0e} | {np.median(ben):7.2f} {b_p95:7.2f} | "
              f"{d_onset:7.2f} {d_peak:7.2f} {d_mean:7.2f} | "
              f"{persist if np.isfinite(persist) else float('nan'):6.2f} {'✓' if ok else '✗':>4s}")

    # 선정: 평시 바닥 제약(floor_ok) 통과분 중 d′onset(펄스 탐지력) 최대, 동률이면 peak
    cands = [r for r in results if r['floor_ok']] or results
    cands.sort(key=lambda r: (r['d_onset'], r['d_peak']), reverse=True)
    best = cands[0]
    cur = next((r for r in results if r['r_gyro'] == 0.5 and abs(r['q_gyro']-5e-4) < 1e-9), None)
    print(f"\n★ 최적(바닥 p95≤{FLOOR_MAX} 제약): R_gyro={best['r_gyro']} Q_gyro={best['q_gyro']:.0e}")
    print(f"   d′온셋 {best['d_onset']:.2f} · d′peak {best['d_peak']:.2f} · 평시 p50 {best['benign_p50']:.2f} · 지속 {best['persist']}")
    if cur:
        print(f"   현행(R0.5 Q5e-4): d′온셋 {cur['d_onset']:.2f} · d′peak {cur['d_peak']:.2f} · 평시 p50 {cur['benign_p50']:.2f}")
    print("\n※ d′는 raw NIS 기준. step 공격은 펄스라 d′평균<d′온셋 이 정상(PX4 보상 침묵).")
    print("  지속비≈1 = 공격 중 흡수 안 됨(좋음). ≪1 = 필터가 공격을 흡수(고집 부족).")

    with open(os.path.join(out, 'qr_tune_result.json'), 'w') as f:
        json.dump({'grid': results, 'best': best}, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(out, 'qr_tune_result.json')}")
    return results


if __name__ == '__main__':
    main()
