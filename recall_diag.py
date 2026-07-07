#!/usr/bin/env python3
"""
recall_diag.py — recall 미탐의 원인 분해: 강도(s) vs 지연(경과스텝)
=====================================================================
새 zu_log(atk_scale, atk_delay 포함, 22컬럼)을 UKF로 재생해 NIS를 만들고,
'탐지'(NIS>평시p90) 기준으로:
  [A] 공격강도 s별 탐지율 → "어느 강도부터 잡히나"(약한공격이 범인?)
  [B] 공격경과스텝별 탐지율 → "공격 시작 후 몇 스텝부터 잡히나"(delay가 범인?)
  [C] (강도×지연) 히트맵 → 둘의 상호작용
  [D] burst 길이 대비 누적 탐지 → "짧은 burst를 놓치나"

판정: [A]가 완만하면 강도 문제(약한공격 못잡음) / [B]가 가파르면 지연 문제(초반 놓침).

사용: python3 recall_diag.py results_zu_s12/zu_log.npz
"""
import argparse, sys
import numpy as np
try:
    from ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
except Exception:
    from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled


def replay_nis(data, dt, calib):
    rst = data[:, 1]; z = data[:, 4:13]; u = data[:, 13:17]; eul = data[:, 17:20]
    nv = np.empty(len(data)); ng = np.empty(len(data)); ukf = None
    for i in range(len(data)):
        if rst[i] > 0.5 or ukf is None:
            ukf = DynamicsUKF(dt=dt, calib=calib)
            ukf.x[0:3] = z[i, 0:3]; ukf.x[3:6] = eul[i]
            ukf.x[6:9] = z[i, 3:6]; ukf.x[9:12] = z[i, 6:9]
        res, Pzz = ukf.step(z[i], u[i])
        _, a = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3.0, offset=0.5)  # vel 저압축(log0.5) — 학습 관측과 일치
        _, b = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)              # gyro log1p 유지
        nv[i] = a; ng[i] = b
    return nv, ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zu'); ap.add_argument('--calib', default='calibration.json')
    ap.add_argument('--combine', choices=['v', 'max', 'g'], default='max',
                    help="탐지신호: v=vel만, g=gyro만, max=max(vel,gyro)")
    args = ap.parse_args()

    npz = np.load(args.zu, allow_pickle=True); data = npz['data']
    dt = float(npz['dt']) if 'dt' in npz else 0.02
    if data.shape[1] < 22:
        sys.exit(f"[!] 컬럼 {data.shape[1]}개 — atk_scale/atk_delay 없음. "
                 f"갱신된 online_rl_main으로 --log-zu 새로 수집하세요(22컬럼 기대).")
    calib = load_calibration(args.calib)

    atk = data[:, 2].astype(int)
    scale = data[:, 20]      # 공격강도 s (공격중에만 >0)
    delay = data[:, 21]      # 공격경과스텝 (공격중 0,1,2..., 평시 -1)
    ep = data[:, 0]

    print(f"로드 {len(data)}스텝 | 평시 {int((atk==0).sum())} 공격 {int((atk==1).sum())}")
    print("UKF 재생 중...")
    nv, ng = replay_nis(data, dt, calib)
    sig = {'v': nv, 'g': ng, 'max': np.maximum(nv, ng)}[args.combine]
    thr = float(np.nanpercentile(sig[atk == 0], 90))      # 평시 p90 = 탐지 임계
    det = sig > thr                                        # '탐지' 여부
    print(f"탐지신호={args.combine}, 임계(평시p90)={thr:.3f}")

    am = atk == 1
    print(f"\n전체 공격 탐지율 = {100*np.mean(det[am]):.0f}%  (단순임계 recall 상한)")

    # ── [A] 강도(s)별 탐지율 ──
    print("\n"+"="*70)
    print("[A] 공격강도 s별 탐지율  (밴드 s∈[1.2,1.3]; 낮을수록 약한공격)")
    print("="*70)
    s_atk = scale[am]; det_a = det[am]
    edges = np.arange(np.floor(s_atk.min()*20)/20, s_atk.max()+0.05, 0.05)  # 0.05 간격
    if len(edges) < 2: edges = np.linspace(s_atk.min(), s_atk.max()+1e-6, 6)
    print("    s 구간      |  n     탐지율")
    for k in range(len(edges)-1):
        lo, hi = edges[k], edges[k+1]
        seg = (s_atk >= lo) & (s_atk < hi)
        if seg.sum() > 0:
            print(f"   [{lo:.2f},{hi:.2f})  | {seg.sum():5d}  {100*np.mean(det_a[seg]):4.0f}%")
    # 상관
    if len(np.unique(s_atk)) > 3:
        c = np.corrcoef(s_atk, det_a.astype(float))[0, 1]
        print(f"   강도↔탐지 상관 = {c:+.2f}  (높을수록 '강도가 탐지를 좌우'=약한공격이 범인)")

    # ── [B] 경과스텝별 탐지율 ──
    print("\n"+"="*70)
    print("[B] 공격 경과스텝별 탐지율  (0=공격 시작 직후)")
    print("="*70)
    d_atk = delay[am].astype(int); det_b = det[am]
    print("   경과스텝 |  n     탐지율   (이게 가파르게 오르면 '지연'이 범인)")
    for d in range(0, min(int(d_atk.max())+1, 20)):
        seg = d_atk == d
        if seg.sum() > 0:
            print(f"   t={d:2d}     | {seg.sum():5d}  {100*np.mean(det_b[seg]):4.0f}%")
    # 첫 탐지까지 평균 스텝
    print()
    # burst별 첫 탐지 지연
    first_det = []
    miss_bursts = 0; tot_bursts = 0
    cur = None
    for i in range(len(data)):
        if am[i]:
            if cur is None or delay[i] == 0:
                if cur is not None:
                    tot_bursts += 1
                    if cur['det'] is None: miss_bursts += 1
                    else: first_det.append(cur['det'])
                cur = {'det': None}
            if cur['det'] is None and det[i]:
                cur['det'] = int(delay[i])
    if cur is not None:
        tot_bursts += 1
        if cur['det'] is None: miss_bursts += 1
        else: first_det.append(cur['det'])
    if first_det:
        print(f"   burst {tot_bursts}개 중: 탐지성공 {len(first_det)}개(첫탐지 평균 {np.mean(first_det):.1f}스텝, "
              f"중앙 {np.median(first_det):.0f}), 완전미탐 {miss_bursts}개({100*miss_bursts/max(tot_bursts,1):.0f}%)")

    # ── [C] 강도×지연 교차 (간단) ──
    print("\n"+"="*70)
    print("[C] 강도×경과 교차 탐지율  (행=강도, 열=경과스텝구간)")
    print("="*70)
    s_bins = [(s_atk.min(), 1.2), (1.2, 1.25), (1.25, s_atk.max()+1e-6)]
    d_bins = [(0, 3), (3, 7), (7, 15), (15, 999)]
    hdr = "   강도\\경과 | " + " ".join([f"{a}-{b if b<999 else '+'}".rjust(7) for a, b in d_bins])
    print(hdr)
    for slo, shi in s_bins:
        row = f"   [{slo:.2f},{shi:.2f}) |"
        for dlo, dhi in d_bins:
            seg = (s_atk >= slo) & (s_atk < shi) & (d_atk >= dlo) & (d_atk < dhi)
            row += f" {(100*np.mean(det_a[seg]) if seg.sum()>5 else float('nan')):6.0f}%" if seg.sum() > 5 else "    -  "
        print(row)

    # ── 판정 ──
    print("\n"+"="*70); print("판정"); print("="*70)
    # 강도 기울기: 약한 구간 vs 강한 구간 탐지율 차
    weak = det_a[s_atk < 1.2]; strong = det_a[s_atk >= 1.25]
    wr = 100*np.mean(weak) if len(weak) > 5 else float('nan')
    sr = 100*np.mean(strong) if len(strong) > 5 else float('nan')
    # 지연 기울기: 초반 vs 후반
    early = det_b[d_atk < 3]; late = det_b[d_atk >= 7]
    er = 100*np.mean(early) if len(early) > 5 else float('nan')
    lr = 100*np.mean(late) if len(late) > 5 else float('nan')
    print(f"  강도: 약한(s<1.2) {wr:.0f}% vs 강한(s≥1.25) {sr:.0f}%  → 차이 크면 '약한공격이 범인'")
    print(f"  지연: 초반(t<3) {er:.0f}% vs 후반(t≥7) {lr:.0f}%      → 차이 크면 '탐지지연이 범인'")
    print(f"\n  처방: 강도 문제 → bias_scale 하한↑(약한공격 제외). "
          f"지연 문제 → fn_per_step↑(빨리 탐지 유도).")

if __name__ == '__main__':
    main()
