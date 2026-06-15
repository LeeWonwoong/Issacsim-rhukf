"""
sweep_aggregate.py — α-sweep 결과 집계 + 결과성 밴드 + CUSUM baseline
====================================================================
Isaac 없이 서버에서 단독 실행 (numpy + csv만 사용).

사용:
    python sweep_aggregate.py [results_dir]   # 기본 ./results

입력 : <dir>/sweep_summary.csv, <dir>/sweep_detail.csv
출력 : (1) 셀별 생존율 → 결과성 밴드
        (2) NIS 정상분포(α=0) → 탐지 임계 후보
        (3) 공격 NIS 분리도 (d', >threshold%)
        (4) CUSUM 비학습 baseline 탐지지연/FAR  ← RHUKF-RL이 이겨야 할 숫자
"""
import sys
import os
import csv
from collections import defaultdict
import numpy as np


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else './results'
    summ = load_csv(os.path.join(d, 'sweep_summary.csv'))
    detail = load_csv(os.path.join(d, 'sweep_detail.csv'))
    alphas = sorted(set(float(r['alpha']) for r in summ))

    # ── 1. 셀별 생존율/추락 ──────────────────────────────────────
    cell_surv = defaultdict(list)
    cell_crashstep = defaultdict(list)
    cell_reason = defaultdict(lambda: defaultdict(int))
    for r in summ:
        a, pol = float(r['alpha']), r['policy']
        cell_surv[(a, pol)].append(int(r['survived']))
        if not int(r['survived']):
            cell_crashstep[(a, pol)].append(int(r['crash_step']))
            cell_reason[(a, pol)][r['crash_reason']] += 1

    print("\n=== (1) 생존율 ===")
    print(f"{'α':>6} | {'track':>6} {'crash(track)':>26} | {'hover':>6}")
    for a in alphas:
        ts = np.mean(cell_surv.get((a, 'track'), [np.nan]))
        hs = np.mean(cell_surv.get((a, 'hover'), [np.nan]))
        tcs = cell_crashstep.get((a, 'track'), [])
        info = f"~{int(np.mean(tcs))}stp {dict(cell_reason[(a,'track')])}" if tcs else ""
        print(f"{a:6.2f} | {ts:6.2f} {info:>26} | {hs:6.2f}")

    band = [a for a in alphas if a > 0
            and np.mean(cell_surv.get((a, 'track'), [1])) < 0.5
            and np.mean(cell_surv.get((a, 'hover'), [0])) > 0.5]
    print("\n[결과성 밴드] track 추락 ∧ hover 생존:")
    if band:
        print(f"  α ∈ [{min(band):.2f}, {max(band):.2f}]  →  "
              f"curriculum_fixed_min≈{min(band):.2f}, curriculum_end_max≈{max(band):.2f}")
        print(f"  eval 강도 후보: {band[:3]}")
    else:
        print("  ⚠ 없음! track이 안 추락(결과성↓)하거나 hover가 못 버팀. "
              "강도범위/패턴/리셋 재검토 (logical_done 재고 신호).")

    # ── 2. NIS 정상분포 (α=0, aggressive, attack-window) ─────────
    def nis_pick(pred):
        v, g = [], []
        for r in detail:
            if pred(r):
                v.append(float(r['nis_v_raw'])); g.append(float(r['nis_g_raw']))
        return np.array(v), np.array(g)

    base_v, base_g = nis_pick(lambda r: float(r['alpha']) == 0.0
                              and r['policy'] == 'track' and r['attack_active'] == '1')
    if len(base_g) == 0:
        base_v, base_g = nis_pick(lambda r: float(r['alpha']) == 0.0 and r['policy'] == 'track')
    thr_v = float(np.percentile(base_v, 99)) if len(base_v) else float('nan')
    thr_g = float(np.percentile(base_g, 99)) if len(base_g) else float('nan')
    print("\n=== (2) NIS 정상분포 (α=0, aggressive) ===")
    print(f"  vel: mean={base_v.mean():.3f} std={base_v.std():.3f} 99pct={thr_v:.3f}")
    print(f"  gyr: mean={base_g.mean():.3f} std={base_g.std():.3f} 99pct={thr_g:.3f}")
    print("  ※ 이 값이 높으면 정상 기동 자체가 NIS를 튀게 함 → ukf_q_gate_gyro 켜기 검토")

    # ── 3. 공격 NIS 분리도 (track, attack-on) ────────────────────
    print("\n=== (3) 공격 NIS 분리도 (gyr 채널, track) ===")
    print(f"{'α':>6} | {'mean':>7} {'95pct':>7} {'d_prime':>8} {'>99pct%':>8}")
    for a in alphas:
        if a == 0:
            continue
        _, ag = nis_pick(lambda r, a=a: float(r['alpha']) == a
                         and r['policy'] == 'track' and r['attack_active'] == '1')
        if len(ag) == 0:
            continue
        dprime = (ag.mean() - base_g.mean()) / (base_g.std() + 1e-9)
        frac = float(np.mean(ag > thr_g)) * 100
        print(f"{a:6.2f} | {ag.mean():7.3f} {np.percentile(ag,95):7.3f} {dprime:8.2f} {frac:7.1f}%")

    # ── 4. CUSUM 비학습 baseline (gyr) ───────────────────────────
    # S_t = max(0, S_{t-1} + (x - k)),  alarm when S > h.
    k = float(base_g.mean() + base_g.std())          # drift = 정상평균 + 1σ
    h = float(max(thr_g - k, 0.05) * 5.0)            # threshold(러프; k/h는 튜닝 여지)
    traces = defaultdict(list)
    for r in detail:
        if r['policy'] != 'track':
            continue
        traces[(float(r['alpha']), int(r['episode']))].append(
            (int(r['step']), float(r['nis_g_raw']), r['attack_active'] == '1'))

    delays, fa_eps, base_eps = [], 0, 0
    for (a, ep), tr in traces.items():
        tr.sort()
        S, onset, alarm = 0.0, None, None
        for st, x, atk in tr:
            if atk and onset is None:
                onset = st
            S = max(0.0, S + (x - k))
            if S > h and alarm is None:
                alarm = st
        if a == 0:
            base_eps += 1
            if alarm is not None:
                fa_eps += 1
        elif onset is not None and alarm is not None and alarm >= onset:
            delays.append(alarm - onset)

    far = fa_eps / max(base_eps, 1)
    md = float(np.mean(delays)) if delays else float('nan')
    print("\n=== (4) CUSUM 비학습 baseline (RHUKF-RL이 이겨야 할 숫자) ===")
    print(f"  k={k:.3f}  h={h:.3f}")
    print(f"  평균 탐지지연 = {md:.1f} step | 검출 에피소드 {len(delays)} | "
          f"FAR(무공격 오경보) = {far:.2f}")
    print("  → RHUKF-RL은 이 (지연 ↓ ∧ FAR ↓)를 동시에 깔아야 RL/필터가 정당화됨.\n")


if __name__ == '__main__':
    main()
