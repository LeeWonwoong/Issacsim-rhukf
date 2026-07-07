"""
sweep_aggregate.py — bias-sweep 결과 집계 + 결과성 밴드 + CUSUM baseline
====================================================================
Isaac 없이 서버에서 단독 실행 (numpy + csv만 사용).

사용:
    python sweep_aggregate.py [results_dir]   # 기본 ./results

입력 : <dir>/sweep_summary.csv, <dir>/sweep_detail.csv
       (신포맷 컬럼: mode,bias,tq_xy,th_n,... / 구포맷 alpha 도 자동 호환)
출력 : (1) 셀별 생존율 → 결과성 밴드(track추락∧hover생존)
        (2) NIS 정상분포(bias=0) → 탐지 임계 후보
        (3) 공격 NIS 분리도 (vel/gyr, d', >99pct%)
        (4) CUSUM 비학습 baseline 탐지지연/FAR  ← RHUKF-RL이 이겨야 할 숫자
"""
import sys, os, csv
from collections import defaultdict
import numpy as np


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bval(r):
    """신포맷 'bias' 우선, 없으면 구포맷 'alpha'."""
    return float(r.get('bias', r.get('alpha', 0.0)))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else './results'
    summ = load_csv(os.path.join(d, 'sweep_summary.csv'))
    detail = load_csv(os.path.join(d, 'sweep_detail.csv'))

    mode = summ[0].get('mode', '?') if summ else '?'
    unit = 'N' if mode == 'thrust' else 'Nm'
    biases = sorted(set(bval(r) for r in summ))

    # bias값 → 실제 주입 (tq_xy, th_n) 매핑 (있으면)
    inj = {}
    for r in summ:
        b = bval(r)
        if 'tq_xy' in r and 'th_n' in r:
            inj[b] = (float(r['tq_xy']), float(r['th_n']))

    def inj_str(b):
        if b in inj:
            return f"tq={inj[b][0]:.3f}Nm th={inj[b][1]:.3f}N"
        return ""

    print(f"\n[mode={mode}]  sweep 단위 = {unit}")

    # ── 1. 셀별 생존율/추락 ──────────────────────────────────────
    cell_surv = defaultdict(list)
    cell_crashstep = defaultdict(list)
    cell_reason = defaultdict(lambda: defaultdict(int))
    for r in summ:
        b, pol = bval(r), r['policy']
        cell_surv[(b, pol)].append(int(r['survived']))
        if not int(r['survived']):
            cell_crashstep[(b, pol)].append(int(r['crash_step']))
            cell_reason[(b, pol)][r['crash_reason']] += 1

    print("\n=== (1) 생존율 ===")
    print(f"{'bias':>7} | {'track':>6} {'crash(track)':>30} | {'hover':>6}  | inject")
    for b in biases:
        ts = np.mean(cell_surv.get((b, 'track'), [np.nan]))
        hs = np.mean(cell_surv.get((b, 'hover'), [np.nan]))
        tcs = cell_crashstep.get((b, 'track'), [])
        info = f"~{int(np.mean(tcs))}stp {dict(cell_reason[(b,'track')])}" if tcs else ""
        print(f"{b:7.3f} | {ts:6.2f} {info:>30} | {hs:6.2f}  | {inj_str(b)}")

    band = [b for b in biases if b > 0
            and np.mean(cell_surv.get((b, 'track'), [1])) < 0.5
            and np.mean(cell_surv.get((b, 'hover'), [0])) > 0.5]
    print("\n[결과성 밴드] track 추락 ∧ hover 생존:")
    if band:
        lo, hi = min(band), max(band)
        print(f"  bias ∈ [{lo:.3f}, {hi:.3f}] {unit}   ({inj_str(lo)} ~ {inj_str(hi)})")
        print(f"  → 학습 공격강도: 이 밴드 중심({np.mean(band):.3f}{unit}) 근처로 샘플링하면"
              f" action(track/hover)이 생존을 가르는 구간에 결정이 몰림")
    else:
        print("  ⚠ 없음! track이 안 추락(결과성↓)하거나 bias=track≈hover(정책무관). "
              "다른 mode/값범위 또는 더 날카로운 기동 검토.")

    # ── 1b. 탐지 데드라인: 지연 호버(dhover{d}) 생존율 vs d ──────────
    #   실측 탐지지연(≈3스텝)에서 생존해야 "탐지→호버" 프레이밍이 성립.
    dh_pols = sorted({r['policy'] for r in summ if r['policy'].startswith('dhover')},
                     key=lambda p: int(p[6:]))
    if dh_pols:
        delays = [int(p[6:]) for p in dh_pols]
        print("\n=== (1b) 탐지 데드라인 — 지연 호버 생존율 (rows: bias, cols: d) ===")
        hdr = f"{'bias':>7} | " + " ".join(f"d={d:>2}" for d in delays) + \
              f" | {'track':>5} {'hover':>5}"
        print(hdr)
        for b in biases:
            if b == 0.0:
                continue
            row = []
            for p in dh_pols:
                s = cell_surv.get((b, p), [])
                row.append(f"{np.mean(s):4.2f}" if s else "  — ")
            ts = np.mean(cell_surv.get((b, 'track'), [np.nan]))
            hs = np.mean(cell_surv.get((b, 'hover'), [np.nan]))
            print(f"{b:7.3f} | " + " ".join(row) + f" | {ts:5.2f} {hs:5.2f}")
        # 데드라인 = 생존율이 0.5 아래로 처음 떨어지는 d (bias별)
        print("  [판독] 밴드 내 bias에서 d=2~3 생존율이 hover 셀에 근접해야 step 공격 채택 OK.")
        print("         d=2~3에서 이미 붕괴하면 → ramp 부분 복원(예: 0.3s) 또는 bias 상한 하향.")
        for b in biases:
            if b == 0.0:
                continue
            deadline = None
            for p in dh_pols:
                s = cell_surv.get((b, p), [])
                if s and np.mean(s) < 0.5:
                    deadline = int(p[6:]); break
            if deadline is not None:
                print(f"    bias={b:.3f}: 생존율<0.5 최초 d = {deadline} (데드라인 ≈ {deadline-1}스텝 이내 전환 필요)")

    # ── 2. NIS 정상분포 (bias=0, track, attack-window) ───────────
    def nis_pick(pred):
        v, g = [], []
        for r in detail:
            if pred(r):
                v.append(float(r['nis_v_raw'])); g.append(float(r['nis_g_raw']))
        return np.array(v), np.array(g)

    base_v, base_g = nis_pick(lambda r: bval(r) == 0.0
                              and r['policy'] == 'track' and r['attack_active'] == '1')
    if len(base_g) == 0:
        base_v, base_g = nis_pick(lambda r: bval(r) == 0.0 and r['policy'] == 'track')
    thr_v = float(np.percentile(base_v, 99)) if len(base_v) else float('nan')
    thr_g = float(np.percentile(base_g, 99)) if len(base_g) else float('nan')
    print("\n=== (2) NIS 정상분포 (bias=0, track) ===")
    print(f"  vel: mean={base_v.mean():.3f} std={base_v.std():.3f} 99pct={thr_v:.3f}")
    print(f"  gyr: mean={base_g.mean():.3f} std={base_g.std():.3f} 99pct={thr_g:.3f}")
    print("  ※ 이 값이 높으면 정상 기동 자체가 NIS를 튀게 함 → ukf_q_gate_gyro 검토")

    # ── 3. 공격 NIS 분리도 (vel+gyr, track & hover) ──────────────
    # 압축 후 d′: vel=log(x+0.5) 저압축 / gyr=log(x+1)=log1p 유지 (관측 채널별 offset).
    def _logc(x, off):
        return np.log(np.asarray(x, dtype=float) + off)
    blv, blg = _logc(base_v, 0.5), _logc(base_g, 1.0)   # 정상(bias0 track) 압축 후 baseline
    print("\n=== (3) 공격 NIS 분리도 (track / hover, attack-on) ===")
    print("  d′ = (공격평균-정상평균)/정상std,  압축 후(vel:log0.5 / gyr:log1p).  판정: vel d′≥2.0, gyr d′ 훼손 없음")
    print(f"{'bias':>7} {'pol':>6} | {'vel mean':>9} {'gyr mean':>9} "
          f"{'vel d′(l.5)':>11} {'gyr d′(l1p)':>11} {'gyr>99pct%':>10}")
    for b in biases:
        if b == 0:
            continue
        for pol in ['track', 'hover']:
            av, ag = nis_pick(lambda r, b=b, pol=pol: bval(r) == b
                              and r['policy'] == pol and r['attack_active'] == '1')
            if len(ag) == 0:
                continue
            alv, alg = _logc(av, 0.5), _logc(ag, 1.0)
            dpv = (alv.mean() - blv.mean()) / (blv.std() + 1e-9)   # vel d′ (log0.5 압축 후)
            dpg = (alg.mean() - blg.mean()) / (blg.std() + 1e-9)   # gyr d′ (log1p 압축 후)
            frac = float(np.mean(ag > thr_g)) * 100
            print(f"{b:7.3f} {pol:>6} | {av.mean():9.3f} {ag.mean():9.3f} "
                  f"{dpv:11.2f} {dpg:11.2f} {frac:9.1f}%")

    # ── 4. CUSUM 비학습 baseline (gyr, track) ────────────────────
    k = float(base_g.mean() + base_g.std())
    h = float(max(thr_g - k, 0.05) * 5.0)
    traces = defaultdict(list)
    for r in detail:
        if r['policy'] != 'track':
            continue
        traces[(bval(r), int(r['episode']))].append(
            (int(r['step']), float(r['nis_g_raw']), r['attack_active'] == '1'))

    delays, fa_eps, base_eps = [], 0, 0
    for (b, ep), tr in traces.items():
        tr.sort()
        S, onset, alarm = 0.0, None, None
        for st, x, atk in tr:
            if atk and onset is None:
                onset = st
            S = max(0.0, S + (x - k))
            if S > h and alarm is None:
                alarm = st
        if b == 0:
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
    print("  → RHUKF-RL은 (지연 ↓ ∧ FAR ↓)를 동시에 깔아야 RL/필터가 정당화됨.\n")


if __name__ == '__main__':
    main()
