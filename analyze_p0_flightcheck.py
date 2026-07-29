"""analyze_p0_flightcheck.py — Phase 0 비행 검증 판독 (2026-07-29)

사용: ~/isaacsim/python.sh analyze_p0_flightcheck.py results_p0_flightcheck

판독 항목
  1) 무공격(bias=0) 생존율 — 모터 지연 주입 후 비행이 안정한가
  2) max_roll/max_pitch 분포 vs 자세 클립 1.05 rad(60.2°) — 클립 도달 빈도
  3) 호버 전환 요 슬루 — 구 프레임 버그의 인공물(중앙 84.5°)이 사라졌는가
  4) circle 패턴 요 추종 — yaw wrap 수정 후 정상인가
  5) 평시 NIS 기준선 (P1 사전 확인용)
"""
import os, sys
import numpy as np

RAD = np.pi / 180.0
outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_p0_flightcheck'


def load_csv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


print("=" * 78)
print(f"Phase 0 비행 검증 판독 — {outdir}")
print("=" * 78)

# ── 1) 생존율 ─────────────────────────────────────────────
sm = load_csv(os.path.join(outdir, 'sweep_summary.csv'))
print(f"\n[1] 무공격(bias=0) 생존율   — 총 {len(sm)} 에피소드")
pats = sorted(set(r['pattern'] for r in sm))
print(f"  {'패턴':>11s} {'ep':>4s} {'생존율':>8s} {'추락사유':>28s}")
all_ok = True
for p in pats:
    rs = [r for r in sm if r['pattern'] == p]
    surv = np.mean([int(r['survived']) for r in rs])
    reasons = {}
    for r in rs:
        if not int(r['survived']):
            reasons[r['crash_reason']] = reasons.get(r['crash_reason'], 0) + 1
    rstr = ', '.join(f'{k}×{v}' for k, v in reasons.items()) or '—'
    flag = '' if surv == 1.0 else '  ← ★'
    if surv != 1.0:
        all_ok = False
    print(f"  {p:>11s} {len(rs):4d} {surv:8.2f} {rstr:>28s}{flag}")
print(f"  판정: {'PASS — 무공격 전 패턴 생존' if all_ok else '★ FAIL — 무공격인데 추락한 셀이 있다'}")

# ── 2) 자세 클립 도달 빈도 ────────────────────────────────
print(f"\n[2] max_roll / max_pitch 분포 vs 자세 클립 1.05 rad ({1.05/RAD:.1f}°)")
mr = np.array([float(r['max_roll']) for r in sm])
mp = np.array([float(r['max_pitch']) for r in sm])
for nm, v in (('max_roll', mr), ('max_pitch', mp)):
    print(f"  {nm:9s} 중앙 {np.median(v)/RAD:6.2f}°  90pct {np.percentile(v,90)/RAD:6.2f}°  "
          f"최대 {v.max()/RAD:6.2f}°   >0.8rad(구클립): {np.mean(v>0.8)*100:5.1f}%  "
          f">1.05rad(신클립): {np.mean(v>1.05)*100:5.1f}%")
print("  → 구 클립(0.8)을 넘는 비율이 유의하면 상향(1.05)이 실제로 필요했다는 뜻.")
print("    신 클립(1.05)을 넘는 비율은 0 에 가까워야 한다(넘으면 모델이 여전히 포화).")

# ── 3~4) zu_log 기반 요 분석 ─────────────────────────────
zp = os.path.join(outdir, 'zu_log.npz')
if not os.path.exists(zp):
    print(f"\n[3-4] zu_log.npz 없음 — 요 분석 생략")
else:
    d = np.load(zp, allow_pickle=True)
    A = d['data']
    cols = [s.strip() for s in str(d['cols']).replace(',', ' ').split()]
    c = {n: i for i, n in enumerate(cols)}
    psi = A[:, c['euler_psi']]
    act = A[:, c['action']]
    ep = A[:, c['episode']]

    print(f"\n[3] 호버 전환 요 슬루 (구 프레임 버그: 전환 후 3s |Δψ| 중앙 84.5°, 90pct 90.0°)")
    tr = np.where((act[1:] == 1) & (act[:-1] == 0) & (ep[1:] == ep[:-1]))[0] + 1
    tr = [i for i in tr if i >= 150 and i + 150 < len(psi)]
    if not tr:
        print("  전환 샘플 없음")
    else:
        after = np.array([np.abs(np.unwrap(psi[i:i+150] - psi[i])).max() for i in tr])
        before = np.array([np.abs(np.unwrap(psi[i-150:i] - psi[i-150])).max() for i in tr])
        print(f"  전환 {len(tr)}회")
        print(f"    전환 후 3s |Δψ|: 중앙 {np.median(after)/RAD:6.2f}°  "
              f"90pct {np.percentile(after,90)/RAD:6.2f}°  최대 {after.max()/RAD:6.2f}°")
        print(f"    전환 전 3s |Δψ|: 중앙 {np.median(before)/RAD:6.2f}°  "
              f"90pct {np.percentile(before,90)/RAD:6.2f}°")
        verdict = 'PASS — 요 슬루 인공물 소멸' if np.median(after) < 20*RAD else '★ 여전히 큼'
        print(f"  판정: {verdict}  (구 84.5° → 현 {np.median(after)/RAD:.2f}°)")

    print(f"\n[4] psi 범위 (wrap 정상 여부)")
    print(f"  psi ∈ [{psi.min():+.4f}, {psi.max():+.4f}] rad "
          f"= [{psi.min()/RAD:+.1f}°, {psi.max()/RAD:+.1f}°]   "
          f"[-pi,pi] 안: {np.abs(psi).max() <= np.pi + 1e-9}")

# ── 5) 평시 NIS 기준선 ────────────────────────────────────
det = load_csv(os.path.join(outdir, 'sweep_detail.csv'))
print(f"\n[5] 평시 NIS 기준선 (bias=0, P1 사전 확인) — {len(det)} 스텝")
print(f"  {'패턴':>11s} {'nis_v_raw':>26s} {'nis_g_raw':>26s} {'v_scaled':>10s} {'g_scaled':>10s}")
print(f"  {'':>11s} {'중앙 / 90pct / 99pct':>26s} {'중앙 / 90pct / 99pct':>26s}")
for p in pats:
    rs = [r for r in det if r['pattern'] == p]
    if not rs:
        continue
    vr = np.array([float(r['nis_v_raw']) for r in rs])
    gr = np.array([float(r['nis_g_raw']) for r in rs])
    vs = np.array([float(r['nis_v_scaled']) for r in rs])
    gs = np.array([float(r['nis_g_scaled']) for r in rs])
    print(f"  {p:>11s} {np.median(vr):7.3f}/{np.percentile(vr,90):7.3f}/{np.percentile(vr,99):8.3f} "
          f"{np.median(gr):7.3f}/{np.percentile(gr,90):7.3f}/{np.percentile(gr,99):8.3f} "
          f"{np.median(vs):10.4f} {np.median(gs):10.4f}")
print("  ※ 압축 ε̃=ln(1+ε)/(1+ln(1+ε)) 의 동작점 점검: scaled 중앙이 0.1~0.5 구간이면 여유 있음.")
print("    0.05 미만이면 신호가 압축 선형구간에 몰려 분리도 손해 — P1 에서 offset 재검토.")
print("=" * 78)
