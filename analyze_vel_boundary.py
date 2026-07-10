"""
analyze_vel_boundary.py — combined vel-활성화 경계 스캔 자동 판독
================================================================
results_combined_ft20/ft25/ft35 을 읽어 셀별:
  injected T(N), vel_mean, gyr_mean, vel/gyr, track/hover/dhover3 생존율
두 임계 판정:
  (a) vel 유의 활성화  = (vel/gyr >= 0.20) OR (vel_mean > 5.0)     [정상 99pct≈3.2]
  (b) 대응 유지        = (hover 생존 >= 0.8) AND (dhover3 생존 >= 0.8)
겹침(=(a)∧(b)) 셀 존재 여부로 시나리오 분기 신호 출력.

vel_mean/gyr_mean 은 track 정책 attack-active 윈도우(=공격 시그니처 노출) 기준.
(hover 값도 참고로 병기.)

사용: python3 analyze_vel_boundary.py
"""
import os, csv
from collections import defaultdict
import numpy as np

FOLDERS = [
    ("results_combined_ft20", 2.0),
    ("results_combined_ft25", 2.5),
    ("results_combined_ft35", 3.5),
]
NORM_VEL_MEAN = 0.27
NORM_VEL_99   = 3.2
A_RATIO = 0.20     # vel/gyr 임계
A_ABS   = 5.0      # vel_mean 절대 임계
B_SURV  = 0.80     # 대응유지 생존율 임계


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bval(r):
    return float(r.get('bias', r.get('alpha', 0.0)))


def cell_vel_gyr(detail, b, pol):
    v, g = [], []
    for r in detail:
        if bval(r) == b and r['policy'] == pol and r['attack_active'] == '1':
            v.append(float(r['nis_v_raw'])); g.append(float(r['nis_g_raw']))
    return (np.mean(v) if v else float('nan'),
            np.mean(g) if g else float('nan'),
            len(v))


def survrate(summ, b, pol):
    s = [int(r['survived']) for r in summ if bval(r) == b and r['policy'] == pol]
    return np.mean(s) if s else float('nan')


rows_out = []
overlap_cells = []

for folder, ftr in FOLDERS:
    sp = os.path.join(folder, 'sweep_summary.csv')
    dp = os.path.join(folder, 'sweep_detail.csv')
    if not (os.path.exists(sp) and os.path.exists(dp)):
        print(f"[!] {folder}: summary/detail 없음 — 스킵")
        continue
    summ = load_csv(sp)
    detail = load_csv(dp)
    inj = {}
    for r in summ:
        b = bval(r)
        if 'th_n' in r:
            inj[b] = float(r['th_n'])
    biases = sorted(set(bval(r) for r in summ if bval(r) > 0))
    for b in biases:
        T = inj.get(b, float('nan'))
        vt, gt, nt = cell_vel_gyr(detail, b, 'track')
        vh, gh, nh = cell_vel_gyr(detail, b, 'hover')
        ratio_t = vt / gt if gt and not np.isnan(gt) and gt != 0 else float('nan')
        trk = survrate(summ, b, 'track')
        hov = survrate(summ, b, 'hover')
        dh3 = survrate(summ, b, 'dhover3')
        a_flag = (not np.isnan(ratio_t) and ratio_t >= A_RATIO) or \
                 (not np.isnan(vt) and vt > A_ABS)
        b_flag = (not np.isnan(hov) and hov >= B_SURV) and \
                 (not np.isnan(dh3) and dh3 >= B_SURV)
        rec = dict(ftr=ftr, s=b, T=T, vt=vt, gt=gt, ratio=ratio_t,
                   vh=vh, gh=gh, trk=trk, hov=hov, dh3=dh3,
                   a=a_flag, b=b_flag)
        rows_out.append(rec)
        if a_flag and b_flag:
            overlap_cells.append(rec)

# ── 표 출력 ──
print("\n" + "=" * 100)
print("  COMBINED vel-활성화 경계 스캔 판독  (track 정책 attack-on 기준; 정상 vel_mean≈0.27, 99pct≈3.2)")
print("=" * 100)
hdr = (f"{'ft':>4} {'s':>5} {'T(N)':>6} | {'vel_m':>7} {'gyr_m':>8} {'vel/gyr':>8} | "
       f"{'track':>5} {'hover':>5} {'dhov3':>5} | {'(a)vel':>6} {'(b)resp':>7} {'both':>4}")
print(hdr)
print("-" * 100)
cur_ft = None
for r in rows_out:
    if cur_ft is not None and r['ftr'] != cur_ft:
        print("-" * 100)
    cur_ft = r['ftr']
    both = 'YES' if (r['a'] and r['b']) else ''
    print(f"{r['ftr']:>4} {r['s']:>5.2f} {r['T']:>6.2f} | "
          f"{r['vt']:>7.3f} {r['gt']:>8.2f} {r['ratio']:>8.4f} | "
          f"{r['trk']:>5.2f} {r['hov']:>5.2f} {r['dh3']:>5.2f} | "
          f"{('Y' if r['a'] else 'n'):>6} {('Y' if r['b'] else 'n'):>7} {both:>4}")
print("=" * 100)
print("  (a)vel 활성화 = vel/gyr>=0.20 OR vel_mean>5.0   |   (b)대응유지 = hover>=0.8 AND dhover3>=0.8")
print(f"  [참고] hover 정책 vel_mean/gyr_mean 도 계산됨(위 표엔 track만 표기).")

# ── 분기 신호 ──
print("\n" + "#" * 100)
if overlap_cells:
    print(f"  ▶ 겹침 발견: (a)∧(b) 동시 만족 셀 {len(overlap_cells)}개  → 시나리오 2 (combined 재검토, 정지)")
    for r in overlap_cells:
        print(f"      ft={r['ftr']} s={r['s']:.2f} T={r['T']:.2f}N  vel/gyr={r['ratio']:.3f} "
              f"vel_m={r['vt']:.2f}  hover={r['hov']:.2f} dhover3={r['dh3']:.2f}")
    print("  SCENARIO=2")
else:
    print("  ▶ 겹침 없음: (a)vel 활성화 ∩ (b)대응 가능 = 공집합  → 시나리오 1 (torque-only 최종 확정)")
    # vel 상승 배율(정상 대비) 참고
    vmax = max((r['vt'] for r in rows_out if not np.isnan(r['vt'])), default=float('nan'))
    print(f"  참고: track vel_mean 최대={vmax:.2f} (정상 {NORM_VEL_MEAN} 대비 ×{vmax/NORM_VEL_MEAN:.1f}), "
          f"모두 gyro의 <?% (표 vel/gyr 참조)")
    print("  SCENARIO=1")
print("#" * 100 + "\n")
