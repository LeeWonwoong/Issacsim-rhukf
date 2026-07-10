#!/usr/bin/env python3
"""
compare_adam_delay.py — 수정전(results_adam) vs 수정후(results_adam_v2) 비교.
  핵심 확인: 탐지지연 분포 이봉(6-7 + 14-15) → 단봉(첫봉 3-4)으로 당겨졌나?
            crash rate가 지연 개선과 함께 떨어졌나?
사용: python3 compare_adam_delay.py [dirA] [dirB]
"""
import csv, sys
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else 'results_adam'       # 수정전
B = sys.argv[2] if len(sys.argv) > 2 else 'results_adam_v2'    # 수정후


def load(d):
    rows = list(csv.DictReader(open(f'{d}/metrics_adam.csv')))
    dd = np.array([float(r['det_delay']) for r in rows])
    cr = np.array([float(r['crashed']) for r in rows])
    f1 = np.array([float(r['f1']) for r in rows])
    rec = np.array([float(r['recall']) for r in rows])
    fp = np.array([float(r['fp_rate']) for r in rows])
    atk = dd >= 0                          # 공격 에피소드(지연 정의됨)
    return dict(n=len(rows), dd=dd, cr=cr, f1=f1, rec=rec, fp=fp, atk=atk)


def hist_delay(dd_atk, label):
    bins = [0, 3, 6, 9, 12, 15, 100]
    names = ['0-2', '3-5', '6-8', '9-11', '12-14', '15+']
    h, _ = np.histogram(dd_atk, bins=bins)
    tot = max(h.sum(), 1)
    print(f"  {label}: n={h.sum()}  중앙값={np.median(dd_atk):.1f} 평균={dd_atk.mean():.1f}")
    for nm, c in zip(names, h):
        bar = '█' * int(40 * c / tot)
        print(f"    {nm:>5}스텝 | {bar} {c} ({100*c/tot:.0f}%)")


def summ(d, name):
    m = d['atk']
    late = d['dd'][m]
    print(f"\n[{name}] 공격ep={m.sum()}/{d['n']}")
    print(f"  탐지지연 median={np.median(late):.1f} mean={late.mean():.1f}")
    print(f"  crash={d['cr'][m].mean():.3f}  F1={np.nanmean(d['f1'][m]):.3f} "
          f"recall={np.nanmean(d['rec'][m]):.3f}  FPr={d['fp'][m].mean():.3f}")
    # 이봉성: 3-5 봉 vs 12+ 봉
    early = ((late >= 3) & (late <= 5)).mean()
    veryearly = (late <= 4).mean()
    latepk = (late >= 12).mean()
    print(f"  조기(≤4스텝) 비율={veryearly:.0%}  둘째봉(≥12스텝) 비율={latepk:.0%}")


def main():
    try:
        da = load(A); has_a = True
    except Exception as e:
        print(f"[{A}] 로드 실패: {e}"); has_a = False
    try:
        db = load(B); has_b = True
    except Exception as e:
        print(f"[{B}] 로드 실패(아직 학습중?): {e}"); has_b = False

    print("=" * 60)
    if has_a:
        summ(da, f'수정전 {A}')
        print("  지연 히스토그램:")
        hist_delay(da['dd'][da['atk']], '수정전')
    if has_b:
        summ(db, f'수정후 {B}')
        print("  지연 히스토그램:")
        hist_delay(db['dd'][db['atk']], '수정후')

    if has_a and has_b:
        print("\n" + "=" * 60)
        print("[비교표] (공격ep 기준)")
        ma, mb = da['atk'], db['atk']
        def line(k, va, vb, fmt='{:.2f}', better='down'):
            arrow = '↓개선' if ((vb < va) == (better == 'down')) else '↑악화'
            print(f"  {k:14s} | 전 {fmt.format(va):>7} → 후 {fmt.format(vb):>7}  {arrow}")
        line('탐지지연median', np.median(da['dd'][ma]), np.median(db['dd'][mb]), '{:.1f}')
        line('crash rate', da['cr'][ma].mean(), db['cr'][mb].mean())
        line('둘째봉≥12비율', (da['dd'][ma] >= 12).mean(), (db['dd'][mb] >= 12).mean())
        line('F1', np.nanmean(da['f1'][ma]), np.nanmean(db['f1'][mb]), '{:.3f}', 'up')
        line('FPr', da['fp'][ma].mean(), db['fp'][mb].mean())


if __name__ == '__main__':
    main()
