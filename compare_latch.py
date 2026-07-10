#!/usr/bin/env python3
"""
compare_latch.py — latch 순효과 측정: v2(latch전) vs v3(latch후).
  핵심: 하단[1.34,1.37] 순수홀드침하 crash가 잡혔나 / delay는 유지되나(latch는 탐지무관) / relapse·min_alt.
사용: python3 compare_latch.py [dirA=results_adam_v2] [dirB=results_adam_v3]
"""
import re, csv, sys
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else 'results_adam_v2'
B = sys.argv[2] if len(sys.argv) > 2 else 'results_adam_v3'


def load(d):
    md = {}
    for r in csv.DictReader(open(f'{d}/metrics_adam.csv')):
        md[int(r['episode'])] = dict(
            tp=int(r['tp']), dd=float(r['det_delay']), crash=int(r['crashed']),
            relapse=int(r['relapse']) if 'relapse' in r and r['relapse'] != '' else None,
            min_alt=float(r['min_alt']) if 'min_alt' in r and r['min_alt'] != '' else None)
    # s(bias) from train.log (Attack ON 라인 3-proxy 평균)
    ep = None; on = None; sm = {}
    for ln in open(f'{d}/train.log'):
        h = re.search(r'TRAIN Ep (\d+)/200', ln)
        if h: ep = int(h.group(1)); on = None; continue
        a = re.search(r'Attack ON.*τxy=\+?([0-9.]+).*τz=\+?([0-9.]+).*thrust=\+?([0-9.]+)', ln)
        if a and ep: on = (float(a.group(1)), float(a.group(2)), float(a.group(3)))
        e = re.search(r'Ep (\d+): .*(CRASH|FLIP|DRIFT|TIMEOUT)', ln)
        if e and ep:
            n = int(e.group(1))
            if on: sm[n] = np.mean([on[0], on[1] / 0.2, on[2] / 1.5])
            ep = None
    rows = []
    for n, s in sm.items():
        m = md.get(n)
        if m: rows.append(dict(ep=n, s=s, **m))
    return rows


def report(rows, name):
    r = rows
    s = np.array([x['s'] for x in r]); cr = np.array([x['crash'] for x in r])
    dd = np.array([x['dd'] for x in r])
    print(f"\n[{name}] 공격ep={len(r)}")
    print(f"  전체 crash={cr.mean():.3f} | 지연median={np.median(dd[dd>=0]):.1f}")
    for lo, hi, lab in [(0, 1.37, '하단<1.37'), (1.37, 9, '상단≥1.37')]:
        m = (s >= lo) & (s < hi)
        if m.sum():
            d = dd[m]; d = d[d >= 0]
            print(f"    {lab}: n={m.sum():2d} crash={cr[m].mean():.0%} 지연med={np.median(d):.1f}")
    # relapse/min_alt (v3만)
    rel = [x['relapse'] for x in r if x['relapse'] is not None]
    ma = [x['min_alt'] for x in r if x['min_alt'] is not None and x['min_alt'] >= 0]
    if rel:
        print(f"  [진단] relapse median={np.median(rel):.0f}(범위{min(rel)}~{max(rel)}) | "
              f"min_alt median={np.median(ma):.1f}m")
        # relapse 있는 에피소드의 생존율 (latch가 플리커를 구제하나)
        rl = np.array([x['relapse'] for x in r if x['relapse'] is not None])
        crr = np.array([x['crash'] for x in r if x['relapse'] is not None])
        hi_rel = rl >= 1
        if hi_rel.sum():
            print(f"    relapse≥1 에피소드 crash={crr[hi_rel].mean():.0%} (n={hi_rel.sum()}) "
                  f"vs relapse=0 crash={crr[~hi_rel].mean():.0%} (n={(~hi_rel).sum()})")


def main():
    try:
        ra = load(A)
    except Exception as e:
        print(f"[{A}] 로드실패: {e}"); ra = None
    try:
        rb = load(B)
    except Exception as e:
        print(f"[{B}] 로드실패(학습중?): {e}"); rb = None
    if ra: report(ra, f'latch전 {A}')
    if rb: report(rb, f'latch후 {B}')
    if ra and rb:
        sa = np.array([x['s'] for x in ra]); ca = np.array([x['crash'] for x in ra])
        sb = np.array([x['s'] for x in rb]); cb = np.array([x['crash'] for x in rb])
        print("\n" + "=" * 55 + "\n[판정] 하단<1.37 crash:")
        la = (sa < 1.37); lb = (sb < 1.37)
        print(f"  {ca[la].mean():.0%} → {cb[lb].mean():.0%}  "
              f"({'대폭↓ latch가 핵심' if cb[lb].mean() < ca[la].mean() - 0.15 else '여전 → latch 부족'})")


if __name__ == '__main__':
    main()
