#!/usr/bin/env python3
"""innov_analysis.py — raw innovation(잔차) vs NIS 를 구간·조건별 분리 (2026-08-17)

  ~/isaacsim/python.sh innov_analysis.py <outdir>

NIS = res^T S^-1 res 는 R 로 정규화됨(R 크면 눌림). raw |res| 는 정규화 전 잔차.
고집 필터(R 큼)면 외란(바람/기동)에 raw innovation 은 커도 NIS 는 작을 수 있음.
→ 둘을 분리해, 바람/기동이 실제로 상태추정을 흔드는지(=raw innovation) 본다.
시간창: 바람 80~220, 공격 140~260.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def region(step):
    if step < 80: return 'clean'
    if step < 140: return 'wind'
    if step < 220: return 'overlap'
    if step < 260: return 'atk'
    return 'recov'


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_innov_check'
    # (pattern, dtype, bias, region) -> list of (res_v,res_g,nis_v,nis_g)
    agg = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(outdir, 'sweep_detail.csv'))):
        if r['policy'] != 'track':
            continue
        try:
            rg = region(int(r['step']))
            k = (r['pattern'], r['disturbance_type'], float(r['bias']), rg)
            agg[k].append((float(r['res_v']), float(r['res_g']),
                           float(r['nis_v_raw']), float(r['nis_g_raw'])))
        except (ValueError, KeyError):
            pass

    def stat(k):
        v = agg.get(k, [])
        if not v: return None
        a = np.array(v)
        return np.median(a, axis=0)   # res_v, res_g, nis_v, nis_g

    print(f"\n{'='*82}\n {outdir}  — raw innovation |res| vs NIS (구간·조건별 p50)\n{'='*82}")
    print("  res=정규화 전 잔차크기,  NIS=R로 정규화(고집이면 눌림)")
    pats = sorted({k[0] for k in agg})
    for pat in pats:
        print(f"\n [{pat}]")
        print(f"  {'조건/구간':>22s} | {'res_v':>7s} {'res_g':>7s} | {'NIS_v':>7s} {'NIS_g':>7s}")
        # 평시: clean / wind (bias0)
        for dtype in ['none', 'wind_turbulence']:
            for rg in ['clean', 'wind']:
                s = stat((pat, dtype, 0.0, rg))
                if s is not None:
                    lab = f"{dtype[:12]}·{rg}(평시)"
                    print(f"  {lab:>22s} | {s[0]:7.3f} {s[1]:7.3f} | {s[2]:7.2f} {s[3]:7.2f}")
        # 공격: overlap(공격+바람) / atk(공격만) — 강풍셀 기준
        for rg in ['overlap', 'atk']:
            s = stat((pat, 'wind_turbulence', 3.05, rg))
            if s is not None:
                print(f"  {'ws8·'+rg+'(공격)':>22s} | {s[0]:7.3f} {s[1]:7.3f} | {s[2]:7.2f} {s[3]:7.2f}")
    print("\n─ 핵심 질문 ─────────────────────────────────────────────")
    print("  · 바람/기동 평시에서 res_g/res_v 가 clean 보다 크면 = 외란이 실제 innovation 을 키움")
    print("    (그런데 NIS 는 R 로 눌려 작게 보였던 것 → 원웅님 지적이 맞음).")
    print("  · res 도 clean 과 같으면 = 외란이 이 sim 에선 정말 innovation 을 안 만듦(모델 정확+약한 외란).")


if __name__ == '__main__':
    main()
