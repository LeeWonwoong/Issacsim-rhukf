#!/usr/bin/env python3
"""strongwind_analysis.py — 강풍(9~15) perceptual aliasing: res·NIS 구간별 + 윈도우 d′ (2026-08-18)

  ~/isaacsim/python.sh strongwind_analysis.py <outdir>

가설: 약풍은 PX4 보상→NIS 무반응. 강풍(무게의 20~54%)은 보상 못해 드론 밀림→innovation↑→aliasing.
시간창: 바람 80~220, 공격 140~260. track 정책. 전 궤적 합산(hover 따로).
각 (외란,세기): 평시 clean/wind 의 res·NIS(바람이 innovation 만드나) +
  탐지 d′(겹침=공격+바람 vs 바람만) 단일 vs 윈도우-max(4), gyro·vel.
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


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return float('nan')
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def winmax(v, w=4):
    return [max(v[max(0, i-w+1):i+1]) for i in range(len(v))]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results_strongwind'
    hover_only = '--hover' in sys.argv
    seq = defaultdict(list)   # (dtype,ws,bias,pat,ep) -> [(step,res_v,res_g,nis_v,nis_g)]
    for r in csv.DictReader(open(os.path.join(outdir, 'sweep_detail.csv'))):
        if r['policy'] != 'track':
            continue
        try:
            k = (r['disturbance_type'], float(r['wind_speed']), float(r['bias']), r['pattern'], r['episode'])
            seq[k].append((int(r['step']), float(r['res_v']), float(r['res_g']),
                           float(r['nis_v_raw']), float(r['nis_g_raw'])))
        except (ValueError, KeyError):
            pass

    # (dtype,ws) -> region -> {'res_v','res_g','nis_v','nis_g'} lists  + 윈도우용 겹침/바람 채널별
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    win = defaultdict(lambda: defaultdict(list))   # (dtype,ws) -> {'ov_g_s','ov_g_w','wo_g_s','wo_g_w', vel...}
    for (dtype, ws, bias, pat, ep), rows in seq.items():
        if hover_only and pat != 'hover':
            continue
        rows.sort(key=lambda x: x[0])
        steps = [r[0] for r in rows]
        rg_ = [r[2] for r in rows]; rv_ = [r[1] for r in rows]
        ng_ = [r[4] for r in rows]; nv_ = [r[3] for r in rows]
        wm_g = winmax(ng_, 4); wm_v = winmax(nv_, 4)
        key = (dtype, ws)
        for i, st in enumerate(steps):
            rg = region(st)
            if rg in ('overlap', 'atk') and bias <= 1e-9:
                continue
            if bias <= 1e-9 and rg in ('clean', 'wind'):
                by[key][rg]['res_v'].append(rv_[i]); by[key][rg]['res_g'].append(rg_[i])
                by[key][rg]['nis_v'].append(nv_[i]); by[key][rg]['nis_g'].append(ng_[i])
                if rg == 'wind':
                    win[key]['wo_g_s'].append(ng_[i]); win[key]['wo_g_w'].append(wm_g[i])
                    win[key]['wo_v_s'].append(nv_[i]); win[key]['wo_v_w'].append(wm_v[i])
            elif bias > 1e-9 and rg == 'overlap':
                by[key]['overlap']['res_g'].append(rg_[i]); by[key]['overlap']['nis_g'].append(ng_[i])
                win[key]['ov_g_s'].append(ng_[i]); win[key]['ov_g_w'].append(wm_g[i])
                win[key]['ov_v_s'].append(nv_[i]); win[key]['ov_v_w'].append(wm_v[i])

    tag = 'hover만' if hover_only else '전궤적'
    print(f"\n{'='*100}\n {outdir} [{tag}] — 강풍 innovation & 탐지 d′ (p50)\n{'='*100}")
    print(f"  {'외란':>15s} {'ws':>4s} | {'clean res_g':>11s} {'wind res_g':>10s} | {'clean nis_v':>11s} {'wind nis_v':>10s} "
          f"| {'gyro d′ 단일/윈':>13s} {'vel d′ 단일/윈':>13s}")
    def med(k, rg, col):
        v = by[k][rg][col]; return np.median(v) if v else float('nan')
    for key in sorted(by, key=lambda k: (k[0], k[1])):
        dtype, ws = key
        w = win[key]
        dg_s = dprime(w.get('ov_g_s', []), w.get('wo_g_s', []))
        dg_w = dprime(w.get('ov_g_w', []), w.get('wo_g_w', []))
        dv_s = dprime(w.get('ov_v_s', []), w.get('wo_v_s', []))
        dv_w = dprime(w.get('ov_v_w', []), w.get('wo_v_w', []))
        print(f"  {dtype:>15s} {ws:4.1f} | {med(key,'clean','res_g'):11.3f} {med(key,'wind','res_g'):10.3f} "
              f"| {med(key,'clean','nis_v'):11.2f} {med(key,'wind','nis_v'):10.2f} "
              f"| {dg_s:5.2f}/{dg_w:5.2f}   {dv_s:5.2f}/{dv_w:5.2f}")
    print("\n─ 읽는 법 ────────────────────────────────────────────")
    print("  · wind res_g/nis_v 가 clean 보다 크게 오르면 = 강풍이 PX4 보상 넘겨 innovation 만듦(aliasing 원천).")
    print("  · 그 강풍서 gyro/vel d′ 가 낮아지면(공격이 바람에 묻힘) = 진짜 aliasing. 윈도우 d′>단일 이면 윈도우가 해소.")


if __name__ == '__main__':
    main()
