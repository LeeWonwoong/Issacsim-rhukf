#!/usr/bin/env python3
"""wind_pomdp_analysis.py — 시간창 바람 POMDP: 양채널(gyro/vel) 구간별 NIS + 윈도우(4) (2026-08-17)

  ~/isaacsim/python.sh wind_pomdp_analysis.py <const_dir> <pulse_dir> [--ws 80 --we 220 --as 140 --ae 260]

시간창(기본): 바람 80~220, 공격 140~260 →
  clean(<80) · 바람만(80~140) · 겹침(140~220) · 공격만(220~260) · 복귀(260+)
공격=토크→gyro 주채널, 바람(turbulence)=병진력→vel 주채널 → aliasing 은 주로 vel.
패턴별(hover=깨끗 / aggressive=기동aliasing) × (외란,세기) 로 나눠, gyro·vel 각각:
  구간 NIS p50/p95, 탐지 d′(겹침 vs 바람만) 단일 vs 윈도우-max(4).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

WS, WE, AS, AE = 80, 220, 140, 260


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return float('nan')
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def winmax(vals, w=4):
    return [max(vals[max(0, i - w + 1):i + 1]) for i in range(len(vals))]


def region(step):
    if step < WS: return 'clean'
    if step < AS: return 'wind'
    if step < WE: return 'overlap'
    if step < AE: return 'atk'
    return 'recov'


def load(outdir):
    seq = defaultdict(list)   # (pat,dtype,ws,bias,ep) -> [(step,nis_g,nis_v)]
    dp = os.path.join(outdir, 'sweep_detail.csv')
    if not os.path.exists(dp):
        return seq
    for r in csv.DictReader(open(dp)):
        if r['policy'] != 'track':
            continue
        try:
            k = (r['pattern'], r['disturbance_type'], float(r['wind_speed']), float(r['bias']), r['episode'])
            seq[k].append((int(r['step']), float(r['nis_g_raw']), float(r['nis_v_raw'])))
        except (ValueError, KeyError):
            pass
    return seq


def collect(seq, ch):
    # ch: 1=gyro, 2=vel.  (pat,dtype,ws) -> region -> {'single':[], 'win':[]}
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (pat, dtype, ws, bias, ep), rows in seq.items():
        rows.sort(key=lambda x: x[0])
        vals = [r[ch] for r in rows]; steps = [r[0] for r in rows]
        wm = winmax(vals, 4)
        key = (pat, dtype, ws)
        for i, st in enumerate(steps):
            rg = region(st)
            if rg in ('overlap', 'atk') and bias <= 1e-9:
                continue
            by[key][rg]['single'].append(vals[i])
            by[key][rg]['win'].append(wm[i])
    return by


def report(name, seq):
    for ch, chn in [(1, 'gyro'), (2, 'vel')]:
        by = collect(seq, ch)
        print(f"\n=== {name} [{chn} NIS] 구간 p50(p95) + 탐지 d′(겹침 vs 바람만) ===")
        print(f"  {'패턴':>10s} {'외란':>15s} {'ws':>4s} | {'clean':>10s} {'바람만':>10s} {'겹침':>10s} {'복귀':>9s} "
              f"| {'단일d′':>6s} {'윈도d′':>6s} {'Δ':>5s}")
        for key in sorted(by, key=lambda k: (k[0], k[1], k[2])):
            pat, dtype, ws = key
            rg = by[key]
            def pp(r):
                v = rg[r]['single']
                return f"{np.median(v):.1f}({np.percentile(v,95):.0f})" if v else "-"
            ov_s, ov_w = np.array(rg['overlap']['single']), np.array(rg['overlap']['win'])
            wo_s, wo_w = np.array(rg['wind']['single']), np.array(rg['wind']['win'])
            d_s = dprime(ov_s, wo_s); d_w = dprime(ov_w, wo_w)
            print(f"  {pat:>10s} {dtype:>15s} {ws:4.1f} | {pp('clean'):>10s} {pp('wind'):>10s} "
                  f"{pp('overlap'):>10s} {pp('recov'):>9s} | {d_s:6.2f} {d_w:6.2f} {d_w-d_s:5.2f}")


def main():
    global WS, WE, AS, AE
    dirs = [x for x in sys.argv[1:] if not x.startswith('--')]
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == '--ws': WS = int(argv[i+1])
        elif x == '--we': WE = int(argv[i+1])
        elif x == '--as': AS = int(argv[i+1])
        elif x == '--ae': AE = int(argv[i+1])
    print(f"창(스텝): 바람 {WS}~{WE}, 공격 {AS}~{AE} | clean<{WS} 바람만{WS}~{AS} 겹침{AS}~{WE} 공격만{WE}~{AE} 복귀{AE}+")
    labels = ['상수(const)', '펄스(pulse)']
    for i, od in enumerate(dirs):
        seq = load(od)
        if not seq:
            print(f"[{od}] detail 없음"); continue
        report(labels[i] if i < len(labels) else od, seq)
    print("\n─ 읽는 법 ──────────────────────────────────────────────")
    print("  · 바람만 NIS > clean 이면 = 바람이 그 채널 NIS 튀김(aliasing 원천). turbulence 는 주로 vel.")
    print("  · aggressive clean NIS > hover clean = 기동 자체가 aliasing(주 원천, 프로젝트 문서).")
    print("  · 겹침 vs 바람만 d′: 바람 위 공격 탐지력. 윈도우d′>단일(Δ>0), 강풍서 Δ↑ = 윈도우가 aliasing 해소.")
    print("  · const vs pulse: 바람서 d′ 더 버티거나 윈도우 이득 큰 쪽 채택.")


if __name__ == '__main__':
    main()
