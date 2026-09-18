#!/usr/bin/env python3
"""analyze_roll_band.py — 하이재킹(이탈) 밴드 판정 (2026-08-13)

  ~/isaacsim/python.sh analyze_roll_band.py <outdir> [<outdir2> ...] [--auth 4.36]

결과성 = 추락이 아니라 **setpoint 이탈**(하이재킹). 3조건:
  ① 결과성    track(무방어) 표류 > θ           (setpoint 벗어남)
  ② 탐지가능  공격 중 gyro NIS 상승
  ③ 대응가능  dhover 표류 ≈ 0 (명령 거부하면 표류 억제) ∧ 생존
밴드 = track 이탈 ∧ dhover 억제 인 δ 구간. (hover 패턴은 baseline≈0 이라 표류=순수 하이재킹량)
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load(outdir):
    surv = defaultdict(list)       # (pat,bias,policy) → survived
    crash = defaultdict(list)      # (pat,bias,policy) → crash_reason
    drift = defaultdict(list)      # (pat,bias,policy) → gt_err (attack on)
    nis = defaultdict(list)        # (pat,bias,policy) → gyro NIS (attack on)
    rollm = defaultdict(list)      # (pat,bias,policy) → |roll| (attack on)
    sp = os.path.join(outdir, 'sweep_summary.csv')
    dp = os.path.join(outdir, 'sweep_detail.csv')
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp)):
            try:
                k = (r['pattern'], float(r['bias']), r['policy'])
                surv[k].append(int(r['survived'])); crash[k].append(r['crash_reason'])
            except (ValueError, KeyError):
                pass
    if os.path.exists(dp):
        for r in csv.DictReader(open(dp)):
            try:
                if r['attack_active'] != '1':
                    continue
                k = (r['pattern'], float(r['bias']), r['policy'])
                drift[k].append(float(r['gt_err']))
                nis[k].append(float(r['nis_g_raw']))
                rollm[k].append(abs(float(r['roll'])))
            except (ValueError, KeyError):
                pass
    return surv, crash, drift, nis, rollm


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    auth = 4.36
    for a in sys.argv[1:]:
        if a.startswith('--auth'):
            auth = float(a.split('=')[1]) if '=' in a else 4.36
    for outdir in args or ['results_roll_band_0813']:
        surv, crash, drift, nis, rollm = load(outdir)
        if not surv:
            print(f"[{outdir}] 데이터 없음"); continue
        pats = sorted({k[0] for k in surv})
        biases = sorted({k[1] for k in surv})
        delays = sorted({int(k[2][6:]) for k in surv if k[2].startswith('dhover')})
        print(f"\n{'='*78}\n {outdir}   (권한 {auth} N·m 정규화)\n{'='*78}")
        for pat in pats:
            print(f"\n [{pat}]  δ=권한대비.  track=무방어 / dhover=명령거부")
            hdr = f"  {'δ':>4s} {'Nm':>5s} | {'trkSv':>5s} " + \
                  ' '.join(f'dh{d}Sv' for d in delays) + \
                  f" | {'trk표류p95':>9s} {'dh표류p95':>8s} | {'NIS공p95':>8s} {'roll°':>5s} | crash"
            print(hdr)
            for b in biases:
                d = b / auth
                def sv(pol):
                    v = surv.get((pat, b, pol), []); return np.mean(v) if v else np.nan
                tk_dr = drift.get((pat, b, 'track'), [np.nan])
                dh_dr = [x for dd in delays for x in drift.get((pat, b, f'dhover{dd}'), [])] or [np.nan]
                ns = nis.get((pat, b, 'track'), [np.nan])
                rl = rollm.get((pat, b, 'track'), [np.nan])
                cr = defaultdict(int)
                for c in crash.get((pat, b, 'track'), []):
                    if c not in ('timeout', '-1', ''):
                        cr[c] += 1
                crs = ','.join(f'{k}:{v}' for k, v in sorted(cr.items(), key=lambda x: -x[1])[:2]) or '-'
                svs = ' '.join(f'{sv("dhover"+str(dd)):5.2f}' for dd in delays)
                print(f"  {d:4.2f} {b:5.2f} | {sv('track'):5.2f} {svs} | "
                      f"{np.nanpercentile(tk_dr,95):9.2f} {np.nanpercentile(dh_dr,95):8.2f} | "
                      f"{np.nanpercentile(ns,95):8.1f} {np.degrees(np.nanmax(rl)):5.1f} | {crs}")
        # 밴드 판정 (hover 패턴 기준: track표류>0.5 ∧ dhover표류<0.3 ∧ track생존>0)
        print(f"\n  ─ 하이재킹 밴드 (track 이탈>0.5m ∧ dhover 억제<0.3m ∧ 미추락) ─")
        for pat in pats:
            band = []
            for b in biases:
                tk = np.nanpercentile(drift.get((pat, b, 'track'), [0]), 95)
                dh = np.nanpercentile([x for dd in delays for x in drift.get((pat, b, f'dhover{dd}'), [])] or [9], 95)
                sv = np.mean(surv.get((pat, b, 'track'), [0]))
                if tk > 0.5 and dh < 0.3 and sv > 0:
                    band.append(round(b/auth, 2))
            print(f"    [{pat}]  δ ∈ {band}" if band else f"    [{pat}]  밴드 없음")


if __name__ == '__main__':
    main()
