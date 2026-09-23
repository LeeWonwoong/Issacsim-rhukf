#!/usr/bin/env python3
"""scripts/check_posthover.py — '첫 hover 뒤 평시 관측 상승'(NONMYOPIC_REPORT 부록 A-1) 판독 · REENGAGE_FIX 검증용 (★09-24)

    python3 scripts/check_posthover.py <런폴더> [<런폴더> ...] [--label 이름]

정의(A-1 / critA-N3 과 같음): 공격 없는 에피(150–300행, HARD 리셋 앞붙음 제외) 중 첫 hover 가 40행 이후인 것.
평시 track 행(prev_action=0)만 쓰고, '첫 hover 전(행 10 ~ 첫 hover)' 과 '마지막 hover 해제 뒤 경과 16–30 / 31–60 / 61+' 를
g_obs·v_obs·gt_err 의 중앙·p90 으로 비교한다. steps 에 'reengage' 열(REENGAGE_FIX=1)이 있으면 재접근 행은 뺀다
(재접근 행은 trajectory_sp=현재 위치라 gt_err 의미가 다르다). hover 없는 에피의 초반(10–119)·후반(≥150)은 시간 대조군.

판정(REENGAGE_FIX=1 런): 해제 뒤 16–30 · 31–60 의 g_obs 중앙/전 비 ≤ 1.10, p90/전 비 ≤ 1.15 이면 '차이 소멸'.
  구 코드 실측: 중앙 ×1.21–1.40, p90 ×1.41–1.71 (옛 6런 n=171 · v12 R1–R3 n=23). n<8 이면 '판정 보류(표본 부족)'.
metrics csv 에 n_release_near·n_reengage·n_resync 가 있으면 합계와 '재동기 ≤ 재접근' 불변식도 찍는다.
"""
import argparse
import csv
import glob
import os

import numpy as np

BINS = [(16, 30), (31, 60), (61, 300)]


def analyse(runs):
    acc = {k: {'pre': [], **{b: [] for b in BINS}} for k in ('g_obs', 'v_obs', 'gt_err')}
    nep = nctl = n_re_rows = 0; ctl = {'early': [], 'late': []}
    for run in runs:
        for f in sorted(glob.glob(os.path.join(run, 'steps', 'ep*.npz'))):
            z = np.load(f, allow_pickle=True); cols = [str(c) for c in z['cols']]; R = z['rows']
            if R.ndim != 2 or len(R) < 150 or len(R) > 300:
                continue
            if np.any(np.diff(R[:, cols.index('step')]) < 0):
                continue
            pa = R[:, cols.index('prev_action')] > 0.5; atk = R[:, cols.index('atk_flag')] > 0.5
            if atk.any():
                continue
            re = R[:, cols.index('reengage')] > 0.5 if 'reengage' in cols else np.zeros(len(R), bool)
            idx = np.arange(len(R))
            if not pa.any():
                g = R[:, cols.index('g_obs')]; nctl += 1
                ctl['early'].extend(g[(idx >= 10) & (idx < 120)].tolist()); ctl['late'].extend(g[idx >= 150].tolist())
                continue
            first = int(np.argmax(pa))
            if first < 40:
                continue
            nep += 1; n_re_rows += int(re.sum())
            last = -10 ** 6; since = np.empty(len(R), int)
            for i in range(len(R)):
                if pa[i]: last = i
                since[i] = i - last
            base = ~pa & ~re
            for k in acc:
                x = R[:, cols.index(k)]
                acc[k]['pre'].extend(x[base & (idx >= 10) & (idx < first)].tolist())
                for b in BINS:
                    acc[k][b].extend(x[base & (idx > first) & (since >= b[0]) & (since <= b[1])].tolist())
    return acc, nep, nctl, ctl, n_re_rows


def fix_counters(runs):
    tot = {'n_release_near': 0, 'n_reengage': 0, 'n_resync': 0}; bad = 0; n = 0
    for run in runs:
        for mf in glob.glob(os.path.join(run, 'metrics_*.csv')):
            with open(mf) as fh:
                for row in csv.DictReader(fh):
                    if 'n_resync' not in row:
                        continue
                    n += 1
                    for k in tot: tot[k] += int(float(row[k]))
                    bad += int(float(row['n_resync']) > float(row['n_reengage']))
    return (tot, bad, n) if n else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+'); ap.add_argument('--label', default='')
    a = ap.parse_args()
    acc, nep, nctl, ctl, n_re = analyse(a.runs)
    fm = lambda x: f"{np.median(x):.3f}·{np.percentile(x, 90):.3f} ({len(x)})" if len(x) else '—'
    print(f'## {a.label or ", ".join(a.runs)}: 공격 없는 에피 n={nep}(첫 hover≥40) · hover 0 대조 n={nctl} · 제외한 재접근 행 {n_re}')
    print('| 양 | 첫 hover 전 중앙·p90 (n) | 해제 뒤 16–30 | 31–60 | 61+ |'); print('|---|---|---|---|---|')
    for k, d in acc.items():
        print(f"| {k} | {fm(d['pre'])} | " + ' | '.join(fm(d[b]) for b in BINS) + ' |')
    if ctl['early']:
        print(f"| 대조 g_obs (hover 0 에피) | 초반 10–119 {fm(ctl['early'])} | 후반 ≥150 {fm(ctl['late'])} | | |")
    g = acc['g_obs']
    if g['pre'] and g[BINS[0]] and g[BINS[1]]:
        pm, pp = np.median(g['pre']), np.percentile(g['pre'], 90)
        rat = [(np.median(g[b]) / pm, np.percentile(g[b], 90) / pp) for b in BINS[:2]]
        print('g_obs 해제 뒤/전 비 (중앙, p90): ' + ' · '.join(f'{b[0]}–{b[1]} ×{m:.2f}, ×{p:.2f}' for b, (m, p) in zip(BINS, rat)))
        if nep < 8:
            print('판정: 보류 (표본 부족 n<8 — 런을 늘리거나 합쳐서 볼 것)')
        else:
            ok = all(m <= 1.10 and p <= 1.15 for m, p in rat)
            print(f'판정: {"차이 소멸 (합격)" if ok else "차이 잔존 (불합격)"}  [기준 중앙 ≤ ×1.10 · p90 ≤ ×1.15 ; 구 코드 ×1.21–1.40 · ×1.41–1.71]')
    fc = fix_counters(a.runs)
    if fc:
        tot, bad, n = fc
        print(f'REENGAGE_FIX 계측 {n}에피: 근접 해제 {tot["n_release_near"]} · 재접근 {tot["n_reengage"]} · 재동기 {tot["n_resync"]} · '
              f'재동기>재접근 에피 {bad} (0 이어야 함)')


if __name__ == '__main__':
    main()
