#!/usr/bin/env python3
"""latch 단독 격리 학습 분석: v3(latch-fix) vs v3_prelatch(broken latch).
- 최종20% crash/min_alt/relapse/delay/f1 비교
- train.log 파싱으로 에피소드별 s(=τ) 복원 → s별 crash 분해
사용: python3 analyze_latch.py [results_adam_v3] [results_adam_v3_prelatch]
"""
import sys, csv, os, re
from collections import defaultdict

NEW = sys.argv[1] if len(sys.argv) > 1 else 'results_adam_v3'
OLD = sys.argv[2] if len(sys.argv) > 2 else 'results_adam_v3_prelatch'


def tail_metrics(d, frac=0.2):
    f = os.path.join(d, 'metrics_adam.csv')
    if not os.path.exists(f):
        return None, 0
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None, 0
    k = max(1, int(len(rows) * frac))
    tail = rows[-k:]
    def a(c):
        v = [float(r[c]) for r in tail if r.get(c) not in (None, '', 'nan')]
        return sum(v) / len(v) if v else float('nan')
    cols = ['crashed', 'min_alt', 'relapse', 'det_delay', 'f1', 'reward', 'fp_rate', 'recall']
    return {c: a(c) for c in cols if c in rows[0]}, len(rows)


def episode_s_map(d):
    """train.log에서 에피소드별 첫 ATK τ값(=s) 복원."""
    log = os.path.join(d, 'train.log')
    if not os.path.exists(log):
        return {}
    ep = None
    ep_s = {}
    hdr = re.compile(r'TRAIN Ep (\d+)/')
    atk = re.compile(r'🔴ATK\(τ([0-9.]+)')
    for line in open(log, errors='ignore'):
        m = hdr.search(line)
        if m:
            ep = int(m.group(1))
            continue
        if ep is not None and ep not in ep_s:
            a = atk.search(line)
            if a:
                ep_s[ep] = round(float(a.group(1)), 2)
    return ep_s


def crash_by_s(d):
    f = os.path.join(d, 'metrics_adam.csv')
    if not os.path.exists(f):
        return {}
    rows = {int(r['episode']): r for r in csv.DictReader(open(f)) if r.get('episode')}
    smap = episode_s_map(d)
    agg = defaultdict(lambda: [0, 0])  # s -> [crash_sum, n]
    for ep, s in smap.items():
        if ep in rows and rows[ep].get('crashed') not in (None, '', 'nan'):
            agg[s][0] += float(rows[ep]['crashed'])
            agg[s][1] += 1
    return {s: (c / n, n) for s, (c, n) in sorted(agg.items())}


print(f"{'':14}{'NEW(latch-fix)':>18}{'OLD(prelatch)':>18}")
mn, nn = tail_metrics(NEW)
mo, no = tail_metrics(OLD)
if mn and mo:
    for c in ['crashed', 'min_alt', 'relapse', 'det_delay', 'f1', 'reward', 'recall', 'fp_rate']:
        if c in mn:
            print(f"{c:14}{mn[c]:>18.3f}{mo.get(c, float('nan')):>18.3f}")
    print(f"{'eps':14}{nn:>18}{no:>18}")
else:
    print(f"  metrics 없음 (NEW={bool(mn)} OLD={bool(mo)})")

print("\n=== s별 crash 분해 (밴드 1.34~1.40) ===")
for tag, d in [('NEW ', NEW), ('OLD ', OLD)]:
    cb = crash_by_s(d)
    line = '  '.join(f"s{s}:{c:.2f}(n{n})" for s, (c, n) in cb.items())
    print(f"{tag}: {line if line else '(s복원 실패/데이터없음)'}")
