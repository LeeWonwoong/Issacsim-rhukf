#!/usr/bin/env python3
"""latch-band sweep 분석: 스크립트 hover의 실제 구제능력(생존 + min_alt).
판정:
  - 스크립트 latch도 밴드서 침하/추락(min_alt 낮음) → hover 물리한계 = 밴드 재정의
  - 스크립트 latch는 살고(min_alt 유지) RL만 침하 → RL 정책 문제
비교 기준: RL latch-fix run min_alt=3.98 (results_adam_v3), delay 9.5 → dhover10과 대응.
사용: python3 analyze_latch_band.py [results_latch_band]
"""
import sys, csv, os
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else 'results_latch_band'
RL_MINALT = 3.98   # results_adam_v3 latch-fix 최종20% (참조선)

f = os.path.join(D, 'sweep_summary.csv')
if not os.path.exists(f):
    print(f"없음: {f}"); sys.exit(1)
rows = list(csv.DictReader(open(f)))
if not rows:
    print("빈 summary"); sys.exit(1)

# (bias, policy) -> list of (survived, min_alt, reason)
cell = defaultdict(list)
for r in rows:
    b = float(r['bias']); pol = r['policy']
    surv = int(r['survived'])
    ma = float(r['min_alt']) if r.get('min_alt') not in (None, '', 'nan') else -1
    cell[(b, pol)].append((surv, ma, r.get('crash_reason', '')))

def stats(recs):
    n = len(recs)
    sr = sum(s for s, _, _ in recs) / n
    mas = [m for _, m, _ in recs if m >= 0]
    mean_ma = sum(mas) / len(mas) if mas else float('nan')
    min_ma = min(mas) if mas else float('nan')     # 최악 에피소드 침하
    return sr, mean_ma, min_ma, n

biases = sorted(set(b for b, _ in cell))
# policy 순서: track, hover, dhover1..14
def pol_key(p):
    if p == 'track': return (0, 0)
    if p == 'hover': return (1, 0)
    return (2, int(p[6:]))
policies = sorted(set(p for _, p in cell), key=pol_key)

print(f"참조: RL latch-fix min_alt≈{RL_MINALT}m (delay 9.5 ≈ dhover10)\n")
print(f"{'bias':>6} | " + " | ".join(f"{p:>14}" for p in policies))
print(f"{'':>6} | " + " | ".join(f"{'surv/mnA/wrst':>14}" for _ in policies))
print("-" * (9 + 17 * len(policies)))
for b in biases:
    parts = []
    for p in policies:
        if (b, p) in cell:
            sr, mean_ma, min_ma, n = stats(cell[(b, p)])
            parts.append(f"{sr:.2f}/{mean_ma:4.1f}/{min_ma:4.1f}")
        else:
            parts.append(f"{'—':>14}")
    print(f"{b:>6.2f} | " + " | ".join(f"{x:>14}" for x in parts))

print("\n=== dhover 데드라인 곡선 (밴드 s, min_alt mean) ===")
for b in biases:
    line = []
    for p in policies:
        if p.startswith('dhover') and (b, p) in cell:
            sr, mean_ma, min_ma, _ = stats(cell[(b, p)])
            d = p[6:]
            line.append(f"d{d}:surv{sr:.2f} mnA{mean_ma:.1f}")
    print(f"  s{b:.2f}: " + "  ".join(line))

print("\n[판정 힌트]")
print(f"  · dhover10 min_alt가 밴드(1.34~1.40)서 {RL_MINALT}m 근처로 낮으면 → 스크립트도 침하 = hover 물리한계 = 밴드 재정의")
print(f"  · dhover1~3 min_alt 높고(호버고도 유지) survive=1이면 → 빠른전환은 물리적으로 구제됨(밴드 유효), RL이 늦어서 못 살린 것")
