#!/usr/bin/env python3
"""agg_p0stage3 — P₀ 를 3번(online_stop) 팔 B 무대에서 직접 잰 스캔 집계.
   창을 명시해서 낸다. agg_stop.py 가 표(후반100)와 짝 차이(전구간)를 다른 창으로
   계산해 부호가 뒤집혀 보였던 사고를 반복하지 않기 위해서다."""
import json, glob, math, os, sys
import numpy as np

D = 'results/claudecodefortest/p0stage3'
P0S = ['0.003', '0.01', '0.03', '0.10']
SEEDS = [42, 43, 44]
WINS = [('전구간 0-299', 0, 300), ('0-99', 0, 100), ('100-199', 100, 200), ('200-299', 200, 300)]

H = {}
missing = []
for p in P0S:
    for sd in SEEDS:
        f = f'{D}/p{p}_s{sd}.json'
        if not os.path.exists(f):
            missing.append(f); continue
        for nm, m in json.load(open(f)).items():
            H.setdefault(p, {}).setdefault(nm, {})[sd] = [x['reward'] for x in m['hist']]
if missing:
    print(f'⚠ 누락 {len(missing)}개: ' + ', '.join(os.path.basename(x) for x in missing))
if not H:
    print('결과 없음'); sys.exit()

# ── 무결성: Adam1e-3 은 P₀ 와 무관하므로 P₀ 간 완전히 동일해야 한다 ──
print('\n=== 무결성 검사: Adam1e-3 은 P₀ 에 영향받지 않아야 한다 ===')
base = None
for p in P0S:
    if p not in H or 'Adam1e-3' not in H[p]: continue
    v = {sd: round(float(np.mean(r)), 6) for sd, r in H[p]['Adam1e-3'].items()}
    if base is None:
        base = v; print(f'  P₀={p:<6} 기준  {v}')
    else:
        same = (v == base)
        print(f'  P₀={p:<6} {"일치 ✓" if same else "★불일치 — 격리 실패"}  {v}')

# ── P₀ 별 SWIRL vs Adam1e-3 짝 차이 ──
for wn, lo, hi in WINS:
    print(f'\n=== 창 {wn} : SWIRL − Adam1e-3 (짝, 같은 시드) ===')
    print(f'  {"P₀":<8}{"SWIRL":>9}{"Adam1e-3":>11}{"Δ":>9}{"t":>8}{"승":>6}')
    for p in P0S:
        if p not in H or 'SWIRL' not in H[p] or 'Adam1e-3' not in H[p]: continue
        sds = [s for s in SEEDS if s in H[p]['SWIRL'] and s in H[p]['Adam1e-3']]
        if len(sds) < 2: continue
        sw = [float(np.mean(H[p]['SWIRL'][s][lo:hi])) for s in sds]
        ad = [float(np.mean(H[p]['Adam1e-3'][s][lo:hi])) for s in sds]
        d = [a - b for a, b in zip(sw, ad)]
        m, s_ = float(np.mean(d)), float(np.std(d, ddof=1))
        t = m / (s_ / math.sqrt(len(d))) if s_ > 0 else float('nan')
        w = sum(1 for x in d if x > 0)
        star = ' ★' if m == max(float(np.mean([
            float(np.mean(H[q]['SWIRL'][s][lo:hi])) - float(np.mean(H[q]['Adam1e-3'][s][lo:hi]))
            for s in sds])) for q in P0S if q in H and 'SWIRL' in H[q]) else ''
        print(f'  {p:<8}{np.mean(sw):9.2f}{np.mean(ad):11.2f}{m:+9.3f}{t:+8.2f}{w:>4}/{len(d)}{star}')
print()
