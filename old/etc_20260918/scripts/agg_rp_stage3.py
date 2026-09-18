#!/usr/bin/env python3
"""agg_rp_stage3 — 교정된 3번 무대(v5e·EP_RNG=1)에서 P₀ 저단과 R 축을 함께 읽는다.
   P₀ 스캔(p0stage3)과 R 스캔(rp_stage3)을 한 표로 모아 본선 하이퍼를 고른다.
   창을 반드시 명시한다 — agg_stop.py 가 표(후반100)와 짝차(전구간)를 다른 창으로 계산해
   부호가 뒤집혀 보였던 사고를 반복하지 않기 위해서다."""
import json, glob, math, os, re
import numpy as np

WINS = [('전구간 0-299', 0, 300), ('0-99', 0, 100), ('100-199', 100, 200), ('200-299', 200, 300)]
SEEDS = [42, 43, 44]

def load(pat, label_fn):
    """pat 에 맞는 json 들을 {라벨: {학습기: {시드: [보상]}}} 으로."""
    H = {}
    for f in sorted(glob.glob(pat)):
        lab = label_fn(os.path.basename(f))
        if lab is None: continue
        sd = int(re.search(r'_s(\d+)\.json$', f).group(1))
        for nm, m in json.load(open(f)).items():
            H.setdefault(lab, {}).setdefault(nm, {})[sd] = [x['reward'] for x in m['hist']]
    return H

# P₀ 스캔(R=2 고정) + R 스캔(P₀=0.01 고정)
H = {}
H.update(load('results/claudecodefortest/p0stage3/p*_s*.json',
              lambda b: f"P{re.match(r'p([0-9.]+)_s', b).group(1)}·R2"))
H.update(load('results/claudecodefortest/rp_stage3/*_s*.json',
              lambda b: (lambda m: f"P{m.group(1)}·R{m.group(2)}" if m else None)(
                  re.match(r'p([0-9.]+)_r([0-9.]+)_s', b))))
if not H:
    print('결과 없음'); raise SystemExit

def keyf(lab):
    m = re.match(r'P([0-9.]+)·R([0-9.]+)', lab)
    return (float(m.group(2)), -float(m.group(1)))
labs = sorted(H, key=keyf)

print('\n=== 무결성: Adam1e-3 은 SWIRL 하이퍼에 영향받지 않아야 한다 ===')
base = None
for lab in labs:
    if 'Adam1e-3' not in H[lab]: continue
    v = {sd: round(float(np.mean(r)), 6) for sd, r in sorted(H[lab]['Adam1e-3'].items())}
    if base is None: base = v; print(f'  {lab:<14} 기준  {v}')
    else: print(f'  {lab:<14} {"일치 ✓" if v == base else "★불일치 — 격리 실패"}  {v}')

for wn, lo, hi in WINS:
    print(f'\n=== 창 {wn} : SWIRL − Adam1e-3 (짝, 같은 시드·같은 시나리오) ===')
    print(f'  {"조합":<14}{"SWIRL":>9}{"Adam":>9}{"Δ":>9}{"t":>8}{"승":>7}')
    rows = []
    for lab in labs:
        if 'SWIRL' not in H[lab] or 'Adam1e-3' not in H[lab]: continue
        sds = [s for s in SEEDS if s in H[lab]['SWIRL'] and s in H[lab]['Adam1e-3']]
        if len(sds) < 2: continue
        sw = [float(np.mean(H[lab]['SWIRL'][s][lo:hi])) for s in sds]
        ad = [float(np.mean(H[lab]['Adam1e-3'][s][lo:hi])) for s in sds]
        d = [a - b for a, b in zip(sw, ad)]
        m, sd_ = float(np.mean(d)), float(np.std(d, ddof=1))
        t = m / (sd_ / math.sqrt(len(d))) if sd_ > 0 else float('nan')
        rows.append((lab, np.mean(sw), np.mean(ad), m, t, sum(1 for x in d if x > 0), len(d)))
    best = max((r[3] for r in rows), default=None)
    for lab, s, a, m, t, w, n in rows:
        print(f'  {lab:<14}{s:9.2f}{a:9.2f}{m:+9.3f}{t:+8.2f}{w:>4}/{n}{"  ★" if m == best else ""}')
print()
