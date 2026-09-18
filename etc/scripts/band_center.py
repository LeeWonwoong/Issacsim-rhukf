import csv, numpy as np, sys
from collections import defaultdict
R = list(csv.DictReader(open('results_fineband/sweep_summary.csv')))
cell = defaultdict(list)
for r in R:
    cell[(round(float(r['bias']) / 4.36, 2), r['policy'])].append(float(r['survived']))
ds = sorted({k[0] for k in cell})
# 밴드 = track 생존<0.3 ∧ dhover 생존>0.7 인 δ (바람 합산). 종착 = 그 중 최소(가장 이른 결과성)
band = [d for d in ds if np.mean(cell.get((d, 'track'), [1])) < 0.3
        and np.mean(cell.get((d, 'dhover3'), [0])) > 0.7]
center = band[0] if band else 0.80
print(f"{center:.2f}")   # stdout = 밴드 하단 δ (ramp 종착)
sys.stderr.write("[band] δ별 track/dhover: " +
    " ".join(f"{d}:{np.mean(cell.get((d,'track'),[1])):.1f}/{np.mean(cell.get((d,'dhover3'),[0])):.1f}" for d in ds) +
    f"\n밴드하단={center}\n")
