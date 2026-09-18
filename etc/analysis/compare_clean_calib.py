"""clean(§6-2) vs cal-error 5% — 기동에서 gyro가 튀게 됐나 + 공격 분리도 비교."""
import csv, os
import numpy as np
from collections import defaultdict

def comp(x):
    return np.minimum(np.log1p(np.sqrt(np.maximum(x, 0.0))), 3.0)
def winmax(v, w=4):
    return np.array([v[max(0, i-w+1):i+1].max() for i in range(len(v))])
def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 5 or b.size < 5:
        return float('nan')
    return abs(a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-9)

def stats(prefix, ws):
    p = f'{prefix}_ws{ws}/sweep_detail.csv'
    if not os.path.exists(p):
        return None
    seq = defaultdict(list)
    for r in csv.DictReader(open(p)):
        try:
            seq[r['episode']].append((int(r['step']), float(r['nis_g_raw']), float(r['nis_v_raw']), int(r['attack_active'])))
        except (ValueError, KeyError):
            pass
    ben_g, atk_g, benw, atkw = [], [], [], []
    for k, s in seq.items():
        s.sort(key=lambda x: x[0])
        ng = comp(np.array([x[1] for x in s])); ngw = winmax(ng)
        atk = np.array([x[3] for x in s])
        graw = np.array([x[1] for x in s])
        for i in range(len(s)):
            if atk[i] == 1:
                atk_g.append(ng[i]); atkw.append(ngw[i])
            else:
                ben_g.append(ng[i]); benw.append(ngw[i])
    ben_raw = [x[1] for k, s in seq.items() for x in s if x[3] == 0]
    return dict(
        ben_graw_p90=np.percentile(ben_raw, 90) if ben_raw else float('nan'),
        ben_graw_p99=np.percentile(ben_raw, 99) if ben_raw else float('nan'),
        d_single=dprime(atk_g, ben_g), d_win=dprime(atkw, benw))

print("="*80)
print(" clean(§6-2) vs cal-error 5% — gyro 기동 aliasing & 공격 분리도")
print("="*80)
print(f"{'조건':>22s} {'ws':>3s} | {'평시 gyroNIS p90':>15s} {'p99':>7s} | {'공격 d′ 단일/윈':>16s}")
print("-"*80)
for ws in [0, 9]:
    for label, pre in [('clean(§6-2)', 'results_62'), ('cal-error 5%', 'results_calib5')]:
        s = stats(pre, ws)
        if s:
            print(f"{label:>22s} {ws:3d} | {s['ben_graw_p90']:15.2f} {s['ben_graw_p99']:7.2f} | {s['d_single']:6.2f}/{s['d_win']:6.2f}")
        else:
            print(f"{label:>22s} {ws:3d} | (없음)")
print("\n판정: cal-error서 평시 gyroNIS p90/p99가 올라가면 = 기동 aliasing 생김(원하는 것).")
print("  공격 d′ 단일이 내려가고(모호↑) 윈도우가 여전히 잡으면(윈>>단일) = 원웅님 목표 프레임워크.")
