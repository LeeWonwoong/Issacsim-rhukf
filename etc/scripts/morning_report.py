#!/usr/bin/env python3
"""morning_report — 야간 Isaac 체인(holdcheck · dh1check · slowcap) 집계 → results/claudecodefortest/night/morning_report.txt
  ① 즉각 호버(dhover1) vs dhover3, ws5 vs ws6 (dh1check)   ② hold 5 s 밴드 확장 (holdcheck)
  ③ 궤적 속도 −15% (slowcap): 밴드·즉사형·aliasing·추종 RMSE 를 속도 1.0 과 비교"""
import os, csv, re, collections, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); R = 'results/claudecodefortest'; A = 4.36
L = []
def P(s=''): L.append(str(s)); print(s)
def load(d):
    try: return list(csv.DictReader(open(f'{d}/sweep_summary.csv')))
    except Exception: return []
def surv(rows, dl, w, pol):
    r = [x for x in rows if round(float(x['bias'])/A, 2) == dl and int(round(float(x['wind_speed']))) == w and x['policy'] == pol]
    return (sum(int(float(x['survived'])) for x in r), len(r), [(x['pattern'][:3], (int(float(x['crash_step']))-180)/10) for x in r if int(float(x['survived'])) == 0])
# ① dh1check
rows = load(f'{R}/dh1check')
P("① dh1check — arm 0.05 · ON 3 s · 생존/셀(10) [사망 패턴, 온셋 후 s]")
P(f"{'δ':>5s} {'ws':>3s} {'track':>7s} {'dhover1':>9s} {'dhover3':>9s}")
for dl in (0.76, 0.78, 0.8):
    for w in (5, 6):
        t = surv(rows, dl, w, 'track'); h1 = surv(rows, dl, w, 'dhover1'); h3 = surv(rows, dl, w, 'dhover3')
        P(f"{dl:5} {w:3d} {t[0]:3d}/{t[1]:<3d} {h1[0]:5d}/{h1[1]:<3d} {h3[0]:5d}/{h3[1]:<3d}   dh1 사망 {h1[2]}  dh3 사망 {h3[2]}")
# ② holdcheck
rows = load(f'{R}/holdcheck')
P("\n② holdcheck — arm 0.05 · hold 5 s · 생존/셀(15)")
for dl in (0.74, 0.76, 0.78):
    for w in (0, 6):
        t = surv(rows, dl, w, 'track'); h3 = surv(rows, dl, w, 'dhover3')
        P(f"  δ{dl} ws{w}: track {t[0]}/{t[1]}  dhover3 {h3[0]}/{h3[1]}  track 사망 {collections.Counter(p for p, _ in t[2])}  dh3 사망 {h3[2]}")
# ③ slowcap vs 속도 1.0
slow = load(f'{R}/slowcap'); slow30 = load(f'{R}/slowcap30'); base0 = load('results_bandcap'); base6 = load(f'{R}/bandcap_arm005')
P("\n③ slowcap (TRAJ_SCALE 0.85 / 0.70) vs 속도 1.0 — track 생존/10 (dhover3 생존/10)")
P(f"{'δ':>5s} | {'v1.0 ws0':>10s} {'v0.85 ws0':>10s} {'v0.70 ws0':>10s} | {'v1.0 ws6':>10s} {'v0.85 ws6':>10s} {'v0.70 ws6':>10s}")
for dl in (0.25, 0.74, 0.76, 0.78, 0.8):
    c = []
    for w, base in ((0, base0), (6, base6)):
        for rws in (base, slow, slow30):
            t = surv(rws, dl, w, 'track'); h = surv(rws, dl, w, 'dhover3'); c.append(f"{t[0]:2d} ({h[0]:2d})")
    P(f"{dl:5} | {c[0]:>10s} {c[1]:>10s} {c[2]:>10s} | {c[3]:>10s} {c[4]:>10s} {c[5]:>10s}")
# 추종 RMSE (δ0 track 셀) 속도 1.0 vs 0.85
def rmse(d, w):
    out = collections.defaultdict(list)
    try:
        for r in csv.DictReader(open(f'{d}/sweep_detail.csv')):
            if r['policy'] != 'track' or float(r['bias']) != 0 or int(round(float(r['wind_speed']))) != w or int(r['step']) < 30: continue
            if 'ref_x' not in r or r['ref_x'] in ('', 'nan'): continue
            out[r['pattern']].append((float(r['pos_x'])-float(r['ref_x']))**2 + (float(r['pos_y'])-float(r['ref_y']))**2)
    except Exception: pass
    return {p: np.sqrt(np.mean(v)) for p, v in out.items()}
P("\n   추종 XY RMSE (δ0, track):  패턴  v1.0 ws0 / v0.85 ws0 | v1.0 ws6 / v0.85 ws6")
b0 = rmse('results_bandcap', 0); s0 = rmse(f'{R}/slowcap', 0); t0 = rmse(f'{R}/slowcap30', 0); b6 = rmse('results_bandcap', 6); s6 = rmse(f'{R}/slowcap', 6); t6 = rmse(f'{R}/slowcap30', 6)
for p in ('waypoint', 'circle', 'figure8', 'aggressive', 'scurve'):
    P(f"   {p:10s} {b0.get(p, float('nan')):.2f} / {s0.get(p, float('nan')):.2f} / {t0.get(p, float('nan')):.2f} | {b6.get(p, float('nan')):.2f} / {s6.get(p, float('nan')):.2f} / {t6.get(p, float('nan')):.2f}")
# aliasing/NIS: slowcap analysis summary 의 E/G2 핵심 행
for tag in ("slowcap", "slowcap30"):
    P(f"\n   {tag} E/G2 (analysis/summary.txt 발췌):")
    sp = f'{R}/{tag}/analysis/summary.txt'
    if os.path.exists(sp):
        for line in open(sp, errors='replace'):
            if re.search(r'^### |무풍 평시|aggressive|waypoint|ws6 바람|δ0\.25 ws|δ0\.8 ws', line): P('   ' + line.rstrip()[:170])
    else: P('   (분석 미완)')
os.makedirs(f'{R}/night', exist_ok=True); open(f'{R}/night/morning_report.txt', 'w').write("\n".join(L)); print(f"\n저장: {R}/night/morning_report.txt")
