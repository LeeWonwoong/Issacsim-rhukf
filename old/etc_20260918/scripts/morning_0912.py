#!/usr/bin/env python3
"""morning_0912 — 09-12 밤 결과 집계 → results/claudecodefortest/night/morning_0912.txt
 ① FS2 failsafe 강화 판정 ② 데드라인 곡선 ③ Isaac 학습 v5 4런 (SWIRL/Adam × 벌점 0/−5) ④ surrogate v5 S1/S2"""
import os, csv, json, glob, collections, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); R = 'results/claudecodefortest'; A = 4.36; L = []
def P(s=''): L.append(str(s)); print(s)
def surv(D, keyf):
    try: rows = list(csv.DictReader(open(f'{D}/sweep_summary.csv')))
    except Exception: return {}
    t = collections.defaultdict(lambda: [0, 0])
    for r in rows: k = keyf(r); t[k][1] += 1; t[k][0] += int(float(r['survived']))
    return t
# ① FS2
P("① failsafe 강화(FS2: I 1.5·P 8) 판정: " + ("채택 (FS_PARAMS.env 존재)" if os.path.exists(f'{R}/FS_PARAMS.env') else "기각 → I 0.8 유지"))
for D, tag in ((f'{R}/cert_FS2_wp', 'FS2'),):
    det = collections.defaultdict(list)
    try:
        for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
            if r['policy'] == 'dhover3': det[(round(float(r['bias'])/A, 2), r['pattern'], int(round(float(r['wind_speed']))), int(r['episode']))].append(r)
    except Exception: pass
    agg = collections.defaultdict(list)
    for k, rr in det.items():
        rr.sort(key=lambda r: int(r['step'])); st = np.array([int(r['step']) for r in rr]); x = np.array([float(r['pos_x']) for r in rr]); y = np.array([float(r['pos_y']) for r in rr]); i0 = np.searchsorted(st, 183)
        if i0 >= len(st) - 3: continue
        d = np.hypot(x - x[i0], y - y[i0]); on = (st >= 183) & (st < 210); i3 = min(np.searchsorted(st, 209), len(d) - 1); agg[(k[0], k[1])].append((d[on].max(), d[i3]))
    base = {(0.8, 'waypoint'): (7.2, 5.7), (0.84, 'waypoint'): (6.9, 5.2), (0.8, 'circle'): (3.1, 0.7), (0.84, 'circle'): (4.6, 2.4)}
    for k, v in sorted(agg.items()):
        v = np.array(v); P(f"   δ{k[0]} {k[1]:8s}: 이탈 med {np.median(v[:,0]):.1f} m (기준 {base[k][0]}) · 3 s 오차 {np.median(v[:,1]):.1f} m (기준 {base[k][1]})  n{len(v)}")
# ② 데드라인
P("\n② 데드라인 곡선 (cert_deadline): 전환 지연 d 별 failsafe hover 생존/셀 (δ0.80 | δ0.84, 무풍·ws6 합)")
t = surv(f'{R}/cert_deadline', lambda r: (r['policy'], round(float(r['bias'])/A, 2)))
if t:
    for pol in ('dhover5', 'dhover10', 'dhover15', 'dhover20', 'dhover25'):
        P(f"   {pol:9s} ({int(pol[6:])/10:.1f} s): " + " | ".join(f"δ{d}: {t[(pol,d)][0]}/{t[(pol,d)][1]}" for d in (0.8, 0.84) if (pol, d) in t))
else: P("   (미완)")
# ③ 학습 v5
P("\n③ Isaac 학습 v5 — 에피 구간 평균 reward · F1(late 51+) · fpr · 지연 · 추락수  [pen0 vs pen−5]")
for D in sorted(glob.glob(f'{R}/train_v5_*')):
    fs = glob.glob(f'{D}/metrics_*.csv')
    if not fs: P(f"   {os.path.basename(D):22s}: (metrics 없음) ep={len([l for l in open(f'{D}/train.log', errors='replace') if 'TRAIN Ep' in l]) if os.path.exists(f'{D}/train.log') else 0}"); continue
    rows = list(csv.DictReader(open(fs[0]))); rw = np.array([float(r['reward']) for r in rows]); n = len(rw)
    f1 = [float(r['f1']) for r in rows if int(float(r['tp'])) + int(float(r['fn'])) > 0]; fpr = [float(r['fp_rate']) for r in rows]; dl = [float(r['det_delay']) for r in rows if float(r['det_delay']) >= 0]; cr = sum(int(float(r['crashed'])) for r in rows)
    seg = lambda a, b: f"{np.mean(rw[a:b]):6.0f}" if n > a else "    - "
    late = slice(max(0, n // 2), n)
    P(f"   {os.path.basename(D):22s}: ep {n:3d} | rwd 1-40 {seg(0,40)} 41-100 {seg(40,100)} 101-150 {seg(100,150)} 151-200 {seg(150,200)} | F1 late {np.mean([float(r['f1']) for r in rows[late] if int(float(r['tp']))+int(float(r['fn']))>0] or [0]):.3f} fpr {np.mean(fpr[late]):.3f} 지연 {np.mean([float(r['det_delay']) for r in rows[late] if float(r['det_delay'])>=0] or [-1]):.1f} | 추락 {cr}")
# ④ surrogate v5
P("\n④ surrogate v5 — S1 (α0.1 고정, 63 SWIRL + Adam + SGD, 벌점 0)")
for tag in ('S1', 'S2'):
    res = {}
    for f in glob.glob(f'{R}/night/v5_w*_{tag}.json'): res.update(json.load(open(f)))
    if not res: P(f"   {tag}: (없음/진행 중)"); continue
    P(f"   {tag} ({len(res)} runs)  {'run':36s} {'F1':>6s} {'fpr':>6s} {'delay':>6s} {'rwd':>6s} {'e40':>6s} {'auc100':>7s} {'crash':>5s}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]['F1'])[:15]:
        P(f"      {k:36s} {v['F1']:6.3f} {v['fpr']:6.3f} {v['delay']:6.2f} {v['rwd']:6.0f} {v['e40']:6.0f} {v['auc100']:7.0f} {v['crash']:5d}")
    for nm in ('Adam_s42', 'SGD_s42', 'Adam_pen0_s42', 'Adam_pen-5_s42'):
        if nm in res: v = res[nm]; P(f"      {nm:36s} {v['F1']:6.3f} {v['fpr']:6.3f} {v['delay']:6.2f} {v['rwd']:6.0f} {v['e40']:6.0f} {v['auc100']:7.0f} {v['crash']:5d}")
os.makedirs(f'{R}/night', exist_ok=True); open(f'{R}/night/morning_0912.txt', 'w').write("\n".join(L)); print(f"\n저장: {R}/night/morning_0912.txt")
