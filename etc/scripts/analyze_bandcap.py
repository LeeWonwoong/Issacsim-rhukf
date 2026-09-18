#!/usr/bin/env python3
"""analyze_bandcap — results_bandcap (FF on, ws{0,6} × δ{0,0.25,0.74~0.82} × {track,dhover3}) 종합.
  ① 생존표 (밴드)  ② 패턴별 time-step NIS 그림 (E vs G2, gyro/vel, track)  ③ 패턴별 3D 궤적 그림 (ref vs 실제, 정책별)
  ④ E vs G2 통계: 평시/바람/공격/OFF 의 gyro·vel, 온셋 지연, 평시 초과 run(오탐 proxy), vel 의 바람 AUC
사용: python3 etc/scripts/analyze_bandcap.py [results_dir] [ON스텝=30]   → <dir>/analysis/"""
import sys, os, csv, collections, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name(); break
matplotlib.rcParams['axes.unicode_minus'] = False
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled

D = sys.argv[1] if len(sys.argv) > 1 else 'results_bandcap'
OUT = os.path.join(D, 'analysis'); os.makedirs(OUT, exist_ok=True)
A = 4.36; PATS = ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve']
ON = int(sys.argv[2]) if len(sys.argv) > 2 else 30   # 공격 ON 스텝 (bandcap 30 / bandcap_on5 5 / bandcap_on15 15)
ATK0, ATK1 = 180, 180 + ON
CFG = {'E': dict(), 'G2': dict(qg=2e-3, rg=0.2, qe=5e-4, rv=0.1)}
LOG = []
def P(s=''):
    LOG.append(str(s)); print(s, flush=True)

# ── 요약/상세 ──
summ = list(csv.DictReader(open(os.path.join(D, 'sweep_summary.csv'))))
for r in summ: r['delta'] = round(float(r['bias']) / A, 2); r['ws'] = int(round(float(r['wind_speed'])))
det = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, 'sweep_detail.csv'))):
    det[(int(r['cell_idx']), int(r['episode']))].append(r)
for k in det: det[k].sort(key=lambda r: int(r['step']))
deltas = sorted(set(r['delta'] for r in summ)); wss = sorted(set(r['ws'] for r in summ))

# ① 생존표
P("① 생존표 (track 생존 / dhover3 생존, 셀=5패턴×에피)")
P(f"{'δ':>5s} " + " ".join(f"{'ws'+str(w)+' track':>12s} {'dhover3':>8s}" for w in wss) + "   패턴별 track 생존 (ws0 | ws6)")
for d in deltas:
    row = f"{d:5}"
    for w in wss:
        for pol in ('track', 'dhover3'):
            rs = [r for r in summ if r['delta'] == d and r['ws'] == w and r['policy'] == pol]
            row += f" {sum(int(float(r['survived'])) for r in rs):3d}/{len(rs):<2d}   " if pol == 'track' else f" {sum(int(float(r['survived'])) for r in rs):3d}/{len(rs):<2d}  "
    pp = []
    for w in wss:
        pp.append(" ".join(f"{p[:3]}{sum(int(float(r['survived'])) for r in summ if r['delta']==d and r['ws']==w and r['policy']=='track' and r['pattern']==p)}" for p in PATS))
    P(row + "   " + " | ".join(pp))
lag = collections.defaultdict(list)
for r in summ:
    if r['policy'] == 'track' and int(float(r['survived'])) == 0: lag[(r['delta'], r['ws'])].append((int(float(r['crash_step'])) - ATK0) / 10)
P("track 추락 lag(s) med/min/max: " + ", ".join(f"δ{k[0]} ws{k[1]}: {np.median(v):.1f}/{min(v):.1f}/{max(v):.1f} (n{len(v)})" for k, v in sorted(lag.items())))

# ── zu 재생 (E, G2) ──
zpath = os.path.join(D, 'zu_log.npz')
NIS = {}
if os.path.exists(zpath):
    d = np.load(zpath, allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
    calib = load_calibration('calibration/calibration.json')
    starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    nseg = len(starts) - 1
    gps_fresh = np.ones(len(data), bool); gps_fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); gps_fresh[data[:, 1] > 0.5] = True
    atkf = data[:, 2] > 0.5
    # 세그먼트 → 요약 행 (실행 순서). 스텝 정렬: 첫 공격 행 = ATK0 (무공격 셀은 중앙값 오프셋)
    offs = []
    for k in range(nseg):
        a = np.where(atkf[starts[k]:starts[k+1]])[0]
        if len(a): offs.append(a[0])
    off_med = int(np.median(offs)) if offs else 0
    rl = np.zeros(len(data))
    for k in range(nseg):
        a = np.where(atkf[starts[k]:starts[k+1]])[0]; o = a[0] if len(a) else off_med
        rl[starts[k]:starts[k+1]] = ATK0 + (np.arange(starts[k+1] - starts[k]) - o) * dt * 10
    def replay(qg=None, rg=None, qe=None, rv=None):
        ukf = DynamicsUKF(dt=dt, calib=calib)
        for sl, v in ((slice(3, 6), qe), (slice(9, 12), qg)):
            if v is not None:
                for i in range(sl.start, sl.stop): ukf.Q[i, i] = v
        for sl, v in ((slice(3, 6), rv), (slice(6, 9), rg)):
            if v is not None:
                for i in range(sl.start, sl.stop): ukf.R[i, i] = v
        nv = np.zeros(len(data)); ng = np.zeros(len(data))
        for k, row in enumerate(data):
            z = row[4:13].copy(); u = row[13:17].copy()
            if row[1] > 0.5:
                ukf.x = np.zeros(12); ukf.x[0:3] = z[0:3]; ukf.x[3:6] = row[17:20]; ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]
                ukf.P = np.eye(12) * 0.1; ukf.is_ukf_initialized = True
            res, Pzz = ukf.step(z, u, gps_fresh=bool(gps_fresh[k]))
            _, nv[k] = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3); _, ng[k] = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3)
        return nv, ng
    for nm, kw in CFG.items():
        P(f"[replay {nm}] rows={len(data)} segs={nseg} (summary {len(summ)})"); NIS[nm] = replay(**kw)
    fresh = gps_fresh
    def seg_series(k, nm, T=np.arange(0, 300)):
        m = (np.arange(len(data)) >= starts[k]) & (np.arange(len(data)) < starts[k+1]) & fresh
        t = rl[m]; nv, ng = NIS[nm]
        if len(t) < 10: return None, None
        return np.interp(T, t, ng[m], left=np.nan, right=np.nan), np.interp(T, t, nv[m], left=np.nan, right=np.nan)
    segs_of = collections.defaultdict(list)   # (delta, ws, pattern, policy) → seg idx
    for k in range(min(nseg, len(summ))):
        r = summ[k]; segs_of[(r['delta'], r['ws'], r['pattern'], r['policy'])].append(k)

    # ② NIS 그림 (track) — 패턴별: 행 δ, 열 ws
    T = np.arange(0, 300)
    for p in PATS:
        fig, ax = plt.subplots(len(deltas), len(wss), figsize=(6.5 * len(wss), 2.6 * len(deltas)), sharex=True, squeeze=False)
        for ri, dl in enumerate(deltas):
            for ci, w in enumerate(wss):
                a = ax[ri][ci]; ks = segs_of[(dl, w, p, 'track')]
                for nm, ls in (('E', '-'), ('G2', '--')):
                    G = []; V = []
                    for k in ks:
                        g, v = seg_series(k, nm, T)
                        if g is not None: G.append(g); V.append(v)
                    if G:
                        a.plot(T, np.nanmean(G, 0), 'r' + ls, lw=1.1, label=f'gyro {nm}'); a.plot(T, np.nanmean(V, 0), 'b' + ls, lw=1.1, label=f'vel {nm}')
                if w > 0: a.axvspan(60, 290, color='c', alpha=.07)
                if dl > 0: a.axvspan(ATK0, ATK1, color='r', alpha=.12)
                # 추락 표시
                for k in ks:
                    r = summ[k]
                    if int(float(r['survived'])) == 0: a.axvline(int(float(r['crash_step'])), color='k', lw=0.8, ls=':')
                a.set_ylim(0, 3.1); a.grid(alpha=.3); a.set_title(f'{p} · δ{dl} · ws{w} (track, n={len(ks)})', fontsize=9)
                if ri == 0 and ci == 0: a.legend(fontsize=7, ncol=2)
        fig.suptitle(f'{p}: time-step NIS — E(실선) vs G2(점선), gyro 빨강·vel 파랑, 공격 {ATK0}~{ATK1}(3s), 바람창 60~290, 점선 세로=추락')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f'nis_{p}.png'), dpi=100); plt.close(fig)
    P(f"② NIS 그림 저장: {OUT}/nis_<pattern>.png")

    # ④ 통계 E vs G2
    lab = np.full(len(data), '', dtype=object)
    dl_row = np.array([summ[s]['delta'] if s < len(summ) else -1 for s in np.repeat(np.arange(nseg), np.diff(starts))])
    ws_row = np.array([summ[s]['ws'] if s < len(summ) else -1 for s in np.repeat(np.arange(nseg), np.diff(starts))])
    pol_row = np.array([summ[s]['policy'] if s < len(summ) else '' for s in np.repeat(np.arange(nseg), np.diff(starts))])
    pat_row = np.array([summ[s]['pattern'] if s < len(summ) else '' for s in np.repeat(np.arange(nseg), np.diff(starts))])
    trk = pol_row == 'track'
    lab[(rl >= 15) & (rl < 55) & fresh & trk] = 'clean'
    lab[(rl >= 70) & (rl < 178) & fresh & trk & (ws_row > 0)] = 'wind'
    lab[(rl >= 70) & (rl < 178) & fresh & trk & (ws_row == 0)] = 'clean2'
    lab[(rl >= ATK0 + 1) & (rl < ATK1) & fresh & trk & atkf] = 'atk'
    lab[(rl >= ATK1 + 1) & (rl < ATK1 + 15) & fresh & trk & (dl_row > 0)] = 'post'
    def auc(x, y):
        o = np.argsort(x); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
        return ((np.arange(1, len(ys)+1)[ys == 1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1*n0 else float('nan')
    def runs(x, th):
        L = []; c = 0
        for v in x:
            if v > th: c += 1
            else:
                if c: L.append(c); c = 0
        if c: L.append(c)
        return np.array(L) if L else np.array([], int)
    P("\n④ E vs G2 통계 (track, GPS-fresh 스텝)")
    for nm in CFG:
        nv, ng = NIS[nm]
        fi = np.where(fresh)[0]; GW = np.zeros(len(ng)); VW = np.zeros(len(nv)); GW[fi] = np.convolve(ng[fi], np.ones(4)/4, 'same'); VW[fi] = np.convolve(nv[fi], np.ones(4)/4, 'same')
        P(f"\n### {nm}")
        m = (lab == 'clean2') | (lab == 'clean'); g95 = np.percentile(ng[m], 95)
        P(f"  무풍 평시 gyro med/p95/p99 {np.median(ng[m]):.2f}/{g95:.2f}/{np.percentile(ng[m],99):.2f}  vel {np.median(nv[m]):.2f}/{np.percentile(nv[m],95):.2f}")
        for p in PATS:
            mm = m & (pat_row == p)
            if mm.sum() < 20: continue
            L = runs(ng[mm], g95)
            P(f"    {p:10s} 평시 gyro p95 {np.percentile(ng[mm],95):.2f} max {ng[mm].max():.2f} | p95(전체) 초과 run ≥3/≥5 per100: {np.sum(L>=3)/mm.sum()*100:.1f}/{np.sum(L>=5)/mm.sum()*100:.1f}")
        mw = lab == 'wind'
        if mw.sum() > 20:
            L = runs(ng[mw], g95); y = np.r_[np.zeros(m.sum()), np.ones(mw.sum())]
            P(f"  ws6 바람 gyro med/p95/p99 {np.median(ng[mw]):.2f}/{np.percentile(ng[mw],95):.2f}/{np.percentile(ng[mw],99):.2f}  vel {np.median(nv[mw]):.2f}/{np.percentile(nv[mw],95):.2f} | 초과 run ≥3/≥5 per100: {np.sum(L>=3)/mw.sum()*100:.1f}/{np.sum(L>=5)/mw.sum()*100:.1f} | clean→wind AUC gyro {auc(np.r_[GW[m],GW[mw]],y):.3f} vel {auc(np.r_[VW[m],VW[mw]],y):.3f}")
        for dl in [x for x in deltas if x > 0]:
            for w in wss:
                ma = (lab == 'atk') & (dl_row == dl) & (ws_row == w); mp = (lab == 'post') & (dl_row == dl) & (ws_row == w)
                if ma.sum() < 5 or mp.sum() < 5: continue
                # 온셋 지연: 세그먼트별 첫 gyro>g95 스텝 − 180
                dls = []
                for k in segs_of.get((dl, w), []) if False else [k for k in range(min(nseg, len(summ))) if summ[k]['delta'] == dl and summ[k]['ws'] == w and summ[k]['policy'] == 'track']:
                    g, v = seg_series(k, nm, np.arange(ATK0, ATK0 + 15))
                    if g is None: continue
                    idx = np.where(g > g95)[0]; dls.append(idx[0] if len(idx) else 15)
                base = (lab == 'wind') & (ws_row == w) if w > 0 else m
                if base.sum() < 5: continue
                y = np.r_[np.zeros(base.sum()), np.ones(ma.sum())]
                P(f"  δ{dl} ws{w}: 공격중 gyro med/min {np.median(ng[ma]):.2f}/{np.percentile(ng[ma],5):.2f} vel {np.median(nv[ma]):.2f} | OFF후 gyro {np.median(ng[mp]):.2f} vel {np.median(nv[mp]):.2f} | 온셋지연(>평시p95) med {np.median(dls) if dls else -1:.0f} | AUC(4창) vs {'바람' if w else '평시'}: gyro {auc(np.r_[GW[base],GW[ma]],y):.3f} vel {auc(np.r_[VW[base],VW[ma]],y):.3f}")
else:
    P("(zu_log.npz 없음 — NIS/통계 생략)")

# ③ 3D 궤적 (detail csv) — 패턴별: 행 δ, 열 ws×policy
for p in PATS:
    cols = [(w, pol) for w in wss for pol in ('track', 'dhover3')]
    fig = plt.figure(figsize=(4.2 * len(cols), 3.6 * len(deltas)))
    for ri, dl in enumerate(deltas):
        for ci, (w, pol) in enumerate(cols):
            a = fig.add_subplot(len(deltas), len(cols), ri * len(cols) + ci + 1, projection='3d')
            cells = [((int(r['cell_idx']), int(r['episode'])), int(float(r['survived']))) for r in summ if r['delta'] == dl and r['ws'] == w and r['policy'] == pol and r['pattern'] == p]
            first = True
            for ck, surv in cells:
                rs = [r for r in det.get(ck, []) if int(r['step']) >= 20]
                if not rs: continue
                x = np.array([float(r['pos_x']) for r in rs]); y = np.array([float(r['pos_y']) for r in rs]); z = np.array([float(r['alt']) for r in rs])
                rx = np.array([float(r.get('ref_x', 'nan')) for r in rs]); ry = np.array([float(r.get('ref_y', 'nan')) for r in rs]); rz = np.array([float(r.get('ref_alt', 'nan')) for r in rs])
                st = np.array([int(r['step']) for r in rs]); act = np.array([int(float(r['attack_active'])) for r in rs])
                if first: a.plot(rx, ry, rz, 'k--', lw=0.8, label='ref'); first = False
                on = (act == 1) if dl > 0 else np.zeros(len(st), bool); pre = (st < ATK0) & ~on; post = (st >= ATK0) & ~on
                a.plot(x[pre], y[pre], z[pre], color='gray', lw=0.9); a.plot(x[on], y[on], z[on], color='r', lw=1.4); a.plot(x[post], y[post], z[post], color='orange', lw=0.9)
                if surv == 0: a.scatter([x[-1]], [y[-1]], [z[-1]], marker='x', color='k', s=40)
            a.set_title(f'{p} δ{dl} ws{w} {pol}', fontsize=8); a.set_xlabel('x'); a.set_ylabel('y'); a.set_zlabel('alt'); a.tick_params(labelsize=6)
    fig.suptitle(f'{p}: 3D 궤적 — ref(검정 점선) / 공격 전(회색) / 공격 중(빨강) / 후(주황) / ×=추락', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'traj3d_{p}.png'), dpi=90); plt.close(fig)
P(f"③ 3D 그림 저장: {OUT}/traj3d_<pattern>.png")
open(os.path.join(OUT, 'summary.txt'), 'w').write("\n".join(LOG))
P(f"요약 저장: {OUT}/summary.txt")
