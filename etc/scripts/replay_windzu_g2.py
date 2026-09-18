#!/usr/bin/env python3
"""replay_windzu_g2 — windzu zu_log 를 E(현행) vs G2(08-23) 필터로 재생. 라벨: 셀별 (ws, δ) + 창(바람 60~280, 공격 180~195).
출력: 조건별 gyro/vel NIS 표 + AUC(바람 vs 바람+약공격, vel 보조 기여) + 시계열 그림 (δ × ws, 그림 형식 = 아티팩트).
사용: python3 etc/scripts/replay_windzu_g2.py [results_windzu]"""
import sys, os, csv, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf'); os.chdir('/home/acsl/projects/Issacsim-rhukf')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(fp): fm.fontManager.addfont(fp); matplotlib.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
D = sys.argv[1] if len(sys.argv) > 1 else 'results_windzu'
d = np.load(os.path.join(D, 'zu_log.npz'), allow_pickle=True); data = d['data'].astype(np.float64); dt = float(d['dt'])
calib = load_calibration('calibration/calibration.json')
summ = list(csv.DictReader(open(os.path.join(D, 'sweep_summary.csv'))))
starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
seg = np.full(len(data), -1)
for k in range(len(starts) - 1): seg[starts[k]:starts[k+1]] = k
gps_fresh = np.ones(len(data), bool); gps_fresh[1:] = np.any(np.abs(data[1:, 4:10] - data[:-1, 4:10]) > 1e-12, axis=1); gps_fresh[data[:, 1] > 0.5] = True
# RL 스텝(10Hz) ≈ 5 UKF 스텝: 에피 내 인덱스/5
# ★ 스윕 스텝 번호로 정렬: 에피 내 첫 공격 행 = 스윕 스텝 180 (SWEEP_ATK_START). 공격 없는 셀은 요약의 다른 셀과 같은 오프셋 사용(중앙값).
rl_step = np.zeros(len(data)); _atk = data[:, 2] > 0.5; _offs = []
for k in range(len(starts) - 1):
    a = np.where(_atk[starts[k]:starts[k+1]])[0]
    if len(a): _offs.append(a[0])
_off_med = int(np.median(_offs)) if _offs else 0
for k in range(len(starts) - 1):
    a = np.where(_atk[starts[k]:starts[k+1]])[0]; o = a[0] if len(a) else _off_med
    rl_step[starts[k]:starts[k+1]] = 180 + (np.arange(starts[k+1] - starts[k]) - o) * dt * 10
atkflag = data[:, 2] > 0.5
def cell(k):
    r = summ[k] if k < len(summ) else None
    return (round(float(r['bias']) / 4.36, 2), int(round(float(r['wind_speed']))), r['pattern']) if r else (None, None, None)
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
def auc(x, y):
    o = np.argsort(x); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
    return ((np.arange(1, len(ys)+1)[ys == 1].sum() - n1*(n1+1)/2) / (n1*n0)) if n1*n0 else 0.5
def logit(X, y, it=3000, lr=0.2):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); Xb = np.c_[X, np.ones(len(X))]; w = np.zeros(Xb.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-Xb@w)); w -= lr * Xb.T @ (p - y) / len(y)
    return Xb @ w
CFG = [('E 현행', dict()), ('G2 (Qg2e-3 Rg.2 Qe5e-4 Rv.1)', dict(qg=2e-3, rg=0.2, qe=5e-4, rv=0.1)), ('E+Qg2e-3+Rg0.2', dict(qg=2e-3, rg=0.2))]
cells = [cell(k) for k in range(len(starts) - 1)]
dl = np.array([cells[s][0] if s >= 0 else np.nan for s in seg]); wsv = np.array([cells[s][1] if s >= 0 else -1 for s in seg])
fresh = gps_fresh
lab = np.full(len(data), '', dtype=object)
lab[(rl_step >= 15) & (rl_step < 55) & fresh] = 'clean'                                    # 바람 전
lab[(rl_step >= 70) & (rl_step < 178) & fresh & (wsv > 0)] = 'wind'                         # 바람만
lab[(rl_step >= 70) & (rl_step < 178) & fresh & (wsv == 0)] = 'clean2'                      # 무풍 셀 같은 구간
lab[(rl_step >= 181) & (rl_step < 195) & fresh & atkflag & (wsv > 0)] = 'wind+atk'          # 바람+공격(온셋 1스텝 제외)
lab[(rl_step >= 181) & (rl_step < 195) & fresh & atkflag & (wsv == 0)] = 'atk'
lab[(rl_step >= 196) & (rl_step < 210) & fresh & (dl > 0.05)] = 'post'                      # OFF 직후
res = {}
for nm, kw in CFG:
    nv, ng = replay(**kw); res[nm] = (nv, ng)
    fi = np.where(fresh)[0]; VW = np.zeros(len(nv)); GW = np.zeros(len(nv)); VW[fi] = np.convolve(nv[fi], np.ones(4)/4, 'same'); GW[fi] = np.convolve(ng[fi], np.ones(4)/4, 'same')
    print(f"\n### {nm}")
    print(f"  {'조건':14s} {'n':>5s} | gyro med/p95 | vel med/p95")
    for L in ('clean', 'clean2', 'wind', 'atk', 'wind+atk', 'post'):
        for w in sorted(set(wsv)):
            m = (lab == L) & (wsv == w)
            if L in ('atk', 'wind+atk', 'post'):
                for dd in (0.2, 0.4):
                    mm = m & (np.abs(dl - dd) < 0.01)
                    if mm.sum(): print(f"  {L+f' δ{dd}':14s} {mm.sum():5d} | ws{w}: {np.median(ng[mm]):.2f}/{np.percentile(ng[mm],95):.2f}   | {np.median(nv[mm]):.2f}/{np.percentile(nv[mm],95):.2f}")
            elif m.sum(): print(f"  {L:14s} {m.sum():5d} | ws{w}: {np.median(ng[m]):.2f}/{np.percentile(ng[m],95):.2f}   | {np.median(nv[m]):.2f}/{np.percentile(nv[m],95):.2f}")
    print("  AUC(4창): 바람만 vs 바람+약공격 / clean vs 바람 / clean vs 무풍공격")
    for w in (6, 8):
        for dd in (0.2, 0.4):
            m0 = (lab == 'wind') & (wsv == w); m1 = (lab == 'wind+atk') & (wsv == w) & (np.abs(dl - dd) < 0.01)
            if not (m0.sum() and m1.sum()): continue
            y = np.r_[np.zeros(m0.sum()), np.ones(m1.sum())]
            print(f"    ws{w} δ{dd}: gyro {auc(np.r_[GW[m0],GW[m1]],y):.3f} | vel {auc(np.r_[VW[m0],VW[m1]],y):.3f} | gyro+vel {auc(logit(np.c_[np.r_[GW[m0],GW[m1]],np.r_[VW[m0],VW[m1]]],y),y):.3f}")
    m0 = (lab == 'clean') & (wsv > 0)
    for w in (6, 8):
        m1 = (lab == 'wind') & (wsv == w); y = np.r_[np.zeros(m0.sum()), np.ones(m1.sum())]
        print(f"    clean→wind ws{w}: gyro {auc(np.r_[GW[m0],GW[m1]],y):.3f} | vel {auc(np.r_[VW[m0],VW[m1]],y):.3f}")
    m0 = (lab == 'clean2'); 
    for dd in (0.2, 0.4):
        m1 = (lab == 'atk') & (np.abs(dl - dd) < 0.01)
        if m0.sum() and m1.sum():
            y = np.r_[np.zeros(m0.sum()), np.ones(m1.sum())]; print(f"    무풍 clean→atk δ{dd}: gyro {auc(np.r_[GW[m0],GW[m1]],y):.3f} | vel {auc(np.r_[VW[m0],VW[m1]],y):.3f}")
# 시계열 그림: 행 δ{0.2,0.4}, 열 ws{0,6,8}; E vs G2 겹쳐
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for ri, dd in enumerate((0.2, 0.4)):
    for ci, w in enumerate((0, 6, 8)):
        a = ax[ri][ci]
        for nm, ls in (('E 현행', '-'), ('G2 (Qg2e-3 Rg.2 Qe5e-4 Rv.1)', '--')):
            nv, ng = res[nm]; segs = [k for k, c in enumerate(cells) if c[0] is not None and abs(c[0] - dd) < 0.01 and c[1] == w]
            if not segs: continue
            T = np.arange(0, 300); G = []; V = []
            for k in segs:
                m = (seg == k) & fresh; t = rl_step[m]; g = ng[m]; v = nv[m]
                G.append(np.interp(T, t, g)); V.append(np.interp(T, t, v))
            a.plot(T, np.mean(G, 0), 'r' + ls, lw=1.2, label=f'gyro {nm[:2]}'); a.plot(T, np.mean(V, 0), 'b' + ls, lw=1.2, label=f'vel {nm[:2]}')
        a.axvspan(60, 280, color='c', alpha=.08); a.axvspan(180, 195, color='r', alpha=.12); a.set_title(f'δ{dd} · ws{w}'); a.grid(alpha=.3); a.set_ylim(0, 3.1)
        if ri == 0 and ci == 0: a.legend(fontsize=8)
fig.suptitle('windzu 재생 — E(실선) vs G2(점선): gyro(빨강)·vel(파랑), 바람창 60~280·공격 180~195 (ON 1.5s)'); fig.tight_layout()
out = os.path.join(D, 'windzu_E_vs_G2.png'); fig.savefig(out, dpi=110); print('\n그림:', out)
