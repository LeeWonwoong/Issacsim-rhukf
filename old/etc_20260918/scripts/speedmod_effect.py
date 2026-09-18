#!/usr/bin/env python3
"""speedmod_effect — 속도변조(가감속)가 (a) 토크 권한 사용 |τ|/4.36 (b) gyro NIS 에 주는 기여를 기존 로그에서 측정.
   sweep_detail(10Hz): ref_x/ref_y → 기준 접선가속 a_t 로 스텝을 가속/감속/등속 구간으로 나눔 → 구간별 nis_g, u_norm, omega_norm
   zu_log(50Hz): u1_tx,u2_ty (N·m) → 권한 사용률 분포 (패턴별, δ0 track)  + 추락 직전 권한 사용
사용: python3 etc/scripts/speedmod_effect.py <dir> ..."""
import sys, csv, collections, numpy as np
A = 4.36
for D in sys.argv[1:]:
    print(f"\n=== {D}")
    det = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{D}/sweep_detail.csv')):
        if r['policy'] != 'track' or r.get('ref_x', '') in ('', 'nan'): continue
        det[(int(r['cell_idx']), int(r['episode']))].append(r)
    byp = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, rows in det.items():
        rows.sort(key=lambda r: int(r['step'])); b = float(rows[0]['bias']); ws = int(round(float(rows[0]['wind_speed']))); p = rows[0]['pattern']
        if b != 0: continue
        rx = np.array([float(r['ref_x']) for r in rows]); ry = np.array([float(r['ref_y']) for r in rows])
        v = np.hypot(np.gradient(rx), np.gradient(ry)) * 10; at = np.gradient(v) * 10   # m/s, m/s²
        g = np.array([float(r['nis_g_scaled']) for r in rows]); un = np.array([float(r['u_norm']) for r in rows]); om = np.array([float(r['omega_norm']) for r in rows])
        st = np.array([int(r['step']) for r in rows]); m = st >= 30
        for lab, sel in (('가속 a_t>+0.6', at > 0.6), ('감속 a_t<-0.6', at < -0.6), ('등속 |a_t|<0.2', np.abs(at) < 0.2)):
            s = sel & m
            if s.sum() < 5: continue
            byp[(p, ws)][lab + ' g'].extend(g[s]); byp[(p, ws)][lab + ' u'].extend(un[s]); byp[(p, ws)][lab + ' w'].extend(om[s]); byp[(p, ws)][lab + ' v'].extend(v[s])
    print("① 기준궤적 접선가속 구간별 (δ0 track, step≥30): gyro NIS(E) p50/p95 · u_norm p95 · ω p95 · 속도 평균")
    for (p, ws), dd in sorted(byp.items()):
        line = f"  {p:10s} ws{ws}: "
        for lab in ('가속 a_t>+0.6', '감속 a_t<-0.6', '등속 |a_t|<0.2'):
            g = dd.get(lab + ' g', []); u = dd.get(lab + ' u', []); w = dd.get(lab + ' w', []); vv = dd.get(lab + ' v', [])
            if g: line += f"| {lab}: g {np.percentile(g,50):.2f}/{np.percentile(g,95):.2f} u {np.percentile(u,95):.2f} ω {np.percentile(w,95):.2f} v {np.mean(vv):.2f} (n{len(g)}) "
        print(line)
    # zu: 권한 사용률
    try:
        d = np.load(f'{D}/zu_log.npz', allow_pickle=True); data = d['data'].astype(float); cols = str(d['cols']).split(',')
    except Exception as e: print('zu 없음', e); continue
    summ = list(csv.DictReader(open(f'{D}/sweep_summary.csv'))); starts = list(np.where(data[:, 1] > 0.5)[0]) + [len(data)]
    ci = {c: i for i, c in enumerate(cols)}; tx = np.abs(data[:, ci['u1_tx']]) / A; ty = np.abs(data[:, ci['u2_ty']]) / A; tn = np.hypot(data[:, ci['u1_tx']], data[:, ci['u2_ty']]) / A
    print("② 토크 권한 사용률 |τ_xy|/4.36 (δ0 track, 세그 시작 3s 이후): p50 / p95 / p99 / max   [x축 p95, y축 p95]")
    au = collections.defaultdict(list)
    for k in range(min(len(starts) - 1, len(summ))):
        r = summ[k]
        if float(r['bias']) != 0 or r['policy'] != 'track': continue
        s0, s1 = starts[k] + 150, starts[k + 1]; au[(r['pattern'], int(round(float(r['wind_speed']))))].append((tn[s0:s1], tx[s0:s1], ty[s0:s1]))
    for key, L in sorted(au.items()):
        n = np.concatenate([a for a, _, _ in L]); x = np.concatenate([b for _, b, _ in L]); y = np.concatenate([c for _, _, c in L])
        print(f"  {key[0]:10s} ws{key[1]}: {np.percentile(n,50):.3f} / {np.percentile(n,95):.3f} / {np.percentile(n,99):.3f} / {n.max():.3f}   [x {np.percentile(x,95):.3f}, y {np.percentile(y,95):.3f}]")
    print("③ 추락 에피소드: 공격 온셋 직전 1s 권한 p95 · 온셋→추락 사이 |τ_xy| 명령 max (포화=1.0 근접?)  (track, 사망 셀)")
    atk = data[:, 2] > 0.5
    for k in range(min(len(starts) - 1, len(summ))):
        r = summ[k]
        if r['policy'] != 'track' or int(float(r['survived'])) == 1: continue
        s0, s1 = starts[k], starts[k + 1]; a = np.where(atk[s0:s1])[0]
        if not len(a): continue
        o = s0 + a[0]; pre = tn[max(s0, o - 50):o]; post = tn[o:s1]
        print(f"  δ{float(r['bias'])/A:.2f} ws{int(round(float(r['wind_speed'])))} {r['pattern']:10s} 온셋전 p95 {np.percentile(pre,95):.3f}  공격중 max {post.max():.3f} p95 {np.percentile(post,95):.3f}  추락 {(int(float(r['crash_step']))-180)/10:.1f}s")
