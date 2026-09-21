#!/usr/bin/env python3
"""plot_policy_log.py — f13_policy 로그(CSV) → 3D 궤적(공격 on/off·hover on/off 색) + 시간축(NIS 관측·δ·행동·Q). 젯슨(matplotlib 만)에서도 돈다.

    python3 plot_policy_log.py field_logs/f13_policy_circle_XXXX.csv                 # 창으로 보기
    python3 plot_policy_log.py <csv> --out fig.png                                   # 파일 저장(헤드리스)
    python3 plot_policy_log.py <csv> --live                                          # 비행 중 실시간 갱신(0.5 s, 파일을 계속 읽음)
    python3 plot_policy_log.py --latest field_logs --live                            # 폴더에서 가장 새 로그
"""
import argparse, csv, glob, os, sys, time
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    # 첫 ENGAGED 구간만 (t_seq 가 되감기면 재진입 → 이후 행 제외)
    ts = [float(r['t_seq']) for r in rows]; eng = [i for i, r in enumerate(rows) if r['state'] == 'ENGAGED']
    for a, b in zip(eng[:-1], eng[1:]):
        if ts[b] < ts[a] - 1.0: rows = rows[:b]; break
    f = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: f(k) for k in ('t_seq', 'nis_v_raw', 'nis_g_raw', 'obs_v', 'obs_g', 'q_track', 'q_hover', 'action', 'hovering', 'atk_active', 'delta', 'x', 'y', 'z')}
    d['state'] = np.array([r['state'] for r in rows]); d['t'] = f('t_wall'); d['t'] -= d['t'][0]
    d['atk'] = (d['delta'] > 0) | (d['atk_active'] > 0)
    if 'sp_x' in rows[0]:
        for k in ('sp_x', 'sp_y', 'sp_z'): d[k] = np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan for r in rows])
        d['err'] = np.hypot(d['x'] - d['sp_x'], d['y'] - d['sp_y'])
    return d


def draw(fig, d, title=''):
    import matplotlib.pyplot as plt
    fig.clf()
    eng = d['state'] == 'ENGAGED'
    has_sp = 'sp_x' in d
    ax3 = fig.add_subplot(2, 2, 1, projection='3d') if has_sp else fig.add_subplot(1, 2, 1, projection='3d')
    x, y, z = d['x'], d['y'], -d['z']
    cols = np.where(d['hovering'] > 0, '#d1483a', np.where(d['atk'], '#e8a33c', '#2f62e6'))
    ax3.scatter(x[eng], y[eng], z[eng], c=cols[eng], s=6, depthshade=False)
    ax3.plot(x[eng], y[eng], z[eng], color='gray', lw=0.5, alpha=.5)
    ax3.set_xlabel('N [m]'); ax3.set_ylabel('E [m]'); ax3.set_zlabel('alt [m]'); ax3.set_title('궤적 — 파랑 track · 주황 공격 중(track) · 빨강 hover')
    if eng.any():   # ★ z 축 자동 눈금이 수 cm 로 잡혀 '출렁임'처럼 보이는 것 방지: 수평 폭에 비례한 최소 범위
        span = max(np.ptp(x[eng]), np.ptp(y[eng]), 2.0); zc = np.median(z[eng]); half = max(0.5 * span * 0.35, np.ptp(z[eng]) / 2 + 0.05)
        ax3.set_zlim(zc - half, zc + half); ax3.set_box_aspect((np.ptp(x[eng]) + 1e-3, np.ptp(y[eng]) + 1e-3, 2 * half * 0.6))
    if has_sp:
        a2d = fig.add_subplot(2, 2, 3); m = eng & np.isfinite(d['sp_x'])
        a2d.plot(d['sp_y'][m], d['sp_x'][m], 'k--', lw=1, label='명령 설정점'); a2d.scatter(d['y'][m], d['x'][m], c=cols[m], s=6, label='실제')
        a2d.set_aspect('equal'); a2d.set_xlabel('E [m]'); a2d.set_ylabel('N [m]'); a2d.grid(alpha=.3); a2d.legend(fontsize=7)
        trk = m & (d['hovering'] == 0) & (~d['atk'])
        a2d.set_title(f"탑뷰 — 무공격 track 추종 RMSE {np.sqrt(np.nanmean(d['err'][trk]**2)):.2f} m · 최대 {np.nanmax(d['err'][trk]):.2f} m" if trk.any() else '탑뷰')
    t = d['t']
    a1 = fig.add_subplot(4, 2, 2); a1.plot(t, d['obs_g'], color='#d1483a', lw=1, label='gyro 관측'); a1.plot(t, d['obs_v'], color='#2f62e6', lw=1, label='vel 관측'); a1.set_ylim(0, 1); a1.legend(fontsize=7, loc='upper left'); a1.set_ylabel('obs')
    a2 = fig.add_subplot(4, 2, 4, sharex=a1); a2.plot(t, d['delta'], color='#e8a33c', lw=1.2); a2.set_ylabel('δ (공격)'); a2.set_ylim(0, 0.9)
    a3 = fig.add_subplot(4, 2, 6, sharex=a1); a3.step(t, d['action'], color='#d1483a', lw=1, label='정책 action'); a3.step(t, d['hovering'] * 0.8, color='k', lw=0.8, label='hover 실행'); a3.set_ylim(-0.1, 1.2); a3.legend(fontsize=7, loc='upper left'); a3.set_ylabel('행동')
    a4 = fig.add_subplot(4, 2, 8, sharex=a1); a4.plot(t, d['q_hover'] - d['q_track'], color='#7b3fe0', lw=1); a4.axhline(0, color='gray', ls=':'); a4.set_ylabel('Q_hover − Q_track'); a4.set_xlabel('t [s]')
    for a in (a1, a2, a3):
        for s0, s1 in _spans(t, d['atk']): a.axvspan(s0, s1, color='#e8a33c', alpha=.15)
        for s0, s1 in _spans(t, d['hovering'] > 0): a.axvspan(s0, s1, color='#d1483a', alpha=.12)
    n_decl = int(np.sum(np.diff(np.r_[0, d['hovering']]) > 0)); n_atk = int(np.sum(np.diff(np.r_[0, d['atk'].astype(int)]) > 0))
    fig.suptitle(f'{title}  —  공격 사건 {n_atk} · hover 선언 {n_decl} · 스텝 {len(t)} · gyro obs max {d["obs_g"].max():.2f}', fontsize=10)
    fig.tight_layout()


def _spans(t, mask):
    m = mask.astype(int); starts = np.flatnonzero(np.diff(np.r_[0, m]) > 0); ends = np.flatnonzero(np.diff(np.r_[m, 0]) < 0)
    return [(t[s], t[min(e, len(t) - 1)]) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?'); ap.add_argument('--latest', help='폴더에서 가장 새 f13_policy_*.csv')
    ap.add_argument('--live', action='store_true'); ap.add_argument('--out', default=None); ap.add_argument('--every', type=float, default=0.5)
    a = ap.parse_args()
    path = a.csv or sorted(glob.glob(os.path.join(a.latest, 'f13_policy_*.csv')), key=os.path.getmtime)[-1]
    import matplotlib
    if a.out and not a.live: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(14, 7))
    if a.live:
        plt.ion(); last = 0
        while plt.fignum_exists(fig.number):
            n = os.path.getsize(path)
            if n != last:
                d = load(path)
                if d is not None: draw(fig, d, os.path.basename(path)); fig.canvas.draw_idle()
                last = n
            plt.pause(a.every)
        return
    d = load(path)
    if d is None: sys.exit('빈 로그')
    draw(fig, d, os.path.basename(path))
    if a.out: fig.savefig(a.out, dpi=120); print('저장', a.out)
    else: plt.show()


if __name__ == '__main__':
    main()
