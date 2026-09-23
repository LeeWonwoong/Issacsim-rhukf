#!/usr/bin/env python3
"""plot_obs_live.py — 정책 입력 12-D(4스텝 창 × [vel, gyro, prev_action])와 매 스텝 행동을 time-step 축으로 실시간 갱신 (2026-09-22).

    python3 px4_field/plot_obs_live.py --latest px4_field/field_logs --live            # 비행 중, 1 s 마다 갱신, 최근 30 s
    python3 px4_field/plot_obs_live.py <csv> --out fig.png                             # 사후 저장(헤드리스)
    python3 px4_field/plot_obs_live.py <csv> --window 60 --every 0.5 --live            # 창 60 s, 0.5 s 갱신
    python3 plot_obs_live.py --latest field_logs --live --light --every 2 --window 20   # 젯슨(ssh -X): 히트맵 없이 가볍게

  ① 정책 입력: vel/gyro 관측(0–1)·직전 행동 시계열 + 최근 4스텝 창(보라) + 지금 12-D 벡터 막대(오른쪽)
  ② 실제 값: √NIS(마할라노비스 거리, 로그축) + 공격 δ(주황 점선)
  ③ NN 행동 · hover 실행 · Q_hover−Q_track.  배경: 주황 = 공격(δ>0) · 빨강 = hover 실행
  CSV 에 o0..o11 열이 있으면(09-22 이후 f13 로그) 노드가 실제로 정책에 넣은 벡터를 그대로 쓰고,
  없으면(구 로그) obs_v/obs_g/action 을 4스텝 시프트해 재구성한다(재접근 리셋 구간만 미세 차이).
  matplotlib 만 쓴다(젯슨 OK).
"""
import argparse, csv, glob, os, sys, time
import numpy as np

W = 4; FEATS = ('vel', 'gyro', 'act')


def load(path, window_s=None):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 2: return None
    f = lambda k: np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan for r in rows])
    d = {k: f(k) for k in ('t_wall', 'obs_v', 'obs_g', 'action', 'hovering', 'delta', 'q_track', 'q_hover', 'nis_v_raw', 'nis_g_raw')}
    d['t'] = d['t_wall'] - d['t_wall'][0]; d['state'] = np.array([r['state'] for r in rows])
    if 'o0' in rows[0]:                                    # 노드가 기록한 정책 입력 벡터
        O = np.array([[float(r[f'o{i}']) if r[f'o{i}'] != '' else np.nan for i in range(W * 3)] for r in rows])
    else:                                                  # 구 로그: 재구성 (프레임 k = [obs_v_k, obs_g_k, action_{k−1}])
        prev = np.r_[0, d['action'][:-1]]; fr = np.stack([d['obs_v'], d['obs_g'], prev], 1); O = np.full((len(rows), W * 3), np.nan)
        for k in range(W - 1, len(rows)): O[k] = fr[k - W + 1:k + 1].reshape(-1)
    d['O'] = O
    if window_s:
        m = d['t'] >= d['t'][-1] - window_s
        d = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in d.items()}
    return d


def draw(fig, d, title, light=False):
    """위: 정책 입력(12-D) = 세 채널의 최근 4스텝 창 (시계열 + 지금 벡터 막대) · 중간: 실제 NIS(마할라노비스 거리) · 아래: 행동"""
    fig.clf(); t = d['t']; n = len(t)
    prev = np.r_[0, d['action'][:-1]]                      # 프레임의 세 번째 성분 = 직전 행동
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 0.9], hspace=0.32, wspace=0.35)
    ax_o = fig.add_subplot(gs[0, :3] if not light else gs[0, :]); ax_b = None if light else fig.add_subplot(gs[0, 3])
    ax_n = fig.add_subplot(gs[1, :], sharex=ax_o); ax_a = fig.add_subplot(gs[2, :], sharex=ax_o)
    # ── (1) 정책 입력 채널 시계열 + 최근 4스텝 창 표시
    ax_o.plot(t, d['obs_v'], color='#2f62e6', lw=1.3, label='vel 관측 = log1p(√NIS_vel)/4')
    ax_o.plot(t, d['obs_g'], color='#d1483a', lw=1.3, label='gyro 관측 = log1p(√NIS_gyro)/4')
    ax_o.step(t, prev * 0.95, color='#555', lw=1.0, where='post', label='직전 행동 (0 track / 1 hover)')
    if n >= W: ax_o.axvspan(t[-W], t[-1], color='#7b3fe0', alpha=.18, label='지금 정책에 들어간 4스텝 창')
    ax_o.set_ylim(0, 1.02); ax_o.set_ylabel('정책 입력 (0–1)'); ax_o.legend(fontsize=7, loc='upper left', ncol=2); ax_o.grid(alpha=.3)
    ax_o.set_title('① NN 이 보는 것: 세 채널의 최근 4스텝 = 12차원', fontsize=9, loc='left')
    # ── (1b) 지금 벡터 12-D 막대 (프레임 t−3…t × [vel, gyro, act])
    if ax_b is not None:
        O = d['O'][-1] if np.all(np.isfinite(d['O'][-1])) else np.nan_to_num(d['O'][-1])
        cols = ['#2f62e6', '#d1483a', '#555'] * W; xs = np.arange(W * 3) + np.repeat(np.arange(W), 3) * 0.6
        ax_b.bar(xs, O, color=cols, width=0.8); ax_b.set_ylim(0, 1.02)
        ax_b.set_xticks([xs[3 * k + 1] for k in range(W)]); ax_b.set_xticklabels([f't−{W - 1 - k}' if k < W - 1 else 't', ] if False else [f't−{W - 1 - k}' if k < W - 1 else 't' for k in range(W)], fontsize=8)
        ax_b.set_title('지금 12-D 벡터\n(파랑 vel · 빨강 gyro · 회색 행동)', fontsize=8); ax_b.grid(alpha=.3, axis='y')
    # ── (2) 실제 값: 마할라노비스 거리 √NIS (로그축) + 공격 δ
    dv = np.sqrt(np.maximum(d['nis_v_raw'], 0)); dg = np.sqrt(np.maximum(d['nis_g_raw'], 0))
    ax_n.plot(t, dg, color='#d1483a', lw=1.2, label='gyro √NIS (마할라노비스 거리)'); ax_n.plot(t, dv, color='#2f62e6', lw=1.2, label='vel √NIS')
    ax_n.set_yscale('log'); ax_n.set_ylim(0.05, max(30, float(np.nanmax(np.r_[dg, dv, 1])) * 1.5)); ax_n.set_ylabel('√NIS (로그)'); ax_n.grid(alpha=.3, which='both')
    ax_n.axhline(1.0, color='gray', ls=':', lw=0.8); ax_n.legend(fontsize=7, loc='upper left')
    ax_d = ax_n.twinx(); ax_d.plot(t, d['delta'], color='#e8a33c', lw=1.4, ls='--', label='공격 δ (조종기 VRA on → ATK_TQ_MAX)'); ax_d.set_ylim(0, 1.0); ax_d.set_ylabel('δ', color='#e8a33c'); ax_d.legend(fontsize=7, loc='upper right')
    ax_n.set_title('② 실제 잔차: √NIS (1 = 모델과 일치, 클수록 설명 안 되는 회전/속도) · 주황 점선 = 공격 δ', fontsize=9, loc='left')
    # ── (3) 행동
    ax_a.step(t, d['action'], color='#d1483a', lw=1.6, where='post', label='NN 행동 (1 = hover 선언)')
    ax_a.step(t, d['hovering'] * 0.8, color='k', lw=0.9, where='post', label='hover 실행 중')
    ax_a.set_ylim(-0.1, 1.2); ax_a.set_ylabel('행동'); ax_a.set_xlabel('t [s]'); ax_a.legend(fontsize=7, loc='upper left')
    ax_q = ax_a.twinx(); dq = d['q_hover'] - d['q_track']; ax_q.plot(t, dq, color='#7b3fe0', lw=0.8, alpha=.8); ax_q.axhline(0, color='#7b3fe0', ls=':', lw=0.6); ax_q.set_ylabel('Q_hover − Q_track', color='#7b3fe0', fontsize=8)
    ax_a.set_title('③ NN 결정 (보라 = Q 차이, 0 을 넘으면 hover)', fontsize=9, loc='left')
    for a in (ax_o, ax_n, ax_a):
        for s0, s1 in _spans(t, d['delta'] > 0): a.axvspan(s0, s1, color='#e8a33c', alpha=.15)
        for s0, s1 in _spans(t, d['hovering'] > 0): a.axvspan(s0, s1, color='#d1483a', alpha=.10)
    last = d['state'][-1]
    fig.suptitle(f'{title}  —  {last} · 스텝 {n} · 행동 {"HOVER" if d["action"][-1] > 0 else "track"} · δ {d["delta"][-1]:.2f} · √NIS gyro {dg[-1]:.2f} vel {dv[-1]:.2f}     [배경: 주황 = 공격 δ>0 · 빨강 = hover 실행]', fontsize=9)


def _spans(t, mask):
    m = mask.astype(int); st = np.flatnonzero(np.diff(np.r_[0, m]) > 0); en = np.flatnonzero(np.diff(np.r_[m, 0]) < 0)
    return [(t[a], t[min(b, len(t) - 1)]) for a, b in zip(st, en)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='?'); ap.add_argument('--latest', help='폴더에서 가장 새 f13_policy_*.csv')
    ap.add_argument('--live', action='store_true'); ap.add_argument('--every', type=float, default=1.0, help='갱신 주기 [s]')
    ap.add_argument('--window', type=float, default=30.0, help='표시 창 [s] (0=전체)'); ap.add_argument('--out', default=None)
    ap.add_argument('--light', action='store_true', help='젯슨용 경량: 12-D 막대 생략·작은 창 (--every 2 권장)')
    a = ap.parse_args()
    def latest():
        fs = sorted(glob.glob(os.path.join(a.latest or 'field_logs', 'f13_policy_*.csv')), key=os.path.getmtime)
        return fs[-1] if fs else None
    if a.csv:
        path = a.csv
    else:
        path = latest()          # ★09-22 버그 수정: 이전엔 sys.exit 와 같은 줄이라 path 가 배정되지 않았다
        if path is None: sys.exit('f13_policy_*.csv 없음')
    import matplotlib
    if a.out and not a.live: matplotlib.use('Agg')
    else:
        for _bk in ('TkAgg', 'Qt5Agg', 'GTK3Agg'):      # ★ 화면 백엔드를 명시적으로 고른다(ssh -X 에서 Agg 로 떨어지는 것 방지)
            try:
                matplotlib.use(_bk); import matplotlib.pyplot as _plt; _plt.figure(); _plt.close('all'); break
            except Exception:
                continue
    import matplotlib.pyplot as plt
    print('matplotlib backend:', matplotlib.get_backend())
    if not a.out and matplotlib.get_backend().lower().startswith('agg'):
        sys.exit('화면 백엔드 없음(DISPLAY 미설정) — ssh -X 로 접속하거나, 노트북에서 rsync 로 로그를 받아 그리거나, --out 파일로 저장하세요')
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f: font_manager.fontManager.addfont(f)
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(9, 5) if a.light else (13, 8)); win = a.window if a.window > 0 else None
    if a.live:
        from matplotlib.animation import FuncAnimation
        st = {'path': path, 'last': -1}
        def update(_frame):
            try:
                if not a.csv:                      # 새 소티가 시작되면 가장 새 로그로 자동 전환
                    p2 = latest()
                    if p2 and p2 != st['path']: st['path'] = p2; st['last'] = -1
                n = os.path.getsize(st['path'])
                if n != st['last']:
                    nrows = sum(1 for _ in open(st['path'])) - 1
                    if nrows < 2:                              # ★ 노드가 아직 행을 안 씀(UKF 입력 대기 등) → 흰 창 대신 안내
                        fig.clf(); fig.text(0.5, 0.5, f'로그 대기 중: {os.path.basename(st["path"])}\n행 {max(nrows, 0)} — 노드가 "UKF 초기화" 뒤 10 Hz 로 씁니다', ha='center', va='center', fontsize=12)
                        print(f'로그 대기 중: {st["path"]} 행 {max(nrows, 0)}', flush=True); st['last'] = n; return []
                    d = load(st['path'], win)
                    if d is not None: draw(fig, d, os.path.basename(st['path']), a.light)
                    st['last'] = n
                    if not st.get('shown'): st['shown'] = True; print('첫 그림 완료:', os.path.basename(st['path']), '스텝', 0 if d is None else len(d['t']), flush=True)
            except Exception as e:                        # 파일이 쓰이는 중이면 다음 주기에 다시
                print('갱신 건너뜀:', e)
            return []
        update(0)
        anim = FuncAnimation(fig, update, interval=int(a.every * 1000), cache_frame_data=False)
        plt.show()                                        # 이벤트 루프가 그림을 실제로 그린다 (창 닫으면 종료)
        return
    d = load(path, win if a.window > 0 and not a.out else None)
    if d is None: sys.exit('빈 로그')
    draw(fig, d, os.path.basename(path), a.light)
    if a.out: fig.savefig(a.out, dpi=120); print('저장', a.out)
    else: plt.show()


if __name__ == '__main__':
    main()
