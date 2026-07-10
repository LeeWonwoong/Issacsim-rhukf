#!/usr/bin/env python3
"""오프라인 gyro Q·R 2D sweep (재학습·재시뮬 불필요).

기록된 zu_log.npz(z,u 50Hz 스트림)에 DynamicsUKF를 (Q_gyro × R_gyro) 12조합으로 재실행 →
온셋정렬 gyro NIS(침묵 d4~10 채워지나) + 정상/급기동 NIS(FAR 위험) + 분리도/SNR 재계산.

검증 가설(2026-07-10): d4~10 침묵은 PX4 보상이 아니라 UKF가 드리프트를 흡수(gt_err↑·NIS↓)한 것.
Q↓ 또는 R↑(고집필터)면 innovation 유지 → NIS가 t=2~10 연속으로 이어짐.
Q만 sweep은 정규화 트레이드오프의 단면 → 2D(Q×R)로 확장. gyro 먼저(주 채널).

■ 격자 (Q 넓게 × R 좁게)
  Q_gyro ∈ {5e-4(현행), 5e-5, 5e-6, 5e-7}   (모델불확실성 → 자유도 큼)
  R_gyro ∈ {0.5(현행 class default), 1.0, 2.0}  (실측노이즈 → 1.0 근처만)
  ★R 제약: R은 센서 실측노이즈에 묶인 물리값. 크게 벗어나면 정상 추정도 나빠짐 → 0.5~2.0.

■ 지표 5종 (3D 트레이드오프)
  [신호↑]  1. d4~10 gyro NIS median   2. d2~10 정상p99 넘는 '연속' 스텝수(핵심)   3. SNR=(d4~10 med)/(정상 p99)
  [비용↓]  4. 정상 gyro NIS med/p95/p99(FAR)   5. 급기동 med + 급기동↔공격 분리도 d'(안전장치)

★ sanity: 현행(Q5e-4, R0.5=class default)이 로그된 온셋 t=2 gyro NIS≈3.15 를 재현하는지 먼저 확인.
  (주의: ukf_filter.py:99 gyro R 리터럴은 0.5 — 주석 '0.5→1.0'은 gyro엔 미적용. 현행=0.5.)

사용: python3 replay_q_sweep.py [zu_log.npz] [heatmap_out.png]
컬럼: 0 episode,1 reset,2 attack,3 action, 4-12 z(9), 13-16 u(4), 17-19 euler, 20 scale, 21 delay
"""
import sys, os, numpy as np
from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled

Z       = sys.argv[1] if len(sys.argv) > 1 else 'results_capture/zu_log.npz'
PNG_OUT = sys.argv[2] if len(sys.argv) > 2 else 'results_capture/qr_heatmap.png'

Q_CANDS = [5e-4, 5e-5, 5e-6, 5e-7]          # 5e-4 = 현행 baseline
R_CANDS = [0.5, 1.0, 2.0]                   # 0.5 = 현행 class default (ukf_filter.py:99)
BASELINE_Q, BASELINE_R = 5e-4, 0.5          # 로그를 만든 필터 = sanity 기준셀
SANITY_DELAY, SANITY_TARGET = 2, 3.15       # 로그된 t=2 온셋 gyro NIS

d = np.load(Z, allow_pickle=True)
data = d['data'].astype(np.float64)
dt = float(d['dt'])
calib = load_calibration('calibration.json')
print(f"# zu_log: {data.shape[0]} steps @ dt={dt} (q_gate={float(d.get('q_gate', 0.0))})")

# gyro R 은 9-차원 관측벡터([pos3, vel3, gyro3])의 인덱스 6,7,8.
GYRO_R_IDX = (6, 7, 8)
GYRO_Q_IDX = (9, 10, 11)

def replay(qg, rg):
    ukf = DynamicsUKF(dt=dt, calib=calib, q_gate=0.0)
    for i in GYRO_Q_IDX:
        ukf.Q[i, i] = qg
    for i in GYRO_R_IDX:
        ukf.R[i, i] = rg
    out = np.empty((data.shape[0], 6))     # episode, attack, scale, delay, ng_raw, ng_scl
    for k, row in enumerate(data):
        reset = row[1]
        z = row[4:13]; u = row[13:17]; euler = row[17:20]
        if reset > 0.5:
            ukf.x = np.zeros(12)
            ukf.x[0:3] = z[0:3]; ukf.x[3:6] = euler
            ukf.x[6:9] = z[3:6]; ukf.x[9:12] = z[6:9]
            ukf.P = np.eye(12) * 0.1
            ukf.is_ukf_initialized = True
        res, Pzz = ukf.step(z, u)
        ng_raw, ng_scl = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)
        out[k] = (row[0], row[2], row[20], row[21], ng_raw, ng_scl)
    return out

# episode 경계: reset==1 이 새 에피소드 시작
ep_start = np.where(data[:, 1] > 0.5)[0]
ep_bounds = list(ep_start) + [data.shape[0]]

def analyze(res):
    # res cols: 0 ep_id,1 attack,2 scale,3 delay,4 ng_raw,5 ng_scl
    norm_raw, man_raw = [], []
    onset_raw = {}                          # delay(10Hz) -> list(raw NIS)
    for a, b in zip(ep_bounds[:-1], ep_bounds[1:]):
        seg = res[a:b]; seg_data = data[a:b]
        atk_ever = seg[:, 1].max() > 0.5
        if not atk_ever:
            norm_raw.extend(seg[:, 4])
            # 급기동 스텝 = 명령토크 |u1..u3| 상위 20% (정상 내 기동)
            tqmag = np.linalg.norm(seg_data[:, 14:17], axis=1)
            thr = np.percentile(tqmag, 80) if len(tqmag) else 0
            man_raw.extend(seg[tqmag >= thr, 4])
        else:
            # track(=action 항상 0) 에피소드만 순수 온셋 시그니처
            if seg_data[:, 3].max() > 0.5:
                continue
            for j in range(len(seg)):
                if seg[j, 1] > 0.5:
                    dl = int(round(seg[j, 3]))    # 10Hz 공격경과
                    onset_raw.setdefault(dl, []).append(seg[j, 4])
    return np.asarray(norm_raw), np.asarray(man_raw), onset_raw


def _pct(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
def _med(a):    return float(np.median(a)) if len(a) else float('nan')

def dprime(atk, man):
    """급기동↔공격 분리도. |μ_a-μ_m| / sqrt(0.5(σa²+σm²)). 높을수록 안전."""
    if len(atk) < 2 or len(man) < 2:
        return float('nan')
    va, vm = np.var(atk, ddof=1), np.var(man, ddof=1)
    denom = np.sqrt(0.5 * (va + vm))
    return abs(np.mean(atk) - np.mean(man)) / denom if denom > 0 else float('nan')

def consec_above(onset_raw, thr):
    """d2~10 온셋 median 이 thr(정상 p99)을 넘는 '연속' 스텝 최장길이 (탐지창 연속성)."""
    best = cur = 0
    for dl in range(2, 11):
        seq = onset_raw.get(dl)
        med = np.median(seq) if seq else -np.inf
        if med > thr:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best

def metrics(norm_raw, man_raw, onset_raw):
    n_med, n_p95, n_p99 = _med(norm_raw), _pct(norm_raw, 95), _pct(norm_raw, 99)
    d410_parts = [np.asarray(onset_raw[dl]) for dl in range(4, 11) if onset_raw.get(dl)]
    d410 = np.concatenate(d410_parts) if d410_parts else np.array([])
    d410_med = _med(d410)
    snr = d410_med / n_p99 if (n_p99 and n_p99 > 0) else float('nan')
    consec = consec_above(onset_raw, n_p99)
    m_med = _med(man_raw)
    dp = dprime(d410, man_raw)
    return dict(n_med=n_med, n_p95=n_p95, n_p99=n_p99, d410_med=d410_med,
                snr=snr, consec=consec, m_med=m_med, dprime=dp)

# ── 재실행 (12조합) ──────────────────────────────────────────────
print("\n" + "=" * 92)
res_cache = {}   # (qg,rg) -> analyze tuple
M = {}           # (qg,rg) -> metrics dict
for rg in R_CANDS:
    for qg in Q_CANDS:
        nr, mr, oraw = analyze(replay(qg, rg))
        res_cache[(qg, rg)] = (nr, mr, oraw)
        M[(qg, rg)] = metrics(nr, mr, oraw)

# ── SANITY ───────────────────────────────────────────────────────
base_oraw = res_cache[(BASELINE_Q, BASELINE_R)][2]
base_t2 = np.median(base_oraw[SANITY_DELAY]) if base_oraw.get(SANITY_DELAY) else float('nan')
ok = abs(base_t2 - SANITY_TARGET) <= 0.6
print(f"★ SANITY: 현행(Q={BASELINE_Q:.0e}, R={BASELINE_R}) t={SANITY_DELAY} gyro NIS median = "
      f"{base_t2:.3f} (목표≈{SANITY_TARGET}) → {'PASS ✅' if ok else 'MISMATCH ⚠ replay 정합성부터 확인'}")
print("=" * 92)

# ── 표1: 온셋정렬 gyro NIS(raw median) — R블록별 delay×Q ──────────
for rg in R_CANDS:
    tag = "  (현행 R)" if rg == BASELINE_R else ""
    print(f"\n★ 표1 [R={rg}]{tag}: 온셋정렬 gyro NIS(raw median), delay(10Hz) × Q")
    print(f"{'delay':>5} | " + " | ".join(f"Q={qg:.0e}".rjust(10) for qg in Q_CANDS))
    for dl in range(0, 12):
        cells = []
        for qg in Q_CANDS:
            o = res_cache[(qg, rg)][2]
            cells.append(f"{np.median(o[dl]):>10.3f}" if o.get(dl) else f"{'—':>10}")
        marker = " ←침묵구간" if 4 <= dl <= 10 else ""
        print(f"{dl:>5} | " + " | ".join(cells) + marker)

# ── 표2: 12조합 트레이드오프 ─────────────────────────────────────
print("\n★ 표2: 12조합 트레이드오프  [신호 d4-10med·연속·SNR | 비용 정상p99·급기동d']")
hdr = (f"{'Q':>8} {'R':>5} | {'d4-10med':>9} {'연속':>4} {'SNR':>6} | "
       f"{'정상med':>7} {'p95':>6} {'p99':>6} | {'급기동med':>9} {'급vs공dp':>8}")
print(hdr); print("-" * len(hdr))
for rg in R_CANDS:
    for qg in Q_CANDS:
        m = M[(qg, rg)]
        base = (qg == BASELINE_Q and rg == BASELINE_R)
        tag = "  ← baseline" if base else ""
        print(f"{qg:>8.0e} {rg:>5.1f} | {m['d410_med']:>9.3f} {m['consec']:>4d} {m['snr']:>6.2f} | "
              f"{m['n_med']:>7.4f} {m['n_p95']:>6.3f} {m['n_p99']:>6.3f} | "
              f"{m['m_med']:>9.3f} {m['dprime']:>8.2f}{tag}")

# ── 히트맵 (Q행 × R열): 신호 / 정상p99 / 급기동d' / SNR ───────────
def _grid(key):
    return np.array([[M[(qg, rg)][key] for rg in R_CANDS] for qg in Q_CANDS])

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
        if os.path.exists(_fp):
            try:
                font_manager.fontManager.addfont(_fp)
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=_fp).get_name()
            except Exception:
                pass
            break
    plt.rcParams['axes.unicode_minus'] = False

    panels = [
        ('d410_med', '신호: d4-10 gyro NIS median (↑좋음)', 'viridis', False),
        ('n_p99',    '비용: 정상 gyro NIS p99 (↓좋음, FAR)', 'Reds',    False),
        ('dprime',   '안전: 급기동↔공격 분리도 d\' (↑좋음)',  'viridis', False),
        ('snr',      'SNR = d4-10med / 정상p99 (↑좋음)',      'viridis', False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    qlabels = [f"{q:.0e}" for q in Q_CANDS]
    rlabels = [f"{r:.1f}" for r in R_CANDS]
    for ax, (key, title, cmap, _) in zip(axes.ravel(), panels):
        G = _grid(key)
        im = ax.imshow(G, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(R_CANDS))); ax.set_xticklabels(rlabels)
        ax.set_yticks(range(len(Q_CANDS))); ax.set_yticklabels(qlabels)
        ax.set_xlabel('R_gyro'); ax.set_ylabel('Q_gyro')
        ax.set_title(title, fontsize=10)
        for i in range(len(Q_CANDS)):
            for j in range(len(R_CANDS)):
                v = G[i, j]
                txt = '—' if not np.isfinite(v) else (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}")
                ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                        color='white' if cmap != 'Reds' else 'black')
        # baseline 셀 테두리
        bi, bj = Q_CANDS.index(BASELINE_Q), R_CANDS.index(BASELINE_R)
        ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False, edgecolor='cyan', lw=2.5))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('gyro Q·R 2D sweep — 신호/비용/안전 트레이드오프 (청록테두리=현행)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(PNG_OUT) or '.', exist_ok=True)
    fig.savefig(PNG_OUT, dpi=130)
    print(f"\n[heatmap] 저장 → {PNG_OUT}")
except Exception as e:
    print(f"\n[heatmap] 스킵 ({type(e).__name__}: {e})")

print("\n판정 가이드:")
print("  sweet spot = d4-10med 가 정상 p99 위(연속≥3~4) ∧ 정상 p99 감당 ∧ 급기동 d' 유지(≥~5) 셀.")
print("  → 그 (Q,R)로 고집필터 확정 후 재학습(관측 재생성 → 5-클래스 분리도 재검증).")
print("  어느 셀도 정상/급기동 폭증 없이 신호 못 채우면 → 필터튜닝 실패 → 펄스활용(reward 대수술).")
