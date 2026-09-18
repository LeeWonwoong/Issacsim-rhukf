#!/usr/bin/env python3
"""
nis_separability.py — "이 환경이 RL에게 풀 수 있는가"를 측정으로 답한다
========================================================================
zu_log을 UKF로 재생(현재 튜닝 Q/R)해 NIS를 만들고, 세 가지 NIS 상승원인
  ① 공격  ② 급기동  ③ 내 hover선택
이 RL의 실제 입력(윈도우4 × [nis_v,nis_g,action])으로 구별 가능한지 정량화.

핵심 측정:
 [A] per-step d' vs 윈도우-평균 d' vs ★RL 12차원 feature 다변량 d'
     → RL 입력이 per-step보다 얼마나 더 잘 분리하나 (윈도우 효과)
 [B] NIS '상승 지속길이' 분포: 정상(급기동/hover) vs 공격
     → 급기동 스파이크가 윈도우(4) 안에 사라지나? 공격은 더 기나?
 [C] 정상 상승의 원인 분해: hover(③) / 급기동(②) / 기타
     → a_{t-1}이 ③을 얼마나 커버하나
 [D] 공격 강도별 탐지율 (recall 천장 진단)

사용: python3 nis_separability.py results_zu/zu_log.npz [--window 4]
"""
import argparse, sys
import numpy as np
try:
    from ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled
except Exception:
    from env.ukf_filter import DynamicsUKF, load_calibration, compute_nis_scaled


def replay_nis(data, dt, calib):
    """zu_log을 UKF로 재생 → per-step scaled NIS (vel, gyro)."""
    rst = data[:, 1]; z = data[:, 4:13]; u = data[:, 13:17]; eul = data[:, 17:20]
    nv = np.empty(len(data)); ng = np.empty(len(data))
    ukf = None
    for i in range(len(data)):
        if rst[i] > 0.5 or ukf is None:
            ukf = DynamicsUKF(dt=dt, calib=calib)      # 파일의 현재 튜닝 Q/R 사용
            ukf.x[0:3] = z[i, 0:3]; ukf.x[3:6] = eul[i]
            ukf.x[6:9] = z[i, 3:6]; ukf.x[9:12] = z[i, 6:9]
        res, Pzz = ukf.step(z[i], u[i])
        _, a = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3.0)  # 통일압축 offset=1.0 (학습 관측과 일치)
        _, b = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)              # gyro log1p 유지
        nv[i] = a; ng[i] = b
    return nv, ng


def dprime(x, a):
    xn, xa = x[a == 0], x[a == 1]
    if len(xn) < 5 or len(xa) < 5: return 0.0
    return (np.nanmean(xa) - np.nanmean(xn)) / np.sqrt(0.5*(np.nanvar(xn)+np.nanvar(xa)) + 1e-9)


def mahalanobis_dprime(X, a):
    """다변량 d': 두 클래스 평균의 마할라노비스 거리(pooled cov). RL 선형분리 상한."""
    Xn, Xa = X[a == 0], X[a == 1]
    if len(Xn) < 20 or len(Xa) < 20: return 0.0
    mn, ma = Xn.mean(0), Xa.mean(0)
    S = 0.5*(np.cov(Xn, rowvar=False) + np.cov(Xa, rowvar=False)) + 1e-6*np.eye(X.shape[1])
    diff = ma - mn
    try:
        return float(np.sqrt(diff @ np.linalg.solve(S, diff)))
    except np.linalg.LinAlgError:
        return 0.0


def causal_winmean(x, ep, win):
    """에피소드별 인과적 윈도우 평균(마지막 win스텝)."""
    out = np.copy(x)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]; xe = x[idx]; c = np.copy(xe)
        for k in range(len(xe)):
            c[k] = xe[max(0, k-win+1):k+1].mean()
        out[idx] = c
    return out


def build_window_feature(nv, ng, act, ep, win):
    """RL 실제 입력 재현: 윈도우 win × [nis_v, nis_g, action] flatten."""
    N = len(nv); D = win*3
    X = np.zeros((N, D))
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for j, k in enumerate(idx):
            for w in range(win):
                src = idx[max(0, j-(win-1-w))]   # 과거→현재
                X[k, w*3:w*3+3] = [nv[src], ng[src], act[src]]
    return X


def run_lengths(mask, ep):
    """에피소드별 연속 True 런 길이 리스트."""
    lens = []
    for e in np.unique(ep):
        m = mask[ep == e]; c = 0
        for v in m:
            if v: c += 1
            elif c > 0: lens.append(c); c = 0
        if c > 0: lens.append(c)
    return np.array(lens) if lens else np.array([0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zu'); ap.add_argument('--calib', default='calibration.json')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--episodes', type=int, default=None)
    args = ap.parse_args()

    npz = np.load(args.zu, allow_pickle=True); data = npz['data']
    dt = float(npz['dt']) if 'dt' in npz else 0.02
    if args.episodes:
        eps = np.unique(data[:, 0])[:args.episodes]; data = data[np.isin(data[:, 0], eps)]
    calib = load_calibration(args.calib)

    ep = data[:, 0]; atk = data[:, 2].astype(int); act = data[:, 3]
    tau = np.linalg.norm(data[:, 14:17], axis=1)     # 명령 토크크기 |u[1:4]|
    win = args.window

    print(f"로드 {len(data)}스텝 | 평시 {int((atk==0).sum())} 공격 {int((atk==1).sum())} | window={win}")
    print("UKF 재생 중...")
    nv, ng = replay_nis(data, dt, calib)
    p90 = float(np.nanpercentile(nv[atk == 0], 90))

    # ── [A] per-step vs 윈도우 vs RL 다변량 d' ──
    print("\n"+"="*78)
    print("[A] 분리도: per-step → 윈도우평균 → ★RL 12D feature (윈도우 효과의 핵심)")
    print("="*78)
    d_ps_v = dprime(nv, atk); d_ps_g = dprime(ng, atk)
    nvw = causal_winmean(nv, ep, win); ngw = causal_winmean(ng, ep, win)
    d_w_v = dprime(nvw, atk); d_w_g = dprime(ngw, atk)
    X = build_window_feature(nv, ng, act, ep, win)
    d_mv = mahalanobis_dprime(X, atk)
    print(f"  per-step      d'vel={d_ps_v:.2f}  d'gyr={d_ps_g:.2f}")
    print(f"  윈도우{win}-평균   d'vel={d_w_v:.2f}  d'gyr={d_w_g:.2f}   (시간누적 효과)")
    print(f"  ★RL {win*3}D feature 다변량 d'={d_mv:.2f}   (= RL이 실제 입력으로 분리 가능한 상한)")
    print(f"    해석: 다변량 d'가 per-step보다 충분히 크면 → RL은 분리 가능(환경 OK).")
    print(f"          d'≈1=약, 2=중, 3+=깔끔. (d'2면 겹침~16%, d'3이면 ~7%)")

    # ── [B] NIS 상승 지속길이: 정상 vs 공격 ──
    print("\n"+"="*78)
    print(f"[B] NIS 상승(>{p90:.3f}=평시p90) 지속길이 분포: 정상 vs 공격")
    print("="*78)
    elev = nv > p90
    rl_norm = run_lengths(elev & (atk == 0), ep)
    rl_atk  = run_lengths(elev & (atk == 1), ep)
    def _stat(name, r):
        pct_le = float(np.mean(r <= win)*100)
        print(f"  {name}: 런 {len(r)}개 | 중앙={np.median(r):.0f} p90={np.percentile(r,90):.0f} "
              f"max={r.max()} | ≤{win}스텝={pct_le:.0f}% >{win}={100-pct_le:.0f}%")
        return pct_le
    p_norm = _stat("정상상승(급기동/hover)", rl_norm)
    p_atk  = _stat("공격상승            ", rl_atk)
    print(f"  ★판정: 정상상승의 {p_norm:.0f}%가 ≤{win}스텝(윈도우가 흡수), "
          f"공격상승의 {100-p_atk:.0f}%가 >{win}스텝(윈도우에 남음).")
    print(f"          정상은 짧고(≤win) 공격은 길면(>win) → 윈도우로 구별 가능.")

    # ── [C] 정상 상승의 원인 분해 ──
    print("\n"+"="*78)
    print("[C] 정상(공격X) NIS 상승의 원인: hover(③) / 급기동(②) / 기타")
    print("="*78)
    norm_elev = elev & (atk == 0)
    tau_hi = np.nanpercentile(tau[atk == 0], 70)
    n_tot = int(norm_elev.sum())
    if n_tot > 0:
        n_hover = int((norm_elev & (act == 1)).sum())
        n_maneu = int((norm_elev & (act == 0) & (tau >= tau_hi)).sum())
        n_other = n_tot - n_hover - n_maneu
        print(f"  정상상승 총 {n_tot}스텝 중:")
        print(f"    ③ hover(action=1)        : {n_hover:5d} ({100*n_hover/n_tot:.0f}%) ← a_t-1이 커버")
        print(f"    ② 급기동(track+높은토크)  : {n_maneu:5d} ({100*n_maneu/n_tot:.0f}%) ← 관측에 단서 부족")
        print(f"    기타                      : {n_other:5d} ({100*n_other/n_tot:.0f}%)")
        # NIS-토크 상관 (급기동 기여도)
        m = (atk == 0)
        corr = np.corrcoef(nv[m], tau[m])[0, 1]
        print(f"  평시 NIS↔명령토크 상관={corr:+.2f} (높을수록 급기동이 NIS를 올림=②비중↑)")

    # ── [D] 공격 강도별 탐지율 (recall 천장) ──
    print("\n"+"="*78)
    print("[D] 공격 강도별 탐지율 (= 어느 강도부터 잡히나, recall 천장 진단)")
    print("="*78)
    thr_det = p90                       # 단순 임계: NIS>평시p90 이면 '탐지'로 간주
    atk_mask = atk == 1
    if atk_mask.sum() > 0:
        # 공격 강도 프록시 = 그 스텝의 명령토크편차는 부적절 → bias가 zu_log에 없음.
        # 대신 '공격 중 NIS 자체'를 강도 프록시로 못 씀. → 공격구간을 NIS크기 5분위로 나눠 탐지율.
        nv_atk = nv[atk_mask]
        qs = np.percentile(nv_atk, [20, 40, 60, 80])
        bins = np.digitize(nv_atk, qs)
        print(f"  (공격 강도 직접값이 zu_log에 없어, 공격구간 NIS 5분위로 대리)")
        for bidx in range(5):
            seg = nv_atk[bins == bidx]
            if len(seg):
                det = float(np.mean(seg > thr_det)*100)
                print(f"    NIS 5분위 {bidx+1}: n={len(seg):5d}  탐지율(>{thr_det:.2f})={det:.0f}%")
        overall = float(np.mean(nv_atk > thr_det)*100)
        print(f"  전체 공격 탐지율(>평시p90) = {overall:.0f}%  ← 단순임계 recall 상한")
        print(f"  주: 강도 직접진단은 zu_log에 bias_thrust/torque 로깅 추가하면 정확해짐.")

    # ── 종합 판정 ──
    print("\n"+"="*78)
    print("종합 판정")
    print("="*78)
    verdict_sep = "✅ 분리 가능(환경 OK)" if d_mv >= 1.8 else ("⚠ 경계(어려움)" if d_mv >= 1.2 else "❌ 분리 어려움")
    verdict_win = "✅ 윈도우가 정상상승 흡수" if p_norm >= 70 and (100-p_atk) >= 50 else "⚠ 윈도우 효과 제한적"
    print(f"  • RL 입력 분리도(다변량 d'={d_mv:.2f}): {verdict_sep}")
    print(f"  • 윈도우 구별({win}스텝): {verdict_win}")
    print(f"  • per-step({d_ps_v:.2f}) → 윈도우({d_w_v:.2f}) → 다변량({d_mv:.2f}) 상승폭이 클수록 RL에 유리")


if __name__ == '__main__':
    main()
