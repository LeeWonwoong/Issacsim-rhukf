"""
baselines_detect.py — 임계값 기반 탐지기(χ²/CUSUM/Euclidean) vs RL 정책 비교
====================================================================
RL 에이전트(학습된 정책)와 '고정 임계값' 탐지기들을 같은 NIS 스트림에서 비교한다.
임계값 탐지기는 임계값을 쓸며 ROC(검출지연 vs 오경보율)를 그리고,
RL 정책은 하나의 운용점(delay, FAR)을 준다. RL이 그 frontier보다 좋으면(아래/왼쪽) 승.

입력 CSV (스텝단위, 에피소드로 그룹): 다음 컬럼 필요
  episode 식별: 'cell_idx'(+'episode')  또는  'ep'
  'attack_active' (0/1), 'nis_v_raw', 'nis_g_raw'
  (선택) 'resid_norm' : 원 innovation ‖r‖ → Euclidean용(없으면 Euclidean 생략)
  (선택) 'action'     : 0=track,1=hover → RL/정책 운용점 계산

사용:
  python baselines_detect.py results_xxx/sweep_detail.csv         # fixed-policy 스트림으로 frontier
  python baselines_detect.py eval_rollouts.csv                    # 학습된 RL eval 로그(action 포함)면 RL점도 같이

탐지기:
  chi2     : 채널 NIS(이미 χ² 통계) > thr 이면 경보  (모델기반 FDI 표준; 공분산 정규화됨)
  cusum    : S=max(0,S+(NIS-k)); S>h 면 경보         (순차 변화탐지; 지속이동에 강함)
  euclid   : ‖r‖ > thr 이면 경보                      (원 잔차; 공분산 정규화 '없음' → χ²와 대비)
"""
import sys, csv
from collections import defaultdict
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    # 에피소드 키
    def epkey(r):
        if 'cell_idx' in r:
            return (r.get('cell_idx', '0'), r.get('policy', ''), r.get('episode', '0'))
        return (r.get('ep', '0'),)
    eps = defaultdict(list)
    for r in rows:
        eps[epkey(r)].append(r)
    # 각 에피소드: 시간순 정렬 + onset
    out = []
    for k, rs in eps.items():
        rs.sort(key=lambda r: int(r.get('step', 0)))
        atk = np.array([int(r['attack_active']) for r in rs])
        g = np.array([float(r['nis_g_raw']) for r in rs])
        v = np.array([float(r.get('nis_v_raw', 0)) for r in rs])
        rn = np.array([float(r['resid_norm']) for r in rs]) if 'resid_norm' in rs[0] else None
        act = np.array([int(r['action']) for r in rs]) if 'action' in rs[0] else None
        onset = int(np.argmax(atk)) if atk.any() else None
        out.append(dict(atk=atk, g=g, v=v, rn=rn, act=act, onset=onset))
    return out


def eval_detector(eps, statfn, thr, cusum=False, k=None):
    """각 임계값에서 (평균 검출지연, FAR). statfn(ep)->1D 통계 시계열."""
    delays, fa_eps, atk_eps, base_eps = [], 0, 0, 0
    for ep in eps:
        x = statfn(ep)
        if x is None:
            return None
        # 경보 시점
        alarm = None
        if cusum:
            S = 0.0
            for t, xt in enumerate(x):
                S = max(0.0, S + (xt - k))
                if S > thr:
                    alarm = t; break
        else:
            hit = np.where(x > thr)[0]
            alarm = int(hit[0]) if len(hit) else None
        if ep['onset'] is None:          # 무공격 에피소드 → FAR
            base_eps += 1
            if alarm is not None:
                fa_eps += 1
        else:                            # 공격 에피소드 → 검출지연
            atk_eps += 1
            if alarm is not None and alarm >= ep['onset']:
                delays.append(alarm - ep['onset'])
    far = fa_eps / max(base_eps, 1)
    mdelay = float(np.mean(delays)) if delays else float('nan')
    detect_rate = len(delays) / max(atk_eps, 1)
    return mdelay, far, detect_rate


def roc(eps, statfn, thrs, cusum=False, k=None):
    pts = []
    for thr in thrs:
        r = eval_detector(eps, statfn, thr, cusum=cusum, k=k)
        if r is None:
            return None
        pts.append((thr, *r))
    return pts


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'results_combined/sweep_detail.csv'
    eps = load(path)
    print(f"\n[입력] {path} | 에피소드 {len(eps)}개 "
          f"(공격 {sum(e['onset'] is not None for e in eps)}, 무공격 {sum(e['onset'] is None for e in eps)})")

    # 통계 정의
    stat_g = lambda e: e['g']                       # gyro NIS (χ²)
    stat_comb = lambda e: e['g'] + e['v']           # vel+gyr 합 NIS
    stat_rn = lambda e: e['rn']                      # 원 잔차(Euclidean)

    # baseline 무공격 분포로 k 추정(CUSUM)
    base_g = np.concatenate([e['g'][e['atk'] == 0] for e in eps if (e['atk'] == 0).any()])
    k_cusum = float(base_g.mean() + base_g.std())

    print("\n=== χ² (gyro NIS 임계) ROC : thr | delay | FAR | detect% ===")
    for thr, d, fa, dr in roc(eps, stat_g, [3, 5, 8, 12, 20, 40, 80]):
        print(f"  thr={thr:6.1f} | delay={d:6.1f} | FAR={fa:.2f} | det={dr*100:4.0f}%")

    print(f"\n=== CUSUM (gyro NIS) ROC : h | delay | FAR | detect%  (k={k_cusum:.2f}) ===")
    for thr, d, fa, dr in roc(eps, stat_g, [10, 20, 40, 60, 100, 150], cusum=True, k=k_cusum):
        print(f"  h={thr:6.1f} | delay={d:6.1f} | FAR={fa:.2f} | det={dr*100:4.0f}%")

    if eps[0]['rn'] is not None:
        print("\n=== Euclidean (‖r‖ 임계) ROC : thr | delay | FAR | detect% ===")
        thrs = np.percentile(np.concatenate([e['rn'] for e in eps]), [50, 70, 85, 95, 99]).tolist()
        for thr, d, fa, dr in roc(eps, stat_rn, thrs):
            print(f"  thr={thr:6.2f} | delay={d:6.1f} | FAR={fa:.2f} | det={dr*100:4.0f}%")
    else:
        print("\n[Euclidean] 'resid_norm' 컬럼 없음 → 생략. (UKF 원 잔차 ‖last_res‖를 로깅하면 활성화)")

    # RL/정책 운용점 (action 컬럼이 있으면)
    if eps[0]['act'] is not None:
        delays, fa_eps, atk_eps, base_eps = [], 0, 0, 0
        for e in eps:
            hov = np.where(e['act'] == 1)[0]
            if e['onset'] is None:
                base_eps += 1
                # 무공격인데 hover하면 FP
                if len(hov):
                    fa_eps += 1
            else:
                atk_eps += 1
                after = hov[hov >= e['onset']]
                if len(after):
                    delays.append(int(after[0]) - e['onset'])
        far = fa_eps / max(base_eps, 1)
        md = float(np.mean(delays)) if delays else float('nan')
        print("\n=== ★ 정책(action=hover) 운용점 ===")
        print(f"  delay={md:.1f} | FAR={far:.2f} | det={len(delays)/max(atk_eps,1)*100:.0f}%")
        print("  → 이 (delay,FAR) 점이 위 ROC frontier보다 왼쪽-아래면 정책이 임계탐지보다 우월.")
    else:
        print("\n[정책점] 'action' 컬럼 없음 → 학습된 RL eval 로그(action 포함)로 다시 실행하면 RL점도 표시.")
    print()


if __name__ == '__main__':
    main()
