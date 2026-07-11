from dataclasses import dataclass
from typing import Tuple


@dataclass
class RewardConfig:
    # ══════════════════════════════════════════════════════════════════
    #  펄스 reward 재조정 v2 (2026-07-12): 급경사 FN 역효과 → 완만 FN + 조기탐지는 TP(상)로
    # ══════════════════════════════════════════════════════════════════
    #  v1(급경사 FN cap-16 + FP grace) 재학습 실패: delay 7→9(악화)·둘째봉↑, FAR 0.004→0.099.
    #  진단: FP grace가 FAR 주범(공격ep transient arm→값싼 hover), 급경사 FN의 '놓침공포'가
    #        오히려 신중함(늦은탐지) 유발. d≤3 basin은 존재(전학습중 11회 달성).
    #  v2 처방: ① FP grace 완전삭제(항상 -2.0) ② FN 완만 단조(-1 기울기, 평탄 제거)
    #           ③ 조기탐지를 벌(FN급경사)이 아니라 상(TP 증액)으로 유도.

    # ── 정탐(TP) = 조기탐지 상: d≤3 크게 보상, 늦을수록 감소(→ 빨리 잡으면 이득) ──
    #   매 hover스텝마다 부여되므로 d=2부터 hold하면 [4.5,4.5,4.0,…] 누적 ≫ d=10부터 [2.0,…].
    r_tp_curve: Tuple[float, ...] = (
        4.5,  # d0
        4.5,  # d1
        4.5,  # d2  ★첫 펄스 골든타임 — 조기정탐 최대보상
        4.5,  # d3  ★목표 데드라인 (d≤3 전부 4.5)
        4.0,  # d4
        3.5,  # d5
        3.0,  # d6
        2.5,  # d7
        2.0,  # d8+ 늦은정탐 = 구 r_tp(2.0) 바닥
    )
    r_tp_cap: int = 8
    r_tp: float = 2.0          # (하위호환/폴백; 실제는 r_tp_curve 사용)
    r_tn: float = 0.5          # 평시 track

    # ── 미탐(FN) = 완만 단조 (급경사 아님; 평탄 제거) ──
    #   기울기 -1/step. d0~1 grace(신호 물리적으로 無). d2~ 완만 하락. 데드라인(하단 d≤14)까지 단조,
    #   그 뒤 cap(회복불가 구간이라 구분 무의미). '놓침공포' 대신 완만 — 조기유인은 TP가 담당.
    fn_curve: Tuple[float, ...] = (
        -0.3, -0.5,                                        # d0,d1 grace
        -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,          # d2..d8
        -9.0, -10.0, -11.0, -12.0, -13.0, -14.0,           # d9..d14 (평탄 제거, 단조 지속)
    )
    delay_cap: int = 14        # FN 곡선 인덱스 상한(하단밴드 데드라인). fn_curve 길이 = delay_cap+1.

    # ── 오탐(FP) = 평시 hover 첫 스텝부터 강하게. grace 없음(항상 -2.0부터). ──
    fp_base: float = -2.0      # fp_run 0 (명확히 손해)
    fp_per_step: float = -1.0  # 연속 오탐당 추가
    fp_cap: int = 5            # FP 에스컬레이션/recovery grace 클램프 (FN delay_cap과 분리)

    terminal_penalty: float = -10.0     # flip/altitude 물리추락 (γ=0.9라 결정에 닿음)

    # ── heavy-tailed 보상 노이즈 (옵티마이저 강건성 실험 KNOB; 기본 OFF) ──
    #    버퍼 저장 reward에만 가산(=칼만 measurement noise 채널). zero-mean mixture.
    reward_noise_enabled: bool = False
    reward_noise_sigma: float = 1.0
    reward_noise_outlier_prob: float = 0.05
    reward_noise_outlier_sigma: float = 10.0

    def __post_init__(self):
        assert len(self.fn_curve) >= self.delay_cap + 1, \
            f"fn_curve(len={len(self.fn_curve)}) < delay_cap+1({self.delay_cap+1})"
        assert all(self.fn_curve[i] >= self.fn_curve[i + 1] for i in range(len(self.fn_curve) - 1)), \
            f"fn_curve 비단조: {self.fn_curve}"
        assert len(self.r_tp_curve) >= self.r_tp_cap + 1, \
            f"r_tp_curve(len={len(self.r_tp_curve)}) < r_tp_cap+1({self.r_tp_cap+1})"
        # 조기탐지 상: 단조 비증가(늦을수록 보상↓) + 바닥이 r_tn보다 커야 TP 유인 유지.
        assert all(self.r_tp_curve[i] >= self.r_tp_curve[i + 1] for i in range(len(self.r_tp_curve) - 1)), \
            f"r_tp_curve 비단조(증가): {self.r_tp_curve}"


DEFAULT_REWARD = RewardConfig()


def sample_reward_noise(rc: RewardConfig = None) -> float:
    """heavy-tailed 보상 노이즈 1샘플 (zero-mean mixture). 비활성/미설정이면 0.0."""
    import numpy as np
    rc = rc if rc is not None else DEFAULT_REWARD
    if not getattr(rc, 'reward_noise_enabled', False):
        return 0.0
    if np.random.rand() < rc.reward_noise_outlier_prob:
        return float(np.random.randn() * rc.reward_noise_outlier_sigma)
    return float(np.random.randn() * rc.reward_noise_sigma)


def calculate_reward(current_action, is_under_attack,
                     attack_delay: int = 0, fp_run: int = 0,
                     rc: RewardConfig = None) -> float:
    """
    current_action:  0=track, 1=hover
    is_under_attack: 현재 스텝 공격 활성 여부 (지면 진실; 관측엔 없음)
    attack_delay:    공격 onset 후 경과 스텝 (TP 조기상 곡선 + FN 완만 곡선 인덱스)
    fp_run:          평시 연속 오탐(hover) 지속 길이 (FP 선형 에스컬레이션)
    rc:              RewardConfig (None이면 DEFAULT_REWARD)
    """
    rc = rc if rc is not None else DEFAULT_REWARD
    if is_under_attack:
        if current_action == 1:                                  # TP: 조기탐지 상(곡선)
            d = min(max(attack_delay, 0), rc.r_tp_cap)
            return rc.r_tp_curve[d]
        d = min(max(attack_delay, 0), rc.delay_cap)              # FN: 완만 단조 곡선
        return rc.fn_curve[d]
    else:
        if current_action == 0:                                  # TN
            return rc.r_tn
        d = min(max(fp_run, 0), rc.fp_cap)                       # FP: 연속오탐 선형 (grace 없음)
        return rc.fp_base + rc.fp_per_step * d
