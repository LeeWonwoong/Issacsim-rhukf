"""
reward.py — 공격 탐지 보상 설계 (분리된 설정 모듈)
=========================================================
보상 파라미터를 RewardConfig 한 곳에 모아, 보상 튜닝을 환경/학습 로직과 분리한다.
calculate_reward()는 기존 호출부와 100% 호환 (rc 미지정 시 모듈 기본값 사용).

행동 정의:
    0 = 궤도 추종 (track)
    1 = 강제 호버링 (hover)

판정 / 보상:
    공격 중 (is_under_attack=True):
        hover(1) → TP (정탐)   : +r_tp
        track(0) → FN (미탐)   : fn_base - fn_escalation * consecutive_fn**escalation_power
    평시 (is_under_attack=False):
        track(0) → TN (정상)   : +r_tn
        hover(1) → FP (오탐)   : fp_base - fp_escalation * consecutive_fp**escalation_power

terminal_penalty 는 env 의 done(논리적/물리적 종료) 로직에서 사용한다.
"""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # ── 정탐 / 정상 보상 ──
    r_tp: float = 0.7            # 공격 중 hover (정탐)
    r_tn: float = 0.7            # 평시 track (정상)

    # ── 미탐(FN) 페널티 ── fn_base - fn_escalation * consecutive_fn**escalation_power
    fn_base: float = -1.0
    fn_escalation: float = 0.2

    # ── 오탐(FP) 페널티 ── fp_base - fp_escalation * consecutive_fp**escalation_power
    fp_base: float = -1.0
    fp_escalation: float = 0.2

    # ── 연속 카운트 가중 차수 ──
    escalation_power: float = 2.0

    # ── 종료 페널티 (env done 로직 전용) ──
    terminal_penalty: float = -7.0


# 모듈 기본 인스턴스 (rc 미지정 호출 호환용)
DEFAULT_REWARD = RewardConfig()


def calculate_reward(current_action, is_under_attack,
                     consecutive_fn: int = 0, consecutive_fp: int = 0,
                     rc: RewardConfig = None) -> float:
    """
    Args:
        current_action:   0=track, 1=hover
        is_under_attack:  현재 스텝 공격 활성 여부
        consecutive_fn:   연속 미탐 카운트 (공격 중 track 지속)
        consecutive_fp:   연속 오탐 카운트 (평시 hover 지속)
        rc:               RewardConfig (None이면 DEFAULT_REWARD)

    Returns:
        스칼라 보상값
    """
    rc = rc if rc is not None else DEFAULT_REWARD

    if is_under_attack:
        if current_action == 1:
            return rc.r_tp
        return rc.fn_base - rc.fn_escalation * (consecutive_fn ** rc.escalation_power)
    else:
        if current_action == 0:
            return rc.r_tn
        return rc.fp_base - rc.fp_escalation * (consecutive_fp ** rc.escalation_power)
