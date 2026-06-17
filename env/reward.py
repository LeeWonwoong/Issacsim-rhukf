"""
reward.py — 공격 탐지 보상 설계 (Quickest-Change-Detection 형)
=========================================================
목표: (1) 검출지연↓  (2) 복구지연↓  (3) 오탐(FP)↓  를 동시에.
행동 = 선언:  0 = track(정상 선언)   1 = hover(공격 선언)

핵심 구조 (윈도우=4의 '지속성'을 활용):
    공격 중 (is_under_attack=True):
        hover(1) → TP            : +r_tp
        track(0) → FN(미탐):
            attack_delay < grace_detect → fn_grace  (★ onset 유예: 1~2스텝 '지속성 확인' 허용
                                                       → 기동 스파이크 transient를 거름 = FP↓)
            그 외                        → fn_base - fn_escalation·(d)^p  (지연↑=페널티 급증 → 빠른 검출)
    평시 (is_under_attack=False):
        track(0) → TN            : +r_tn
        hover(1) → FP(오탐):
            recovery_delay < grace_recover → fp_grace (★ offset 유예: 공격 직후 NIS 식는 1~2스텝 확인)
            그 외                          → fp_base - fp_escalation·(d)^p (지속↑ → 빠른 복귀/순수오탐 억제)
    crash terminal: env done 로직에서 terminal_penalty 가산 (물리 추락 anchor).

튜닝 다이얼 (FP vs 지연 trade-off; 둘 다 0 불가, NIS 분리도가 한계):
    grace_detect↑ → onset 더 오래 확인 → FP↓·검출지연↑
    fn_escalation↑/escalation_power↑ → 검출 더 공격적 → FP↑ 위험
    grace_recover/fp_escalation → 복구쪽 동일.
주의: 빠른 '복구 선언'은 NIS가 실제로 빨리 식어야 가능 — 필터(low Q, ff=1.0)면 복구가 필터단 병목.
"""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # ── 정탐 / 정상 보상 ──
    r_tp: float = 0.7
    r_tn: float = 0.7

    # ── 미탐(FN): fn_base - fn_escalation·(attack_delay-grace_detect)^power ──
    fn_base: float = -1.0
    fn_escalation: float = 0.2
    grace_detect: int = 1        # onset 유예(지속성 확인). 0=즉시검출 강제(FP↑). 1~2 권장.
    fn_grace: float = -0.1       # 유예 구간 FN(거의 무료)

    # ── 오탐(FP): fp_base - fp_escalation·(recovery_delay-grace_recover)^power ──
    fp_base: float = -1.0
    fp_escalation: float = 0.2
    grace_recover: int = 1       # offset 유예(NIS 식는 동안 확인)
    fp_grace: float = -0.1

    escalation_power: float = 2.0
    delay_cap: int = 5               # 에스컬레이션 포화 지점(분산폭주 차단). delay≥cap → 동일 바닥
    terminal_penalty: float = -4.0   # flip/altitude 물리추락에만


DEFAULT_REWARD = RewardConfig()


def calculate_reward(current_action, is_under_attack,
                     attack_delay: int = 0, recovery_delay: int = 0,
                     rc: RewardConfig = None) -> float:
    """
    current_action:  0=track, 1=hover
    is_under_attack: 현재 스텝 공격 활성 여부 (지면 진실; 관측엔 없음)
    attack_delay:    공격 onset 후 경과 스텝 (FN 에스컬레이션·onset grace)
    recovery_delay:  공격 offset 후 경과 스텝 (FP 에스컬레이션·offset grace).
                     순수 FP(공격 이력 없음)는 호출부가 큰 값(grace 넘김)으로 전달 → 내부 cap에서 포화.
    rc:              RewardConfig (None이면 DEFAULT_REWARD)
    """
    rc = rc if rc is not None else DEFAULT_REWARD

    if is_under_attack:
        if current_action == 1:                       # TP
            return rc.r_tp
        d = min(attack_delay, rc.delay_cap) - rc.grace_detect   # FN: onset grace → 에스컬레이션(cap)
        if d < 0:
            return rc.fn_grace
        return rc.fn_base - rc.fn_escalation * (d ** rc.escalation_power)
    else:
        if current_action == 0:                       # TN
            return rc.r_tn
        d = min(recovery_delay, rc.delay_cap) - rc.grace_recover  # FP: offset grace → 에스컬레이션(cap)
        if d < 0:
            return rc.fp_grace
        return rc.fp_base - rc.fp_escalation * (d ** rc.escalation_power)
