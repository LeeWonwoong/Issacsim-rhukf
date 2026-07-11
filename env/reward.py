from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RewardConfig:
    # ── 정탐 / 정상 (TP ≫ TN: 공격을 잡는 게 명확히 이득 → 딜레이↓) ──
    r_tp: float = 2.0          # 공격중 hover
    r_tn: float = 0.5          # 평시 track

    # ══════════════════════════════════════════════════════════════════
    #  미탐(FN) — 펄스 reward 대수술 (2026-07-11): piecewise 단조 곡선
    # ══════════════════════════════════════════════════════════════════
    #  배경(확정 물리): NIS 펄스 t=2~3 gyro 급상승(raw~3.15, 236배) → t=4~9 침묵(innovation 골,
    #    필터로 못채움 확정) → t=10~11 재상승. 데드라인 상단(s1.40) d≤7 / 하단(s1.34) d≤14.
    #  병목 = reward 유인(오라클 delay1 가능). 구(舊) fn_onset_mult(선형×배수)는 d≥5 평탄 →
    #    빨리 잡을 이유 없어 delay≈9 안착(이봉). → 곱셈 폐기, delay 인덱스 lookup으로 단조 재작성.
    #  형태: d0~1 grace(신호 물리적으로 無 — 벌하면 왜곡) / d2~3 최급(첫 펄스=골든타임=목표지점)
    #        d4~7 계속 상승(침묵이라 못잡아도 늦을수록 벌, 데드라인7) / d≥8 cap(비평탄은 per-step
    #        누적으로 보장 — 매 미탐스텝마다 -16 가산되어 미루면 총합 계속 하락).
    #  제약(불변식): 단조증가(인버전 없음) ∧ d2~3 최급 ∧ d≥8 per-step 최대.
    fn_curve: Tuple[float, ...] = (
        -0.3,   # d0  grace (obs창 미충전 + 신호 없음)
        -0.5,   # d1  grace (첫 펄스 직전, 탐지율 물리적 0%)
        -3.0,   # d2  ★첫 펄스 — 골든타임 진입 (기울기 -2.5, 급증)
        -6.0,   # d3  ★펄스 정점 — 최급 기울기 (-3.0), 목표 지점
        -8.0,   # d4  침묵 진입 (-2.0)
        -10.0,  # d5  침묵 (-2.0)
        -12.0,  # d6  침묵 (-2.0)
        -14.0,  # d7  데드라인 상단(s1.40) (-2.0)
        -16.0,  # d8+ cap (데드라인 초과 압박; per-step 누적으로 비평탄 유지)
    )
    delay_cap: int = 8         # 5→8: 데드라인 상단7 덮음. fn_curve 길이 = delay_cap+1 이어야 함.

    # ── 오탐(FP) = 평시 hover 첫 스텝부터 강하게 ──
    fp_base: float = -2.0      # fp_run 0 (명확히 손해)
    fp_per_step: float = -1.0  # 연속 오탐당 추가

    # ── FP 완화: NIS 스파이크 직후 grace (첫 펄스 과감히) ──
    #    첫 펄스 raw~3.15(236배)로 명확 → 그 직후 반응은 FP여도 정당 → 벌 완화(-2.0→-1.0).
    #    '신호 있을 때만': 스파이크 감지(raw gyro NIS ≥ spike_nis_threshold) 후 N스텝만.
    #    평시(무신호) FP 는 fp_base -2.0 그대로 유지 → '일단 다 쳐' 과잉교정 방지.
    fp_spike_grace_steps: int = 2       # 스파이크 후 완화 지속 스텝
    fp_spike_grace_base: float = -1.0   # grace 중 FP 페널티(평탄; per-step 에스컬레이션 미적용)
    spike_nis_threshold: float = 1.5    # grace 무장 임계 (raw gyro NIS; 정상 p99≈0.73 ≪ 1.5 ≪ 펄스 3.15)

    terminal_penalty: float = -10.0     # flip/altitude 물리추락 (γ=0.9라 결정에 닿음)

    # ── heavy-tailed 보상 노이즈 (옵티마이저 강건성 실험 KNOB; 기본 OFF) ──
    #    버퍼 저장 reward에만 가산(=칼만 measurement noise 채널). zero-mean mixture.
    #    §6 강건성 스윕: reward_noise_outlier_sigma를 0,5,10,20으로 쓸며 RHUKF−Adam 이점 측정.
    reward_noise_enabled: bool = False
    reward_noise_sigma: float = 1.0          # 평상 가우시안 std (≈R 자릿수)
    reward_noise_outlier_prob: float = 0.05  # outlier 발생 확률
    reward_noise_outlier_sigma: float = 10.0 # outlier std (heavy-tail 세기 = 주 다이얼)

    def __post_init__(self):
        # fn_curve 는 delay_cap 을 덮어야 함 (0..delay_cap 인덱싱).
        assert len(self.fn_curve) >= self.delay_cap + 1, \
            f"fn_curve(len={len(self.fn_curve)}) < delay_cap+1({self.delay_cap+1})"
        # 단조증가(=값이 단조감소) 불변식 — 인버전 방지.
        assert all(self.fn_curve[i] >= self.fn_curve[i + 1] for i in range(len(self.fn_curve) - 1)), \
            f"fn_curve 비단조: {self.fn_curve}"


DEFAULT_REWARD = RewardConfig()


def sample_reward_noise(rc: RewardConfig = None) -> float:
    """heavy-tailed 보상 노이즈 1샘플 (zero-mean mixture). 비활성/미설정이면 0.0.
       버퍼 저장 reward에만 가산 → 칼만 measurement noise 채널을 직접 자극."""
    import numpy as np
    rc = rc if rc is not None else DEFAULT_REWARD
    if not getattr(rc, 'reward_noise_enabled', False):
        return 0.0
    if np.random.rand() < rc.reward_noise_outlier_prob:
        return float(np.random.randn() * rc.reward_noise_outlier_sigma)   # outlier(꼬리)
    return float(np.random.randn() * rc.reward_noise_sigma)               # 평상


def calculate_reward(current_action, is_under_attack,
                     attack_delay: int = 0, fp_run: int = 0,
                     fp_grace: bool = False,
                     rc: RewardConfig = None) -> float:
    """
    current_action:  0=track, 1=hover
    is_under_attack: 현재 스텝 공격 활성 여부 (지면 진실; 관측엔 없음)
    attack_delay:    공격 onset 후 경과 스텝 (FN piecewise 곡선 인덱스)
    fp_run:          평시 연속 오탐(hover) 지속 길이 (FP 선형 에스컬레이션)
    fp_grace:        NIS 스파이크 직후 grace 창 여부(True면 FP 페널티 완화; 호출부가 판정)
    rc:              RewardConfig (None이면 DEFAULT_REWARD)
    """
    rc = rc if rc is not None else DEFAULT_REWARD
    if is_under_attack:
        if current_action == 1:                                  # TP
            return rc.r_tp
        d = min(max(attack_delay, 0), rc.delay_cap)              # FN: piecewise 단조 곡선
        return rc.fn_curve[d]
    else:
        if current_action == 0:                                  # TN
            return rc.r_tn
        if fp_grace:                                             # 스파이크 직후 = 과감한 반응 허용
            return rc.fp_spike_grace_base                        # 평탄 완화(-1.0)
        d = min(max(fp_run, 0), rc.delay_cap)                    # FP: 연속오탐 선형
        return rc.fp_base + rc.fp_per_step * d
