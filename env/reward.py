from dataclasses import dataclass


@dataclass
class RewardConfig:
    # ── 정탐 / 정상 (TP ≫ TN: 공격을 잡는 게 명확히 이득 → 딜레이↓) ──
    r_tp: float = 2.0          # 공격중 hover
    r_tn: float = 0.5          # 평시 track

    # ── 미탐(FN) = 딜레이 최소화: onset부터 선형 즉발 (fn_base + fn_per_step·delay) ──
    #   로드맵 학습수정(2026-07-08) fn 온셋가중 [강화 2026-07-08]: 온셋 펄스(t=1~2 뒤 침묵, 물리#2)라
    #   초반 탐지 필수. 이전(선형 cap4: d0 -0.8/d1 -1.5/…/d4+ -3.6)은 d≥4 평탄→둘째봉(14-15) 방치·조기유인 약함.
    #   → 온셋창(1≤d≤fn_onset_window) 미탐을 ×fn_onset_mult 가중 + cap 4→5.
    #   신곡선: d0 -0.80 / d1 -2.70 / d2 -3.96 / d3 -5.22 / d4 -6.48 / d5 -7.74 (단조·급).
    fn_base: float = -0.8      # delay 0 (온셋 즉시 유의미한 손해 — obs창 미충전 grace)
    fn_per_step: float = -0.7  # delay당 추가 (선형 → 미루면 바로 아픔)
    fn_onset_mult: float = 1.8    # 온셋창 미탐 가중배수(1.5~2) = 조기탐지 직접 유인(첫봉 6-7→3-4 당김)
    fn_onset_window: int = 5      # onset 후 d≤이 스텝까지 가중 적용

    # ── 오탐(FP) = 오탐 최소화: 평시 hover 첫 스텝부터 강하게 ──
    fp_base: float = -2.0      # fp_run 0 (명확히 손해)
    fp_per_step: float = -1.0  # 연속 오탐당 추가

    delay_cap: int = 5         # 4→5: 데드라인창 gradient 유지(평탄화=둘째봉 방치 방지)
    terminal_penalty: float = -10.0   # flip/altitude 물리추락 (γ=0.9라 결정에 닿음)

    # ── heavy-tailed 보상 노이즈 (옵티마이저 강건성 실험 KNOB; 기본 OFF) ──
    #    버퍼 저장 reward에만 가산(=칼만 measurement noise 채널). zero-mean mixture.
    #    §6 강건성 스윕: reward_noise_outlier_sigma를 0,5,10,20으로 쓸며 RHUKF−Adam 이점 측정.
    reward_noise_enabled: bool = False
    reward_noise_sigma: float = 1.0          # 평상 가우시안 std (≈R 자릿수)
    reward_noise_outlier_prob: float = 0.05  # outlier 발생 확률
    reward_noise_outlier_sigma: float = 10.0 # outlier std (heavy-tail 세기 = 주 다이얼)


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
                     rc: RewardConfig = None) -> float:
    """
    current_action:  0=track, 1=hover
    is_under_attack: 현재 스텝 공격 활성 여부 (지면 진실; 관측엔 없음)
    attack_delay:    공격 onset 후 경과 스텝 (FN 선형 에스컬레이션)
    fp_run:          평시 연속 오탐(hover) 지속 길이 (FP 선형 에스컬레이션).
                     호출부의 continuous_fp_count(또는 recovery_delay)를 그대로 전달하면 됨.
    rc:              RewardConfig (None이면 DEFAULT_REWARD)
    """
    rc = rc if rc is not None else DEFAULT_REWARD
    if is_under_attack:
        if current_action == 1:                                  # TP
            return rc.r_tp
        d = min(max(attack_delay, 0), rc.delay_cap)              # FN: 딜레이 선형
        pen = rc.fn_base + rc.fn_per_step * d
        if 1 <= d <= rc.fn_onset_window:                         # 온셋창 가중 = 조기탐지 직접 유인
            pen *= rc.fn_onset_mult
        return pen
    else:
        if current_action == 0:                                  # TN
            return rc.r_tn
        d = min(max(fp_run, 0), rc.delay_cap)                    # FP: 연속오탐 선형
        return rc.fp_base + rc.fp_per_step * d
