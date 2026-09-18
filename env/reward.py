import os
from dataclasses import dataclass

_FP_ESCAL = os.environ.get('FP_ESCAL', '') == '1'


@dataclass
class RewardConfig:
    # ════════════════════════════════════════════════════════════════════
    #  ★ 2026-08-19 v3: FN 딜레이 그래디언트 복원 (v2 상수는 delay 2.0→3.78 악화·데드라인 초과).
    #    TP/TN/FP/terminal 은 v2 그대로. FN 만 "구 온셋가중 곡선의 절반"으로 =
    #      빠른 탐지 유인 부활 + max|r|≈3.87(≈huber_c) 유지(RHUKF 정합·TD 꼬리 낮음).
    #    FP 는 v2 상수 -1 유지(FP는 이미 0.002로 문제 아니었음 → 단순 유지).
    # ════════════════════════════════════════════════════════════════════
    r_tp: float = 1.0          # ★2026-08-25 0.5→1.0: 탐지강조(recall/F1↑·지연↓). RHUKF 보수화 해소
    r_tn: float = 0.5          # 평시 track   (정상)
    r_fp: float = -0.7         # ★2026-08-25 -1.0→-0.7: FP 덜harsh → 탐지 적극화(recall↑)
    # ★2026-09-09 terminal_penalty 재도입(A/B 중): γ0.85 절단 손실(3.3) < FN 1스텝(3.87) 이라
    #   "사망이 한 스텝 미탐보다 싼" 서열 역전 발견 → -4.0 = max|r| 바로 위(사망>최악FN 최소값).
    #   env TERMINAL_PEN 으로 조절 (0=기존 CartPole형 유지).
    terminal_penalty: float = float(__import__('os').environ.get('TERMINAL_PEN', '0') or 0.0)
    # ★2026-09-10 relapse 패널티(A/B 중): 공격 중 hover→track 재발 스텝에 FN 벌점 위에 추가.
    #   약공격(δ<0.4)에서 PX4 보상으로 잔차가 가라앉아 정책이 track 으로 되돌아가는 플리커 억제용. env RELAPSE_PEN (0=없음).
    relapse_penalty: float = float(__import__('os').environ.get('RELAPSE_PEN', '0') or 0.0)

    # ── FN(미탐) 딜레이 에스컬레이션 = 조기탐지 직접 유인. 구곡선의 절반(2026-08-19). ──
    #   신곡선 d0 -0.40 / d1 -1.35 / d2 -1.98 / d3 -2.61 / d4 -3.24 / d5 -3.87 (단조·온셋가중)
    #   (구 v1: -0.80/-2.70/-3.96/-5.22/-6.48/-7.74 = 이것의 2배, TD 꼬리 9.8이라 폐기)
    fn_base: float = -0.4         # delay 0
    fn_per_step: float = -0.35    # delay당 추가
    fn_onset_mult: float = float(__import__('os').environ.get('FN_ONSET_MULT', '1.8') or 1.8)   # ★09-16 env화(1.0 = 온셋 가중 끔 → FN 상수화)
    fn_onset_window: int = 5
    delay_cap: int = 5

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
                     rc: RewardConfig = None, relapse: bool = False) -> float:
    """
    current_action:  0=track, 1=hover
    is_under_attack: 현재 스텝 공격 활성 여부 (지면 진실; 관측엔 없음)
    attack_delay:    공격 onset 후 경과 스텝 (FN 선형 에스컬레이션)
    fp_run:          평시 연속 오탐(hover) 지속 길이 (FP 선형 에스컬레이션).
                     호출부의 continuous_fp_count(또는 recovery_delay)를 그대로 전달하면 됨.
    rc:              RewardConfig (None이면 DEFAULT_REWARD)
    relapse:         이 스텝이 공격 중 hover→track 재발(직전 행동 1, 현 행동 0)이면 True → rc.relapse_penalty 가산
    """
    rc = rc if rc is not None else DEFAULT_REWARD
    #  ★ 2026-08-19 v3: FN 만 딜레이 에스컬레이션, FP 는 상수. fp_run 인자 미사용.
    if is_under_attack:
        if current_action == 1:                                  # TP
            return rc.r_tp
        d = min(max(attack_delay, 0), rc.delay_cap)              # FN: 딜레이 선형+온셋가중
        # ★2026-09-09 FN_MODE=linear (A/B): -C_FN·min(ℓ_k, ℓ_max), ℓ_k=경과스텝(발생 포함, ≥1).
        #   C_FN=0.35 → 벌점열 0.35/0.70/1.05/1.40/1.75 (ℓ=1..5). 현행(onset_mult 곡선) 대비 단순·완만.
        if os.environ.get('FN_MODE', '') == 'linear':
            _c = float(os.environ.get('FN_C', '0.35'))
            return -_c * min(d + 1, rc.delay_cap) + (rc.relapse_penalty if relapse else 0.0)
        pen = rc.fn_base + rc.fn_per_step * d
        if 1 <= d <= rc.fn_onset_window:
            pen *= rc.fn_onset_mult
        if relapse:
            pen += rc.relapse_penalty                              # ★2026-09-10 재발 추가벌점
        return pen
    else:
        if current_action == 0:
            return rc.r_tn                                        # TN:+0.5
        # env FP_ESCAL=1 → FP 에스컬레이션(첫 오탐 -0.2 싸게, 지속 -1.2까지). 미설정시 기존 상수 -0.7 (2026-08-29 변형실험 V1)
        if _FP_ESCAL:
            return -0.2 - 0.25 * max(0, min(fp_run, 5) - 1)
        return rc.r_fp                                            # FP:-0.7(상수)
