"""
swrl_config.py — 드론 자율 복원 제어 통합 설정 (RHUKF-FV)
==========================================================
RL, 드론 물리, 보상, 공격/외란 풀, 커리큘럼, 시나리오 샘플러,
그리고 RHUKF-FV(error/absolute state) 필터 하이퍼파라미터의 중앙 통제소.

보상 설계는 env/reward.py 의 RewardConfig 로 분리됨 (cfg.reward).
"""
import os
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from env.reward import RewardConfig


@dataclass
class Config:
    # ══════════════════════════════════════════════════════════
    #  시스템
    # ══════════════════════════════════════════════════════════
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    outdir: str = "./results"
    headless: bool = False
    sim_launcher: str = 'isim'
    use_tf32: bool = True            # FP32 연산을 TF32로 가속
    use_compile: bool = False        # torch.compile (실시간 충돌 시 False)

    # ══════════════════════════════════════════════════════════
    #  에피소드 구조
    # ══════════════════════════════════════════════════════════
    warmup_seconds: float = 3.0
    attack_start_range: Tuple[int, int] = (30, 100)
    attack_ramp_duration: float = 0.05
    attack_duration_range: Tuple[int, int] = (50, 120)

    eps_action_probs: List[float] = field(default_factory=lambda: [0.9, 0.1])

    log_interval: int = 10

    # ══════════════════════════════════════════════════════════
    #  드론 물리 (보상/판정용)
    # ══════════════════════════════════════════════════════════
    natural_lag: float = 1.0
    max_error: float = 4.0
    min_altitude: float = -0.5
    flight_altitude: float = 5.0
    flight_radius: float = 5.0
    flight_omega: float = 0.5

    # ══════════════════════════════════════════════════════════
    #  비행 패턴 풀
    # ══════════════════════════════════════════════════════════
    flight_patterns: List[str] = field(default_factory=lambda: [
        'waypoint', 'circle', 'figure8', 'aggressive'
    ])

    # ══════════════════════════════════════════════════════════
    #  액추에이터 공격 풀
    # ══════════════════════════════════════════════════════════
    attack_enabled: bool = True
    attack_types: List[str] = field(default_factory=lambda: [
        'loe_thrust',
        'loe_combined',
    ])
    prob_no_attack: float = 0.15

    # ══════════════════════════════════════════════════════════
    #  커리큘럼
    # ══════════════════════════════════════════════════════════
    curriculum_enabled: bool = True
    curriculum_fixed_min: float = 0.04
    curriculum_start_max: float = 0.05
    curriculum_end_max: float = 0.9
    curriculum_warmup_episodes: int = 50
    curriculum_full_episodes: int = 150

    # ══════════════════════════════════════════════════════════
    #  환경 외란 풀
    # ══════════════════════════════════════════════════════════
    disturbance_enabled: bool = True
    disturbance_types: List[str] = field(default_factory=lambda: [
        'none', 'wind_turbulence', 'wind_constant', 'wind_gust',
    ])
    wind_speed_range: Tuple[float, float] = (3.0, 7.0)

    # ══════════════════════════════════════════════════════════
    #  RL 하이퍼파라미터
    # ══════════════════════════════════════════════════════════
    learning_warmup_steps: int = 10

    max_episodes: int = 200
    episode_max_steps: int = 300

    window_size: int = 4
    dimS: int = 12                   # window_size × 3
    num_actions: int = 2             # 0=궤도추종, 1=강제호버링

    gamma: float = 0.85
    scale_factor: float = 1.0
    batch_size: int = 128
    buffer_size: int = 20000

    # ── 탐험 ──
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000

    # ══════════════════════════════════════════════════════════
    #  D3QN 네트워크 구조
    # ══════════════════════════════════════════════════════════
    use_dueling: bool = True
    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])
    q_layers: List[int] = field(default_factory=lambda: [])   # non-dueling 전용
    activation_fn: str = 'silu'
    init_scheme: str = 'he'          # 'he' | 'xavier' | 'orthogonal'
    use_residual: bool = False

    # ══════════════════════════════════════════════════════════
    #  RHUKF-FV (필터 뇌) — error/absolute state, full-vector covariance
    # ══════════════════════════════════════════════════════════
    filter_form: str = 'covariance'        # RHUKF
    state_form: str = 'error'              # 'error'(기본) | 'absolute'
    decoupling_mode: str = 'fv'            # 현재 FV만 지원
    measurement_mode: str = 'q_target'     # z = r + γ^n·Q_target
    anchor_type: str = 'target'            # error-state θ_anchor
    ddqn_argmax: str = 'online_moving'
    h0_online_moving_init: str = 'theta_target'
    h0_prior_source: str = 'target'
    use_spas: bool = False                 # absolute h=0 sigma-ensemble argmax (off)

    N_horizon: int = 5
    update_interval: int = 1               # learn 빈도 게이트: N 스텝(=N번의 learn 호출)마다 1번만 실제 업데이트 (1=매 스텝)
    tau_srrhuif: float = 0.005             # soft target update 비율
    target_update_mode: str = 'soft'       # 'soft'(선택) | 'hard'
    target_update_period: int = 200

    # ── UKF 시그마포인트 ──
    alpha: float = 0.9
    beta: float = 2.0
    kappa: float = 0.0

    # ── 노이즈/공분산 (eps와 동일 지수 스케줄: init→end) ──
    q_init: float = 1e-2
    q_end: float = 1e-2
    r_init: float = 1.5
    r_end: float = 1.5
    p_init: float = 0.03                   # 초기 파라미터 공분산
    p_delta_init: float = 0.05             # error-state Δ 초기 공분산
    huber_c: float = 1000.0
    tikhonov_lambda: float = 1e-8

    # ── n-step ──
    use_n_step: bool = True
    n_step_size: int = 3

    # ── PER (IS weight는 R^-1 스케일로 적용) ──
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6
    per_apply_is_weight: bool = True       # r_inv_i = r_inv_base · w_i

    # ── 입력 정규화 (항상 ON; 드론 NIS는 [0,1]이라 scale=1.0 → identity) ──
    use_input_norm: bool = True
    obs_scale: List[float] = field(default_factory=lambda: [1.0] * 12)

    # ── 비활성 ──
    use_twin: bool = False

    done_steps: int = 4                    # 논리적 종료 한계 (탐지/복귀/오탐 연속)

    # ══════════════════════════════════════════════════════════
    #  보상 설계 (env/reward.py로 분리)
    # ══════════════════════════════════════════════════════════
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ══════════════════════════════════════════════════════════
    #  평가 (고정 시나리오)
    # ══════════════════════════════════════════════════════════
    eval_interval: int = 20
    eval_scenarios: List[dict] = field(default_factory=lambda: [
        {'pattern': 'aggressive', 'attack_type': 'none',
         'attack_intensity': 0.0, 'attack_start_step': 0,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        {'pattern': 'circle', 'attack_type': 'loe_combined',
         'attack_intensity': 0.055, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        {'pattern': 'figure8', 'attack_type': 'loe_combined',
         'attack_intensity': 0.8, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
    ])

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_init
        self.r_inv = 1.0 / (self.r_init ** 2)
        self.dimS = self.window_size * 3
        if self.obs_scale is None or len(self.obs_scale) != self.dimS:
            self.obs_scale = [1.0] * self.dimS
        os.makedirs(self.outdir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  커리큘럼 스케줄러
# ══════════════════════════════════════════════════════════════
def get_curriculum_intensity(episode: int, cfg: Config) -> Tuple[float, float]:
    if not cfg.curriculum_enabled:
        return (cfg.curriculum_fixed_min, cfg.curriculum_end_max)
    if episode <= cfg.curriculum_warmup_episodes:
        progress = 0.0
    elif episode >= cfg.curriculum_full_episodes:
        progress = 1.0
    else:
        progress = (episode - cfg.curriculum_warmup_episodes) / \
                   (cfg.curriculum_full_episodes - cfg.curriculum_warmup_episodes)
    lo = cfg.curriculum_fixed_min
    hi = cfg.curriculum_start_max + progress * (cfg.curriculum_end_max - cfg.curriculum_start_max)
    return (lo, hi)


# ══════════════════════════════════════════════════════════════
#  시나리오 샘플러
# ══════════════════════════════════════════════════════════════
def sample_episode_scenario(episode: int, cfg: Config) -> dict:
    scenario = {
        'pattern': random.choice(cfg.flight_patterns),
        'attack_type': 'none',
        'attack_intensity': 0.0,
        'attack_start_step': 0,
        'attack_end_step': 99999,
        'disturbance_type': 'none',
        'wind_speed': 0.0,
    }
    if cfg.attack_enabled and random.random() > cfg.prob_no_attack:
        scenario['attack_type'] = random.choice(cfg.attack_types)
        lo, hi = get_curriculum_intensity(episode, cfg)
        scenario['attack_intensity'] = random.uniform(lo, hi)
        scenario['attack_start_step'] = random.randint(*cfg.attack_start_range)
        duration = random.randint(*cfg.attack_duration_range)
        scenario['attack_end_step'] = scenario['attack_start_step'] + duration
    if cfg.disturbance_enabled:
        scenario['disturbance_type'] = random.choice(cfg.disturbance_types)
        if scenario['disturbance_type'] != 'none':
            scenario['wind_speed'] = random.uniform(*cfg.wind_speed_range)
    return scenario


# ══════════════════════════════════════════════════════════════
#  공격 Ramp / 힘·토크 변환
# ══════════════════════════════════════════════════════════════
def compute_attack_ramp(t_since_attack: float, target_intensity: float,
                        ramp_duration: float = 0.1) -> float:
    if ramp_duration <= 0 or t_since_attack >= ramp_duration:
        return target_intensity
    return target_intensity * (t_since_attack / ramp_duration)


def compute_attack_forces(attack_type: str, intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    force = np.zeros(3)
    torque = np.zeros(3)
    mag = intensity * 100
    if attack_type == 'loe_thrust':
        force[2] = -mag
    elif attack_type == 'loe_roll':
        torque[0] = mag * 0.8
    elif attack_type == 'loe_pitch':
        torque[1] = -mag * 0.8
    elif attack_type == 'loe_yaw':
        torque[2] = mag * 0.8
    elif attack_type == 'loe_combined':
        torque[0] = mag * 0.4
        torque[1] = -mag * 0.4
        force[2] = -mag * 0.5
    return force, torque
