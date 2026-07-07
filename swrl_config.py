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
    # ── PX4 토픽 네임스페이스. 'auto'=런타임 자동감지(살아있는 publisher 있는 ns).
    #    인스턴스 번호가 실행마다 비결정적(px4_1/px4_20…)이라 auto 권장. 고정하려면 '/px4_1' 등.
    px4_namespace: str = 'auto'
    kill_stale_px4_on_start: bool = True   # 새 sim 띄우기 전 좀비 PX4(bin/px4) 정리(포트충돌 방지)
    sim_startup_timeout: float = 360.0     # 첫 GT 수신까지 허용 시간(초). 헤드리스 Isaac 콜드 로딩 ~수분
    # ── uXRCE-DDS agent (PX4 /fmu/* ↔ ROS2 브리지). 직접 실행 시 isim이 안 켜줄 수 있어 자동 보장 ──
    xrce_autostart: bool = True
    xrce_agent_cmd: str = 'MicroXRCEAgent udp4 -p 8888'
    use_tf32_forward: bool = False   # forward(matmul)만 TF32 허용(Ampere+); 행렬연산은 항상 FP32. 전역 기본 FP32.
    use_compile: bool = True         # startup에서 학습 hot path 컴파일(inductor→aot_eager→eager 캐스케이드)+사전워밍업 후 spin. Isaac 번들 토치는 inductor 실패 시 자동 폴백
    agent_type: str = "rhukf"        # 'rhukf'(제안) | 'adam'(Adam+Huber baseline)
    adam_lr: float = 3e-4            # Adam baseline 학습률

    # ── DynamicsUKF(탐지 필터) 고집/관측가능성 ──
    #   stubbornness = low Q(현행 유지) + R을 실측 노이즈에 정합(아래 ukf_filter) + ff=1.0.
    #   maneuver-gated Q: 명령 토크에 비례해 gyro 프로세스노이즈 인플레 → 정상 기동 FP 억제.
    #   기본 0.0(off). sweep의 baseline-aggressive 셀에서 정상기동 NIS가 높으면 켜기(예: 0.05~0.2).
    ukf_q_gate_gyro: float = 0.0

    # ══════════════════════════════════════════════════════════
    #  에피소드 구조
    # ══════════════════════════════════════════════════════════
    warmup_seconds: float = 3.0
    attack_start_range: Tuple[int, int] = (50, 150)
    attack_ramp_duration: float = 0.3     # 0.0 = step 공격(즉시 full). EADR 논거(효과적 액추에이터 공격은 abrupt/고진폭) + 상승엣지 선명화.
                                          # 주의: step은 crash 데드라인을 줄임 → sweep의 dhover(지연 호버) 생존곡선으로 대응가능성 검증 후 확정.
    attack_duration_range: Tuple[int, int] = (50, 100)

    eps_action_probs: List[float] = field(default_factory=lambda: [0.8, 0.2])  # 탐험 track:hover=80:20 (50/50은 평시 FP폭증·교란)

    log_interval: int = 5

    # ══════════════════════════════════════════════════════════
    #  드론 물리 (보상/판정용)
    # ══════════════════════════════════════════════════════════
    natural_lag: float = 1.0
    max_error: float = 10.0          # drift 종료용 '통제상실' 경계(추적오차 판정 아님). 회복가능 상황엔 안 터지게 크게
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
        'loe_combined',   # 추력+토크 LoE. 토크 결손이 자세붕괴=결과성+탐지가능. (loe_thrust는 PX4가 보상=무해라 제거; FP압력은 기동/바람이 담당)
    ])
    prob_no_attack: float = 0.15

    # ── 공격 시간 구조: burst(on-off-on 반복)로 비정상성 강조 — FIR이 이기는 regime ──
    attack_mode: str = 'single'                                  # 'burst' | 'single'
    attack_burst_count_range: Tuple[int, int] = (2, 4)          # 에피소드당 버스트 개수
    attack_burst_on_range: Tuple[int, int] = (15, 35)           # 각 버스트 ON 길이 (RL steps@10Hz)
    attack_burst_off_range: Tuple[int, int] = (20, 50)          # 버스트 사이 OFF 길이

    # ── 공격 = 가산(additive) 복합 바이어스 (유일 형태; 곱셈형 LoE는 무해·미검출로 폐기) ──
    #   명령 무관 고정 오프셋을 ramp·intensity로 스케일해 플랜트에 주입.
    #   torque_xy→roll/pitch(gyro NIS), torque_z→yaw(gyro NIS), thrust_n→추력(vel NIS).
    #   복합이라 vel·gyro 두 관측채널 다 반응 + 실패모드 둘(flip/고도상실).
    attack_form: str = 'additive'    # (호환용 필드; additive만 지원)
    #   ★ 동결(2026-07-07 갱신): 공격채널 = COMBINED ft_ratio=1.5 (torque:thrust = 1:1.5, T≈2N).
    #     이전 torque-only 동결은 철회 — 당시 기각은 vel 채널 미확인 상태 판단이었음.
    #     채널분리 압축(vel=log(x+0.5)·gyro=log1p) + thrust 소량 추가로 vel d′ 1.5→2.2 상승,
    #     gyro d′는 3.0→2.5 소폭 하락하나 건재(운용 95pct 신호는 torque와 동등). 순이득 +0.7−0.5>0 → combined 채택.
    #   combined tube = (torque_xy=s, torque_z=0.2·s, thrust=1.5·s). 결과성 밴드 s∈[1.34,1.40]Nm (track추락 ∧ hover생존),
    #     ramp 0.0. s≥1.42 제외(hover도 붕괴=회복불가).
    #   ⚠ 관측 벡터 [nis_vel,nis_gyro,action]는 불변 — 압축 함수만 채널분리(_rl_step_10hz/_sweep_step_10hz 통일).
    sample_bias_box: bool = True                                # True: combined tube 샘플 / False: (bias_*×intensity)
    bias_scale_range: Tuple[float, float] = (1.34, 1.40)       # tube 중심축 s(Nm) — 동결 combined 밴드[1.34,1.40]
    bias_ft_ratio: float = 1.5                                  # thrust = ft_ratio · s (=1.5: combined 채택, T≈2N@s=1.34)
    bias_yaw_ratio: float = 0.2                                 # torque_z = yaw_ratio · s (동결 sweep torque 모드와 동일)
    bias_jitter: float = 0.10                                   # 각 성분 ±10% 지터(tube 두께=공격 다양성)
    #   (sweep/호환용 단일값 — 박스 OFF일 때만 사용)
    bias_torque_xy: float = 0.5
    bias_torque_z:  float = 0.1
    bias_thrust_n:  float = 2.5

    # ── 공격 에피소드 기동: 추락 밴드는 aggressive에서 검증됨 → 공격시 그 패턴으로 결합 ──
    #    (평시 에피소드는 flight_patterns 전체 사용; 타 패턴 밴드는 추후 재검증)
    attack_flight_patterns: List[str] = field(default_factory=lambda: ['aggressive'])

    # ── 탐험 편향은 eps_action_probs(=[0.8,0.2])로 처리 ──
    # ── TD-오차 첨도 로깅(무거운 꼬리 = Huber/유계영향 이점 근거) ──
    log_td_kurtosis: bool = True
    td_hist_size: int = 5000

    # ══════════════════════════════════════════════════════════
    #  커리큘럼 (OFF — 비커리큘럼: 매 에피소드 난이도 급변 = FIR 적응 이점 regime)
    # ══════════════════════════════════════════════════════════
    curriculum_enabled: bool = False
    curriculum_fixed_min: float = 0.15
    curriculum_start_max: float = 0.20
    curriculum_end_max: float = 0.45
    curriculum_warmup_episodes: int = 50
    curriculum_full_episodes: int = 150
    curriculum_fixed_min: float = 0.15
    curriculum_start_max: float = 0.20
    curriculum_end_max: float = 0.45
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
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay_steps: int = 6000

    # ══════════════════════════════════════════════════════════
    #  D3QN 네트워크 구조
    # ══════════════════════════════════════════════════════════
    # ── 순수 DDQN (dueling 제거) : shared_layers → q_layers → nA 단일 Q헤드 ──
    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    q_layers: List[int] = field(default_factory=lambda: [])   # [] = shared_out → nA 단일 선형
    activation_fn: str = 'silu'
    init_scheme: str = 'he'          # 'he' | 'xavier' | 'orthogonal'
    use_residual: bool = False
    adam_force_fp32: bool = True     # custom_env Adam-DDQN baseline은 TF32 끄고 FP32 (공정/재현)

    # ══════════════════════════════════════════════════════════
    #  RHUKF-FV (필터 뇌) — error/absolute state, full-vector covariance
    # ══════════════════════════════════════════════════════════
    filter_form: str = 'covariance'        # RHUKF
    state_form: str = 'error'              # 'error'(기본) | 'absolute'
    decoupling_mode: str = 'fv'            # 현재 FV만 지원
    measurement_mode: str = 'q_target'     # z = r + γ^n·Q_target
    anchor_type: str = 'target'            # error-state θ_anchor
    ddqn_argmax: str = 'online_moving'
    h0_online_moving_init: str = 'prev_est'
    h0_prior_source: str = 'target'
    use_spas: bool = False                 # absolute h=0 sigma-ensemble argmax (off)

    N_horizon: int = 5
    update_interval: int = 1               # Phase0: 1→4 (원본 rhukf.py 정합; transient 누적 완화). N번 learn 호출마다 1번 실제 업데이트
    tau_srrhuif: float = 0.005             # soft target update 비율
    target_update_mode: str = 'soft'       # 'soft'(선택) | 'hard'
    target_update_period: int = 200

    # ── UKF 시그마포인트 ──
    alpha: float = 0.1                     # Phase0: 0.9→0.3 (n_x≈514에서 σ스프레드 3배 축소→발산 억제)
    beta: float = 2.0
    kappa: float = 0.0

    # ── 노이즈/공분산 (eps와 동일 지수 스케줄: init→end) ──
    q_init: float = 1e-2
    q_end: float = 1e-2

    r_init: float = 2.0
    r_end: float = 2.0

    p_init: float = 0.05                   # 초기 파라미터 공분산
    p_delta_init: float = 0.05             # error-state Δ 초기 공분산
    huber_c: float = 8.0                   # 5→3 (residual RMS~3에서 adapt_factor가 실제로 켜지도록). 2~4 사이 튜닝
    tikhonov_lambda: float = 1e-8

    # ── n-step ──
    use_n_step: bool = False
    n_step_size: int = 3

    # ── PER (이번 실험: PER off → Huber-R 단독 outlier 방어로 FIR 기여 isolate) ──
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6
    per_apply_is_weight: bool = True       # r_inv_i = r_inv_base · w_i (use_per=True일 때만 효과)

    # ── 입력 정규화 (항상 ON; 드론 NIS는 [0,1]이라 scale=1.0 → identity) ──
    use_input_norm: bool = True
    obs_scale: List[float] = field(default_factory=lambda: [1.0] * 12)

    # ── 비활성 ──
    use_twin: bool = False

    # ══════════════════════════════════════════════════════════
    #  종료(done) 정책
    # ══════════════════════════════════════════════════════════
    done_steps: int = 4                    # 논리적 종료 한계 (use_logical_done=True일 때만 사용)
    use_logical_done: bool = False         # 논리적 종료(미탐/복귀/오탐) 사용 여부. 기본 False = 물리적 crash만 종료
    drift_patience: int = 10               # crash_drift: >max_error를 연속 10스텝(10Hz≈1.0s) 지속 시에만 = 명백한 통제상실 안전망. truncation(페널티X)
    soft_recovery_timeout: float = 15.0    # SOFT_RECOVERY 복구 실패 시 WARM_RESET 에스컬레이션 (초)

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
         'attack_intensity': 0.25, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        {'pattern': 'figure8', 'attack_type': 'loe_combined',
         'attack_intensity': 0.40, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
    ])

    # ══════════════════════════════════════════════════════════
    #  α-SWEEP (결과성 밴드 + 탐지가능성 + CUSUM baseline 특성화)
    # ══════════════════════════════════════════════════════════
    #  BIAS SWEEP (sweep_mode=True면 학습 OFF, 고정정책 비행, raw NIS+생존 CSV 기록)
    #  online_rl_main.py --sweep 로 켬.
    #  ───────────────────────────────────────────────────────────
    #  sweep_attack_mode = 어느 채널로 공격이 들어오나:
    #    'combined' : 토크+추력 동시.   sweep값 = roll/pitch 토크 b(Nm), 추력=ft_ratio·b, yaw=yaw_ratio·b
    #    'torque'   : 토크만.           sweep값 = roll/pitch 토크 b(Nm), yaw=yaw_ratio·b, 추력=0
    #    'thrust'   : 추력만.           sweep값 = 추력 b(N), 토크=0
    #  sweep_values = 그 채널에서 휩쓸 물리 바이어스 크기(모드에 따라 Nm 또는 N).
    #  → 각 모드별 "track 추락 ∧ hover 생존" 밴드 = 감당 가능 한계를 찾는다.
    #
    #  [권장 grid] — 각 모드의 '붕괴 경계'를 브래킷 (baseline b=0은 자동 추가됨)
    #    combined : [0.8, 1.0, 1.2, 1.3, 1.5, 1.7]      (Nm; 추력=5·b → 4~8.5N; 토크를 flip영역까지)
    #    torque   : [1.0, 1.2, 1.3, 1.4, 1.5, 1.7]      (Nm; 밴드 [1.3,1.5) 정밀화)
    #    thrust   : [8.0, 12.0, 14.0, 16.0, 20.0, 25.0] (N;  ~14N=권한포화→고도붕괴 브래킷)
    # ══════════════════════════════════════════════════════════
    sweep_mode: bool = False
    sweep_attack_mode: str = 'combined'        # 'combined' | 'torque' | 'thrust'
    sweep_values: List[float] = field(default_factory=lambda: [
        0.8, 1.0, 1.2, 1.3, 1.5, 1.7])         # 기본=combined의 토크 b(Nm)
    sweep_combined_ft_ratio: float = 2.0       # 추력/토크_xy 비. 파일럿1: 5.0에선 th=6.5N@b=1.3이 hover까지
                                               # crash_altitude로 죽여 결과성 밴드 소멸 → 토크 우세(2.0)로 하향.
    sweep_torque_yaw_ratio:  float = 0.2       # yaw/roll·pitch 비 (검출 보조)
    sweep_pattern: str = 'aggressive'          # track 셀 비행패턴(명령토크 최대=최악조건)
    sweep_episodes: int = 8                    # 셀당 반복(RNG 노이즈)
    sweep_attack_start: int = 30               # 공격 ON 스텝(@10Hz). 이후 ramp
    # ── 조건 C: 지연 호버(delayed hover) — "공격 시작 후 d스텝 뒤 호버 전환" 정책 ──
    #   추적 기동 관성+교란 자세를 안고 현재위치 호버로 전환하는 '전이 케이스'를 검증.
    #   생존율 vs d 곡선 = 탐지 데드라인. 실측 탐지지연(~3스텝)에서 생존해야 RL 프레이밍 성립.
    #   빈 튜플 () 로 두면 기존 A/B(track/hover) 셀만 실행.
    sweep_hover_delays: Tuple[int, ...] = (1, 2, 3, 4, 5, 8)

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
        'attack_bursts': [],
        'disturbance_type': 'none',
        'wind_speed': 0.0,
    }
    if cfg.attack_enabled and random.random() > cfg.prob_no_attack:
        scenario['attack_type'] = random.choice(cfg.attack_types)
        # 공격 에피소드는 추락밴드가 검증된 기동으로 (sweep=aggressive)
        scenario['pattern'] = random.choice(getattr(cfg, 'attack_flight_patterns', ['aggressive']))
        if getattr(cfg, 'sample_bias_box', True):
            # combined 추락 ray (s, 0.2·s, 5·s) 주변 tube 샘플 — scale 하나로 묶어야 ray를 안 벗어남.
            scenario['attack_intensity'] = 1.0
            s = random.uniform(*cfg.bias_scale_range)
            j = cfg.bias_jitter
            jit = lambda: random.uniform(1.0 - j, 1.0 + j)
            scenario['bias_torque_xy'] = s * jit()
            scenario['bias_torque_z']  = cfg.bias_yaw_ratio * s * jit()
            scenario['bias_thrust_n']  = cfg.bias_ft_ratio  * s * jit()
            scenario['bias_scale'] = s          # 로깅/분석용 (밴드 대비 위치)
        else:
            lo, hi = get_curriculum_intensity(episode, cfg)
            scenario['attack_intensity'] = random.uniform(lo, hi)
            scenario['bias_torque_xy'] = cfg.bias_torque_xy
            scenario['bias_torque_z']  = cfg.bias_torque_z
            scenario['bias_thrust_n']  = cfg.bias_thrust_n

        if getattr(cfg, 'attack_mode', 'single') == 'burst':
            # 버스트 일정: ON 구간을 여러 번 (on-off-on …) → 반복적 빠른 적응 요구
            bursts = []
            t = random.randint(*cfg.attack_start_range)
            for _ in range(random.randint(*cfg.attack_burst_count_range)):
                on = random.randint(*cfg.attack_burst_on_range)
                bursts.append((t, t + on))
                t = t + on + random.randint(*cfg.attack_burst_off_range)
                if t > cfg.episode_max_steps - 10:
                    break
            scenario['attack_bursts'] = bursts
            scenario['attack_start_step'] = bursts[0][0]      # 로깅 호환
            scenario['attack_end_step'] = bursts[-1][1]
        else:
            scenario['attack_start_step'] = random.randint(*cfg.attack_start_range)
            duration = random.randint(*cfg.attack_duration_range)
            scenario['attack_end_step'] = scenario['attack_start_step'] + duration
            scenario['attack_bursts'] = [(scenario['attack_start_step'], scenario['attack_end_step'])]

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


def compute_attack_forces(attack_type: str, intensity: float,
                          bias_torque_xy: float = 0.12,
                          bias_torque_z: float = 0.0,
                          bias_thrust_n: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """가산(additive) 바이어스: 명령 무관 고정 토크/추력 오프셋을 intensity(0~1, ramp 출력)로 스케일.
       크기는 config(bias_*)로 결정 → b-sweep으로 '붕괴 직전' 값 탐색 가능.
       (곱셈형 LoE는 명령(u_ref) 의존이라 run_sim에서 인라인 처리.)"""
    force = np.zeros(3)
    torque = np.zeros(3)
    if attack_type == 'loe_thrust':
        force[2] = -intensity * bias_thrust_n
    elif attack_type == 'loe_roll':
        torque[0] = intensity * bias_torque_xy
    elif attack_type == 'loe_pitch':
        torque[1] = -intensity * bias_torque_xy
    elif attack_type == 'loe_yaw':
        torque[2] = intensity * bias_torque_z
    elif attack_type == 'loe_combined':
        # 원본 결합 형태(roll+, pitch-, 추력 하향)를 config 크기로 스케일.
        torque[0] =  intensity * bias_torque_xy
        torque[1] = -intensity * bias_torque_xy
        torque[2] =  intensity * bias_torque_z
        force[2]  = -intensity * bias_thrust_n
    return force, torque


def sweep_bias_vector(mode: str, value: float,
                      ft_ratio: float = 5.0,
                      yaw_ratio: float = 0.2) -> Tuple[float, float, float]:
    """sweep 모드+물리값 → (bias_torque_xy, bias_torque_z, bias_thrust_n).
       torque  : value=roll/pitch 토크 b(Nm), yaw=yaw_ratio·b, thrust=0
       thrust  : value=추력 b(N), torque=0
       combined: value=roll/pitch 토크 b(Nm), yaw=yaw_ratio·b, thrust=ft_ratio·b
    """
    if mode == 'torque':
        return (float(value), yaw_ratio * float(value), 0.0)
    elif mode == 'thrust':
        return (0.0, 0.0, float(value))
    else:  # 'combined'
        return (float(value), yaw_ratio * float(value), ft_ratio * float(value))
