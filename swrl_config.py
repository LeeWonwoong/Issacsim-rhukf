"""
swrl_config.py — 드론 자율 복원 제어 통합 설정 (RHUKF-FV)
==========================================================
RL, 드론 물리, 보상, 공격/외란 풀, 커리큘럼, 시나리오 샘플러,
그리고 RHUKF-FV(error/absolute state) 필터 하이퍼파라미터의 중앙 통제소.

보상 설계는 env/reward.py 의 RewardConfig 로 분리됨 (cfg.reward).
"""
import math
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
    attack_start_range: Tuple[int, int] = (20, 40)     # FDI 캠페인 시작(2026-08-19)
    attack_ramp_duration: float = 0.0     # ★0.0=step 공격(즉시 full) 확정(2026-07-23). 펄스 시그니처(t=1~2 스파이크) 선명화=즉각감지 서사.
                                          #   EADR 논거(효과적 액추에이터 공격=abrupt/고진폭). ⚠기존 0.3은 온셋 뭉갬→펄스 약화(학습이 0.3으로 돌던 버그).
                                          # 주의: step은 crash 데드라인을 줄임 → sweep의 dhover(지연 호버) 생존곡선으로 대응가능성 검증 후 확정.
    attack_duration_range: Tuple[int, int] = (50, 100)

    eps_action_probs: List[float] = field(default_factory=lambda: [0.85, 0.15])   # ★09-18: env EPS_HOVER_P 폐기 → YAML agent.eps.hover_p  # 2026-08-19 탐험 track:hover=85:15 (★09-13 env EPS_HOVER_P 로 hover 탐험 비율 조정, 약속 D 실험용)

    log_interval: int = 5

    # ══════════════════════════════════════════════════════════
    #  드론 물리 (보상/판정용)
    # ══════════════════════════════════════════════════════════
    natural_lag: float = 1.0
    max_error: float = 10.0          # drift 종료용 '통제상실' 경계(추적오차 판정 아님). 회복가능 상황엔 안 터지게 크게
    min_altitude: float = -0.5
    flight_altitude: float = 2.5   # ★2026-09-07 5.0→2.5 (사용자확정 10×10×4m). tilt공격 추락=crash_flip 이라 고도는 데드라인 아닌 flip여유
    #  ★ 2026-08-07: 실기 속도제한(MPC_XY_VEL_MAX=2.0)에 맞춰 재설정.
    #    구값 R=5.0·w=0.5 → 접선 2.50 m/s 로 실기에서 클램프됐다.
    #    현값 R=3.5·w=0.5 → 1.75 m/s (여유 12.5%). 공간 circle 7x7 m.
    # ★2026-09-07 실공간 복원(사용자 확정: 10×10×4m 정상 비행에서 측정). [OLD] 1.6/0.75 는
    #   v2 회전여기 논리(폐기)의 산물 — 화면으로 확인 시 비정상적으로 좁은 선회였다.
    #   08-12 실기 안전값 2.8 복원 + ω 0.45: 접선 2.8×0.45=1.26, 가감속 amp0.5 피크 1.89 < 상한 2.0 (여유 5%)
    flight_radius: float = 3.4   # ★2026-09-07 2.8→3.4 (반경 확대, circle Ø6.8m). ω 동반하향으로 속도 유지
                                 #   (구 3.5 는 1.75 로 여유 14% 뿐이라 명령속도가 2.00 에 상시 포화 →
                                 #    기체가 뒤처져도 못 따라잡아 추종 RMSE 1.4~2.9 m. 실기 08-12 실측)
                                 #   ★ 실기 f5_pattern.py --R 기본값과 **반드시 같이** 바꾼다.
    flight_omega: float = 0.38   # ★2026-09-07 0.45→0.38 (반경 3.4 확대 동반). 접선 3.4×0.38=1.29·가감속피크 1.94<2.0
    #  figure8 만 반경을 1/sqrt(2) 배로 쓴다.
    #    x=R sin(wt), y=(R/2) sin(2wt) → vx=Rw cos(wt), vy=Rw cos(2wt) 라
    #    t=0 에서 둘이 동시에 최대 → |v|max = R*w*sqrt(2) = 2.47 m/s 로 클램프된다.
    #    R 을 1/sqrt(2) 배 하면 |v|max 가 circle 과 같은 R*w 가 된다.
    fig8_radius_scale: float = 0.70710678
    wp_box_halfwidth: float = 3.2   # ★2026-08-27 waypoint 박스 반폭[m] — R 에서 분리.
    #   구 구현은 kk=R/5 로 R 축소(2.8→1.6)가 박스를 5.6→3.2m 로 같이 줄였는데, 코너 선회반경은
    #   물리 고정(v²/ACC_HOR = 0.48~1.08m)이라 레그 1.6m 에선 코너 라운딩이 레그를 잠식(피크속도
    #   에선 2.16m>1.6m = 모양 붕괴, 코너 체류 38% = 상시 과도 진동). 3.2 → 레그 3.2m, 박스 6.4m
    #   (circle 7×7 공간 내). 실기 f5_pattern 도 동일 값 필요.
    # ★2026-08-27 코너 감속 프로파일 (사용자 확정: '코너 감속 + 직선 가속' = 기하 연동 가감속).
    #   레그별 사다리꼴 속도: 코너 v_corner → a_prof 로 가속 → v_cruise → 코너 전 감속.
    #   코너 라운딩 r = v_corner²/3.0 = 0.08m (각진 코너) · 과도(피치±선회)가 코너마다 구조적으로 발생
    #   = 실기 오토미션과 같은 주행 문법. waypoint 는 사인 SPEED_MOD 대신 이것을 쓴다.
    wp_corner_decel: bool = True
    wp_v_corner: float = 0.5       # 코너 통과 속도 [m/s]
    wp_v_cruise: float = 1.7       # 직선 순항 속도 [m/s] (< 상한 2.0)
    wp_a_prof: float = 2.0         # 프로파일 가감속 [m/s²] (< ACC_HOR 3.0, 횡가속 여유 확보)

    # ── aggressive 패턴 파라미터 (2026-08-07: 하드코딩 → config) ──
    #  ★ **실기에서 안전하게 날 수 있는 값**으로 잡는다. sim 이 그 값을 따른다.
    #    실기에서 못 나는 궤적은 sim 에서 아무리 잘 돌아도 검증할 수 없기 때문.
    #    실기 스크립트: px4_field/f5_pattern.py --pattern aggressive (--agg-R)
    #  구값(하드코딩): R=4.0, dz1=3.0, dz2=2.0 → 반원 2.51 m/s, 수직 1.88 m/s
    #  현값          : R=2.0, dz1=1.0, dz2=0.7 → 반원 1.26 m/s, 수직 0.63 m/s
    #                  (circle R=3·w=0.5 의 접선속도 1.50 m/s 와 비슷한 영역)
    #  ⚠ 완화하면 aliasing 난이도가 내려간다(급기동이 RL 존재 이유이므로).
    #    [3] 재측정에서 분리도가 너무 쉽게 나오면 여기를 올려 재조정할 것.
    agg_radius: float = 2.24       # ★2026-09-07 실공간 복원 (08-12 배율 0.8×2.8). [OLD] 1.28
    # ★2026-09-07 scurve 활성화(5번째 패턴) + 실공간 확대. Rs 1.1→2.0, 접선 v=R·ω=1.29.
    #   a_lat=v²/Rs=0.83, 반전점 Δa=1.66<ACC 4.0. S 폭 = 2·Rs·arcs 방향 전개, 고도 ±0.8.
    scurve_radius: float = 2.0
    scurve_arcs: int = 3
    scurve_dz: float = 0.8
    agg_dz1: float = 1.0           # phase0 상하 진폭 [m]
    agg_dz2: float = 0.7           # phase2 상하 진폭 [m]
    agg_phase_s: float = 5.6       # ★2026-09-10 3.35→4.7→5.6: 속도변조(×1.5 피크) 포함 반원 접선 π·2.24/5.6=1.26, 피크 1.88 < 2.0 (여유 6%, circle 피크 1.94 와 동급).
    #   [4.7 은 변조 미고려: 피크 2.25 > 캡]  원 주석: 반경 2.24 복원(09-07) 때 Tp 를 안 고쳐 반원속도 π·2.24/3.35=2.10 m/s
    #   > MPC_XY_VEL_MAX 2.0 이라 안쪽으로 파고들었다(trajcheck RMSE 1.18). 4.7 → 접선 1.50 m/s, 구심 1.0 m/s².
    #   실기 f5_pattern.py --agg-phase 와 **반드시 같이** 바꾼다.
    # [OLD] 3.35     # phase 하나 길이 [s]  ★2026-08-27 5.0→3.35: 반원 속도 = Ra·π/Tp 라
    #   반경만 2.24→1.28 로 줄이면(08-26 ③) 속도가 1.41→0.80 로 같이 느려진다(Tp 미조정 버그).
    #   3.35 = 1.28π/1.20 → 반원 접선속도 1.20 m/s(circle 과 동일), 선회각속도 0.94 rad/s(circle 0.75 보다 급).

    # ══════════════════════════════════════════════════════════
    #  비행 패턴 풀
    # ══════════════════════════════════════════════════════════
    flight_patterns: List[str] = field(default_factory=lambda: [
        'waypoint', 'circle', 'figure8', 'aggressive', 'scurve'
    ])

    # ══════════════════════════════════════════════════════════
    #  액추에이터 공격 풀
    # ══════════════════════════════════════════════════════════
    attack_enabled: bool = True
    attack_types: List[str] = field(default_factory=lambda: [
        'tilt',   # 2026-08-19 확정: torque-only 틸트(roll+pitch 랜덤방향, thrust=0). (B)궤적이탈·불변식.
    ])
    prob_constant_attack: float = 0.30   # 공격 에피 중 상수(지속) 비율. 나머지는 FDI 랜덤버스트.
    attack_delta_range: Tuple[float, float] = (0.1, 0.7)   # ★2026-08-27 상한 0.8→0.7 (사용자 확정).
    #   근거: 호버 위치격리 문턱 실측 — δ0.6: 1.7m / δ0.7: 4.0m 격리·즉시복귀 / δ0.8: 32m(격리 불가,
    #   공격이 위치루프 권한 초과). 상한 0.7 = 밴드 전체에서 '감지→강제홀딩→복귀' 서사가 성립.
    #   δ0.7 도 관측상 easy anchor 유지(고원 2.4+, BErr<0.01). 하한 0.1 은 현 비교 유지, 본실험서 0.4 검토.
    #   (구 0823 근거: δ0.1=탐지하한, δ0.05 제외 — 유지)
    attack_tq_authority_nm: float = 4.36                   # δ→N·m 환산(주입 정규화용)
    prob_no_attack: float = 0.30     # 2026-08-19: 70%공격/30%무공격(FP 캘리브)

    # ── 공격 시간 구조: burst(on-off-on 반복)로 비정상성 강조 — FIR이 이기는 regime ──
    attack_mode: str = 'burst'     # 2026-08-19 FDI: 'burst'(랜덤 on/off) | 'single'
    attack_burst_count_range: Tuple[int, int] = (2, 4)          # 에피소드당 버스트 개수
    # ★2026-09-07 FROZEN-v3.1 기본값 정착: 버스트 = intermittent fault (온셋킥, 탐지 POMDP 담당).
    #   [OLD] (40,80) 2026-08-28 지속 hijacking — v3 환경 복구로 폐기 (적분 여유가 커서 늦은 탐지가 무벌점이었음).
    #   결과성(절벽)은 버스트가 아니라 ramp 클래스(incipient, 아래 v3.1 블록)가 담당한다.
    attack_burst_on_range: Tuple[int, int] = (10, 20)   # ★v3.1: 1~2s 온셋킥. 창(4스텝)의 2.5~5배 = 지속성 특징 표현
    attack_burst_off_range: Tuple[int, int] = (25, 40)  # ★v3.1: 필터감쇠 2~3스텝 + 복귀결정 여유. ramp 클래스와 동일(OFF로 클래스 누설 방지)

    # ── ★2026-09-07 v3.1 ramp 공격 클래스 (incipient fault — 결과성 절벽 담당) ──
    #   목적: "관측 자명 시점"과 "결정 시점"의 분리 — 자명해지면 늦고, 애매 구간에서 헤징해야 생존.
    #   ⚠ 샘플러 구현은 [B2] whover 지연곡선("늦으면 죽는다") 통과 후 [C] 동결 때 붙인다.
    #   설계 근거·검토 필수사항 = SWRL 프레임워크 아티팩트 §frozen-v31 5절 (코드 설계 전 반드시 읽을 것).
    # ★2026-09-08 최종 (ramp 클래스 폐기 — 점진은 트림 흡수로 비치명 실측): 결과성 = 급작 버스트.
    #   flipband/bandfill 실측: δ0.78~0.82 무풍 track 25/25 사망 ∧ 호버 25/25 생존. LOC lag p10 0.6s 중앙 2.8s.
    #   δ0.85 = 호버도 전멸(천장, 절대 초과 금지). 치명선은 0.70~0.78 에서 패턴·바람 따라 확률적(은닉).
    prob_lethal_attack: float = float(__import__('os').environ.get('PROB_LETHAL','0.30'))   # 공격 에피 중 결과성 비율. env PROB_LETHAL 조절
    lethal_delta_range: Tuple[float, float] = (0.74, 0.80)   # ★2026-09-09 bandfill 확정: 0.72 무해·0.74~0.76 확률적·0.78+ 확정치명. 0.82는 호버도 死(33~58%) → 상한 0.80
    lethal_hold_range: Tuple[int, int] = (40, 60)       # 4~6s (LOC lag 6~32스텝 커버)

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

    # ── 공격 주입 정규화 상수 (2026-08-08, allocator 경로) ──────────────────
    #  외부 wrench 폐기 → PX4 allocator c[0] 가산으로 통일(sim=실기). c[0] 는 정규화값이라
    #  online_rl 이 물리 bias(N·m,N)를 정규화(=권한 대비 비율)로 바꿔 DDS 로 쏜다:
    #      normalized = physical / authority
    #  ⚠⚠ PROVISIONAL: 아래는 구 플랜트 근사(CLAUDE.md). **밴드 재측정 시 정규화 단위로
    #     재정의**하면 이 환산 자체가 불필요해진다(밴드를 처음부터 정규화로 스윕).
    attack_tq_authority:  float = 4.36    # 롤/피치 토크 권한 [N·m] (동결 s=1.34 → 31%)
    attack_yaw_authority: float = 4.36    # 요 토크 권한 [N·m] (미측정 — 롤과 동일 가정)
    attack_th_authority:  float = 40.0    # 최대 추력 [N] ≈ T/W 3 × mg 13.15

    # ── 공격 에피소드 기동: 실제 공격은 모든 기동 궤적에 들어간다 → 평시와 동일한 패턴 분포 ──
    #    (2026-07-23 수정: 기존 ['aggressive'] 강제는 버그였음. 추락 밴드는 aggressive에서 검증됐지만
    #     학습 공격은 전 패턴에 랜덤 세기로 주입하는 것이 의도된 설계. deadline 스윕(hover/waypoint/figure8
    #     @1.37·1.40)이 밴드 유지 확인 역할. flight_patterns 전체 사용.)
    attack_flight_patterns: List[str] = field(default_factory=lambda: [
        'waypoint', 'circle', 'figure8', 'aggressive', 'scurve'
    ])

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
    # ── 바람 외란 고정(2026-07-08, STEP1/2 근거) ──
    #   aggressive=주 aliasing, 바람=보조(ws≈7 turbulence만). 결정: [[wind-turbulence-bug-and-step2]].
    #   wind_constant 제외: PX4가 완전 상쇄(정지비행 틸트만, 잔차0) → aliasing 무용.
    #   wind_gust 제외: 미검증(스윕 근거 없음). 필요시 재검증 후 추가.
    #   turbulence는 run_sim.py OU진폭 수정(sqrt(1-a^2)) 후에만 유효(수정 전=사실상 상수풍).
    disturbance_types: List[str] = field(default_factory=lambda: [
        'none', 'wind_turbulence',
    ])
    disturbance_weights: List[float] = field(default_factory=lambda: [0.4, 0.6])  # none 40% / turbulence 60% (2026-07-22)
    #   바람 확정(2026-07-22, 캡처A 근거): constant/gust 제거(정상 NIS바닥 불변) — turbulence만 바닥 상승(median 0.013→0.030@7).
    #   wind_speed_range (0.3,7.0)→(1.0,5.0): 강도 3~7서 온셋여유 0.9 포화 → 5 초과 무의미, 하한 1.0(무풍은 none 40%가 담당).
    wind_speed_range: Tuple[float, float] = (1.0, 5.0)
    # ── 2026-08-19: 바람 nominal(약풍) 75% / 강풍(aliasing) 25% ──
    prob_strong_wind: float = 0.35   # ★확정0823: 강풍 35% (POMDP 노출 확보)
    wind_nominal_range: Tuple[float, float] = (0.5, 2.0)    # 약풍(현실 ambient)
    wind_strong_range:  Tuple[float, float] = (4.0, 6.0)    # ★2026-09-08 (4,8)→(4,6): ws8+ 는 δ0.8 호버도 사망(실측) = 조건③ 붕괴. ws6 은 호버 12/12 생존. ⚠ bandfill ws7 결과로 상한 7 재검토
    # ★FROZEN-ENV v2: 강풍은 에피 전체가 아니라 윈도우 — 안정화 후 온셋(U 20-100), 길이 U(150,300)스텝(공격 burst 40-80 보다 충분히 큼). 약풍은 에피 전체(무해).
    wind_window_start_range: Tuple[int, int] = (20, 100)
    wind_window_len_range:   Tuple[int, int] = (150, 300)

    # ══════════════════════════════════════════════════════════
    #  RL 하이퍼파라미터
    # ══════════════════════════════════════════════════════════
    learning_warmup_steps: int = 10

    max_episodes: int = 300
    episode_max_steps: int = 400  # ★2026-09-07 v3 기본 400 ([OLD] 300). env override 유지
    sim_speed_factor: float = 10.0   # Isaac Sim 배속(2026-07-22): 모든 캡처·스윕 기본 10. --speed로 override. ※페어링(학습)은 RHUKF learn 지연 확인 필요.

    window_size: int = 4
    dimS: int = 12                   # window_size × 3
    num_actions: int = 2             # 0=궤도추종, 1=강제호버링

    gamma: float = 0.85   # 하이퍼탐색 A: 0.85. 2026-09-09 env화(γ스캔용). Adam·RHUKF 공유.
    scale_factor: float = 1.0
    reward_scale: float = 1.0      # ★2026-08-19: 스케일링 폐기(논문 가독성). 대신 RewardConfig 값 자체를 절반으로 재설계해 max|r|≈3.9(≈huber_c3)로 낮춤. 스케일 상수는 1.0 고정.

    # ── 탐험 ──
    eps_start: float = 0.99
    eps_end: float = 0.01
    eps_z_mu: float = 0.0      # ★09-19 εz-greedy(Dabney·Ostrovski·Barreto ICLR2021): >1 이면 탐험 행동을 n~zeta(μ) 스텝 유지. 0 = 끔(기존과 비트 동일)
    eps_z_cap: int = 30        # εz 지속 상한(스텝)
    eps_decay_steps: int = 10000  # 2026-09-09 env화. 기본 10000(탐험~100ep). CartPole은 2000이었음 — 빠른 decay가 초반 정책차 가시화

    # ══════════════════════════════════════════════════════════
    #  D3QN 네트워크 구조
    # ══════════════════════════════════════════════════════════
    # ── 순수 DDQN (dueling 제거) : shared_layers → q_layers → nA 단일 Q헤드 ──
    shared_layers: List[int] = field(default_factory=lambda: [24, 24])  # 페어링용(2026-07-22): [16,16]→[24,24]. 962 params@12D / 1058@16D. RHUKF·Adam 공통.
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
    filter_mode: str = 'rhukf'             # ★09-13 'rhukf'(FIR 창) | 'ukf'(KTD형 무한기억 UKF-TD) | 'ekf'(EKF-TD, 야코비안)
    decoupling_mode: str = 'fv'            # 현재 FV만 지원
    measurement_mode: str = 'q_target'     # z = r + γ^n·Q_target
    innov_mean: str = 'ut'                 # ★09-20 error-state 잔차 기준: ut(시그마 가중평균, 기존) | center(중심 시그마점 Q(θ̄))
    anchor_type: str = 'target'            # error-state θ_anchor
    act_net: str = 'active'    # ★09-19 SWIRL 행동망: active = θ_T+μ(이번 학습 호출 보정, 기존) | target = 누적 θ_T (호출마다 흔들리는 보정 없이 행동)
    ddqn_argmax: str = 'online_moving'
    h0_online_moving_init: str = 'spas'    # RHUKF 고-K(2026-07-14): prev_est→spas (h0 argmax만 시그마앙상블, 고T_Var 강건)
    h0_prior_source: str = 'target'
    use_spas: bool = True                  # ★2026-09-07 기본 ON (사용자 확정 "아이작심에서 다 spas"). A/B 실측 무해(t=-1.59). [OLD] False

    batch_size: int = 128   # 09-15 19:25 env화(배치 64 절제용). 기본 128 불변
    buffer_size: int = 20000   # 2026-09-09 env화(온라인성 스캔용)
    
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
    q_init: float = 1e-4                   # RHUKF 고-K(2026-07-14): 1e-2→1e-4 (P재팽창 억제, 고T_Var 안정)
    q_end: float = 1e-4

    r_init: float = 1.0                    # ★2026-08-20 1.5→1.0 복귀: 고-T_Var RL은 고-K(r낮게)로 추적+강건성기제(spas/huber/느린타깃)가 문서(LL-6) 검증. r↑(K↓)는 고전칼만직관이나 RL실증과 반대. (2026-07-14: 2.0→1.0)
    # ★2026-08-20 센서 σ 정합(현실성+강건성 축): gyro 실기 실측 σ [rad/s]. GPS는 run_sim(0.9×).
    #   SENSOR_NOISE_SCALE env 로 clean(0)/real(1)/stress(2·3) 스윕. 실기 3소티: [.072,.069,.023]/[.079,.060,.021]/[.134,.116,.032]
    gyro_sensor_sigma: Tuple[float, float, float] = (0.066, 0.070, 0.044)  # 실측 field σ (f11_hover log33, SIM_ALIGN)
    r_end: float = 1.0

    p_init: float = 0.05                   # 초기 파라미터 공분산 (absolute 모드용)
    p_delta_init: float = 0.02             # RHUKF 고-K(2026-07-14): 0.05→0.02 (error-state Δ 초기 공분산)
    huber_c: float = 3.0                   # RHUKF 고-K(2026-07-14): 8→3 (관측 잔차 p90≈3 → 유계영향 실제 활성)
    tikhonov_lambda: float = 1e-8

    # ── n-step ── 로드맵 학습수정(2026-07-08): off→on, n=3 (memory.py 구현됨; 온셋 펄스 신호 부트스트랩 전파)
    use_n_step: bool = True   # 2026-09-09 env화: NSTEP_OFF=1 → 1-step
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

    # ── ★2026-09-18 env → 설정 필드 이전 (YAML 이 값을 넣는다) ──
    gyro_only: bool = False            # 관측에서 vel 제거 (obs.features 로 결정)
    hover_dwell: int = 0               # hover 확약 스텝 (0 = 매 스텝 결정)
    adam_amsgrad: bool = False
    adam_init: str = 'default'         # default(PyTorch) | he (필터 에이전트와 같은 He-normal·bias0)
    adam_optimizer: str = 'adam'       # adam | sgd (momentum 0)
    adam_loss: str = 'huber'           # huber | mse
    adam_huber_beta: float = 1.0
    adam_grad_clip: float = 1.0
    replay_mode: str = 'uniform'       # uniform | cer | recency
    replay_halflife: float = 3000.0
    log_zu: bool = False              # 구 --log-zu (env.isaac.cfg 로 켠다, 09-18 검토)
    log_sysid: bool = False           # 구 --log-sysid

    # ══════════════════════════════════════════════════════════
    #  평가 (고정 시나리오)
    # ══════════════════════════════════════════════════════════
    eval_interval: int = 20
    #  eval 시나리오 [교체 2026-07-08]: 동결밴드 combined ft1.5(bias=s, yaw=0.2·s, thrust=1.5·s)로 통일.
    #    옛 loe_combined intensity 0.25/0.40 = 기본bias(0.12,0,2.0)×0.25 = 비치명(조건① 미성립) → 폐기.
    #    이제 학습(sample_bias_box)과 동일 주입경로: bias_torque_* 키 명시 + intensity=1.0(ramp 타깃).
    #    밴드 3점(하1.34/중1.37/상1.40) × {aggressive, circle} + 무공격 FA baseline.
    eval_scenarios: List[dict] = field(default_factory=lambda: [
        {'pattern': 'aggressive', 'attack_type': 'none',
         'attack_intensity': 0.0, 'attack_start_step': 0,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 밴드 하단 1.34 (aggressive)
        {'pattern': 'aggressive', 'attack_type': 'loe_combined',
         'attack_intensity': 1.0, 'attack_start_step': 60,
         'bias_torque_xy': 1.34, 'bias_torque_z': 0.268, 'bias_thrust_n': 2.010, 'bias_scale': 1.34,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 밴드 중심 1.37 (circle — 패턴 일반화)
        {'pattern': 'circle', 'attack_type': 'loe_combined',
         'attack_intensity': 1.0, 'attack_start_step': 60,
         'bias_torque_xy': 1.37, 'bias_torque_z': 0.274, 'bias_thrust_n': 2.055, 'bias_scale': 1.37,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 밴드 상단 1.40 (aggressive)
        {'pattern': 'aggressive', 'attack_type': 'loe_combined',
         'attack_intensity': 1.0, 'attack_start_step': 60,
         'bias_torque_xy': 1.40, 'bias_torque_z': 0.280, 'bias_thrust_n': 2.100, 'bias_scale': 1.40,
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
    sweep_pattern: str = 'aggressive'          # track 셀 비행패턴(명령토크 최대=최악조건). --sweep-pattern 로 override
    sweep_wind_type: str = 'none'              # sweep 외란 타입 none/wind_constant/wind_gust/wind_turbulence. --sweep-wind-type
    sweep_wind_speed: float = 0.0              # sweep 바람 속도(m/s); force≈0.031·v² N. --sweep-wind-speed
    sweep_episodes: int = 8                    # 셀당 반복(RNG 노이즈)
    sweep_attack_start: int = 30               # 공격 ON 스텝(@10Hz). 이후 ramp
    # ── 조건 C: 지연 호버(delayed hover) — "공격 시작 후 d스텝 뒤 호버 전환" 정책 ──
    #   추적 기동 관성+교란 자세를 안고 현재위치 호버로 전환하는 '전이 케이스'를 검증.
    #   생존율 vs d 곡선 = 탐지 데드라인. 실측 탐지지연(~3스텝)에서 생존해야 RL 프레이밍 성립.
    #   빈 튜플 () 로 두면 기존 A/B(track/hover) 셀만 실행.
    sweep_hover_delays: Tuple[int, ...] = (1, 2, 3, 4, 5, 8)

    # ── NIS 기준선 캡처 격자 (2026-07-20; 바람설계용). capture_mode='normal'|'hijack'|None ──
    #   셀마다 (pattern, disturbance, wind_speed, bias) 지정 → 한 sim 로드로 전 격자 순회.
    #   관측 정의 불변(12D live-P NIS, 로그압축) — 로거 컬럼만 확장.
    capture_mode: str = ''                       # '' | 'normal' | 'hijack' (--capture-mode)
    capture_patterns: List[str] = field(default_factory=lambda: [
        'hover', 'circle', 'figure8', 'waypoint', 'aggressive'])
    #   외란조건: (type, wind_speed). none(0) + {constant,turbulence,gust}×{3,5,7}
    capture_disturbances: List[Tuple[str, float]] = field(default_factory=lambda: [
        ('none', 0.0),
        ('wind_constant', 3.0), ('wind_constant', 5.0), ('wind_constant', 7.0),
        ('wind_turbulence', 3.0), ('wind_turbulence', 5.0), ('wind_turbulence', 7.0),
        ('wind_gust', 3.0), ('wind_gust', 5.0), ('wind_gust', 7.0)])
    capture_biases_hijack: List[float] = field(default_factory=lambda: [1.34, 1.37, 1.40])

    # ── 마감 재측정 격자 (2026-07-22; capture_mode='deadline'). 패턴 고정·지연만 스윕(패턴/지연 교란 제거) ──
    #   셀 = pattern × bias × delay_condition, disturbance=none. delay=dhover{d}, no_response=track(대조군).
    #   = 3 × 3 × 5 = 45 cells × 12ep = 540ep. 산출: 패턴별 지연-생존율 곡선 + 패턴별 마감(생존95% 최대지연).
    deadline_patterns: List[str] = field(default_factory=lambda: ['hover', 'waypoint', 'figure8'])
    deadline_delays: Tuple[int, ...] = (0, 3, 5, 7)     # dhover 전환지연(스텝). + no_response(track) 자동추가.
    deadline_biases: List[float] = field(default_factory=lambda: [1.34, 1.37, 1.40])

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_init
        self.r_inv = 1.0 / (self.r_init ** 2)
        # ★2026-09-18 env 변수 덮어쓰기 전면 폐기 → YAML(configs/*.yaml) + cfgload.py 가 값을 넣는다.
        #   관측 차원은 obs 설정(env/observation.ObsSpec)이 정하고 cfgload 가 dimS/window_size 를 맞춘다.
        self._gyro_only = bool(self.gyro_only)
        self.dimS = self.window_size * (2 if self._gyro_only else 3)
        if self.obs_scale is None or len(self.obs_scale) != self.dimS:
            self.obs_scale = [1.0] * self.dimS
        # ★2026-08-24: eval 시나리오를 확정분포(tilt·연속δ·약/중/강 × 패턴 × 바람)로 재생성.
        #   옛 loe_combined 고정 폐기. 난이도별 분리평가(약=POMDP, 강=easy). 지속공격(60~end, 결정론적).
        import math as _m
        A = self.attack_tq_authority_nm; e = self.episode_max_steps - 5
        def _tilt(delta, alpha, pat, ws):
            roll, pitch = delta*A*_m.cos(alpha), delta*A*_m.sin(alpha)
            return {'pattern': pat, 'attack_type': 'tilt' if delta > 0 else 'none',
                    'attack_intensity': 1.0 if delta > 0 else 0.0,
                    'attack_bursts': ([(s, min(s+10,e), roll, pitch) for s in range(60,e,30)] if delta > 0 else []),  # burst ON10/OFF20
                    'bias_scale': delta, 'attack_direction': alpha,
                    'attack_start_step': 60 if delta > 0 else 0, 'attack_end_step': e,
                    'disturbance_type': 'wind_turbulence' if ws > 0 else 'none', 'wind_speed': ws}
        WS = 8.0  # 강풍 고정(vel POMDP 노출)
        self.eval_scenarios = [
            _tilt(0.0, 0, 'aggressive', 0.0),   _tilt(0.0, 0, 'aggressive', WS),   _tilt(0.0, 0, 'circle', WS),     # 평시(FA baseline, 무풍/강풍)
            _tilt(0.15, 0, 'aggressive', 0.0),  _tilt(0.15, 0, 'circle', WS),      _tilt(0.15, 0, 'aggressive', WS),# 약공격(POMDP)
            _tilt(0.40, 0, 'aggressive', 0.0),  _tilt(0.40, 0, 'circle', 0.0),     _tilt(0.40, 0, 'aggressive', WS),# 중공격
            _tilt(0.70, 0, 'aggressive', 0.0),  _tilt(0.70, 0, 'circle', 0.0),                                     # 강공격(easy)
            _tilt(0.40, _m.pi/4, 'aggressive', 0.0),                                                              # 동시축(방향 일반화)
        ]
        # (결과 폴더는 train.py 가 만든다 — 설정 객체 생성만으로 ./results 를 만들지 않음, 09-18)


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
# ★2026-09-18 구 Isaac 시나리오 샘플러(sample_episode_scenario, env 변수 의존) 제거 → env/scenario.py(sample_isaac_scenario)


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
