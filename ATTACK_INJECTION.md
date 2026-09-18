# 공격 주입 설계 — additive attack on control input (torque/thrust)

> 작성 2026-08-08. 근거: `run_sim.py`, `swrl_config.py`, `env/ukf_filter.py`,
> `online_rl_main.py`, `~/PX4-Autopilot/src/modules/control_allocator/ControlAllocator.cpp`.
> 모든 주장은 `파일:행` 으로 검증 가능하게 달았다.
>
> 이 문서 하나로 "공격을 어디에·어떻게 주입하는가"를 sim·실기 양쪽에서 고정한다.
> **핵심: 공격은 모터 출력(u1~u4)이 아니라 다이나믹스 제어입력(wrench)에 더한다.**

---

## 0. 한 줄 요약 + 불변식

UAV 강체 다이나믹스의 제어입력은 **4차원 wrench** `u = [τ_roll, τ_pitch, τ_yaw, F_thrust]`
(개별 모터 명령이 아니다). 공격은 이 `u` 에 **가산 바이어스** `δ` 를 더한다:

```
컨트롤러 발행:   u                    ← UKF 가 이걸 읽는다 (깨끗)
기체가 겪는 것:   u + δ                ← 공격 δ 가 더해진 제어입력
```

이게 성립하려면 **주입 지점이 UKF 가 `u` 로 읽는 topic 보다 하류**여야 한다.

> ★★ 불변식 ★★
> **공격 δ 는 UKF 가 `u` 로 읽는 `vehicle_torque/thrust_setpoint` 보다 하류에서,
> 실제 플랜트 wrench 에만 더해진다.**
> setpoint 자체를 건드리면 UKF 가 δ 를 그대로 `u` 로 보므로 예측=측정 → 잔차 0
> → NIS 안 뜸 → **탐지할 것이 사라진다.**

이 불변식이 "젯슨에서 setpoint 재발행"과 "케이블 신호 개조"를 둘 다 탈락시키고,
**PX4 control_allocator 의 wrench 입력단**을 유일한 정답으로 남긴다(§5).

---

## 1. 왜 제어입력이 wrench 인가 (모터가 아니다)

강체 다이나믹스:
```
v̇ = (1/m)·R(q)·[0,0,-F_thrust] + g + f_drag/m
ω̇ = I⁻¹·(τ − ω×Iω)                              τ = [τ_roll, τ_pitch, τ_yaw]
```
상태 `x` 를 미분시키는 입력은 `(τ, F_thrust)` 4개다. 모터 u1~u4 는 이 wrench 를
만들어내는 **하위 실행기**일 뿐이고, allocation(믹서)이 `wrench → 모터` 를 담당한다:

```
τ, F  ──[control allocation / mixer]──▶  u1..u4  ──▶ ESC ──▶ 로터
 └ 다이나믹스 제어입력 (여기에 δ)                     └ 실행 계층
```

UKF 도 이 wrench 로 예측한다 — `ukf_filter.py:_f` (168-176행):
```
s[9]  += ((I[1]-I[2])/I[0]·q·r + u[1]/I[0]) · dt     # ω̇_roll  ← u[1]=τ_roll
s[10] += ((I[2]-I[0])/I[1]·p·r + u[2]/I[1]) · dt     # ω̇_pitch ← u[2]=τ_pitch
s[11] += ((I[0]-I[1])/I[2]·p·q + u[3]/I[2]) · dt     # ω̇_yaw   ← u[3]=τ_yaw
f_thrust_body = [0,0,-u[0]]                          #          ← u[0]=F_thrust
```
`u` 는 `to_physical_u`(`ukf_filter.py:44-54`)가 PX4 정규화 setpoint 를
물리단위로 변환한 것: `u[0]=|thrust_z|·C_thrust`, `u[1]=τx·C_torque_x` …

→ **공격을 다이나믹스 제어입력에 넣는다 = 이 4개 `u` 성분에 δ 를 더한다.**
sim·실기 둘 다 여기에 넣어야 같은 공격이 된다.

---

## 2. additive attack 정의

```
ẋ = f(x, u),   u = [τ_roll, τ_pitch, τ_yaw, F_thrust]

플랜트:   ẋ = f(x, u + δ)          δ = [δτ_roll, δτ_pitch, δτ_yaw, δF_thrust]
UKF:     x̂̇ = f(x̂, u)             (δ 를 모름)

innovation = z(측정; u+δ 결과) − ẑ(예측; u)      →  δ 가 만든 불일치  →  NIS
```

δ 의 형태는 `swrl_config.compute_attack_forces`(478-501행)가 정의한다. 공격유형별:
```
loe_thrust  : δF = −intensity·bias_thrust_n           (추력 손실 → vel NIS)
loe_roll    : δτ_x = intensity·bias_torque_xy         (롤 편향 → gyro NIS)
loe_pitch   : δτ_y = −intensity·bias_torque_xy
loe_yaw     : δτ_z = intensity·bias_torque_z
loe_combined: δτ_x, δτ_y, δτ_z, δF 전부 (동결 채널)     (vel·gyro 동시)
```
`sweep_bias_vector`(504-517행)가 (mode, value) → (bias_torque_xy, bias_torque_z,
bias_thrust_n) 로 크기를 정한다. `intensity` 는 `compute_attack_ramp` 출력(0~1).
**동결값: combined, ft_ratio=1.5 — 단, 재정합 후 재측정 대상**(CLAUDE.md 참조).

이것이 POMDP/QCD stopping 프레이밍의 전제다: 공격은 관측(NIS)에 **잔차로만** 드러나고,
정책은 그 잔차 패턴으로 track/hover 를 결정한다. δ 를 상태에서 직접 볼 수 없는 것이 핵심.

---

## 2b. ★ 논문용 공격모델 정당화 (프레이밍 · 레퍼런스, 2026-08-20)

리뷰 방어의 핵심: **wrench 공격을 "새로운 공격 표면"이 아니라 "attack representation/abstraction"으로 주장한다.**
그러면 "wrench attack UAV 논문이 없다"는 문제가 사라진다.

### 공식 정의 (논문 표기)
```
Attack model        : u_a = u_c + a_u,   a_u = [0, a_φ, a_θ, 0]ᵀ,   |a_φ|,|a_θ| ≤ a_max
Physical realization: f_a = A(u_a),  A(·) = nominal control allocation (공격자가 개별 actuator 조합 안 정함)
Attack objective    : bounded roll/pitch deviation WITHOUT loss of flight stability
                      (non-destructive but mission-impacting; a_T=0 고도유지·a_ψ=0 요 공격 X)
Detection           : estimator 는 u_c 를 알지만 plant 엔 u_a 적용
                      x̂_{k+1}=F(x̂_k, u_c)  vs  x_{k+1}=F(x_k, u_c+a_u)  → innovation 변화 → SAC(stopping)
```
공격자는 "motor i 를 X%"가 아니라 **"특정 방향의 bounded trajectory deviation 유도"**로 공격을 정의한다.
allocator 를 **공격하지 않고**(정상 작동) 통과시키므로, actuator 조합의 무수한 경우의 수 문제를 allocator 가 푼다.

### 용어 — "control-input FDI" 써도 된다 (zhu 2026 형식 일치, ★정정)
control-input 에 FDI 를 쓰는 **직접 선례**가 있다 (zhu 2026 등 CPS 문헌):
```
표준 control-input FDI :  ū_k = u_k + Γ_k μ_k
   u_k = controller output(generalized wrench),  μ_k = injected bias,  Γ_k = 공격채널 선택행렬
우리 attack            :  a_u = Γ μ,  Γ = diag(0,1,1,0) (roll/pitch만),  + 버스트 γ_k, bounded |μ|≤a_max
                          → a_{u,k} = γ_k · Γ · μ  = 정확히 zhu 형식
```
- ✅ **"control-input FDI"** 사용 가능. **단 Γ_k 로 "제어입력 채널"임을 명확히 정의** → 센서 FDI 와의 모호성 제거.
- **zhu 2026 을 formulation 근거로 직접 인용** → "wrench/FDI 논문 없음" 문제가 완전히 사라진다(형식 직접 일치).
- 병기 가능: "generalized control-input injection". 하지만 zhu 형식이 있으면 **FDI 가 더 강함**(표준수식+직접인용).
- (구 조언 "FDI 앞세우지 말 것"은 naive 리뷰어 대비 보수론 — zhu 선례로 완화됨.)

### 레퍼런스 3축 (각 축의 역할이 다르다 — ★)
```
① 공격 가능성(Attack surface)  : 기존 UAV actuator/control-channel cyber attack 문헌
                                 → f_a = f + a_f 가 실제 공격모델로 쓰여왔다 = "현실적이다" 증명
② 표현 방법(Representation)     : quadrotor generalized input u=Bf + control allocation
                                 → actuator 공격의 물리효과를 a_u = B·a_f 로 generalized-input 공간에 표현
                                 ⚠ ②는 "공격 가능성"을 증명하지 않는다. 표현 정당화용. 가능성은 ①.
③ 목적(Objective)              : stealthy / deception / impact-limited attack 문헌
                                 → 파괴 없이 성능악화·원하는 궤적유도 가능 = trajectory deviation 정당화
```
구체 후보: ① Secure LQG under FDI (IET 2022)·Enforcing Safety under Actuator Attacks (arXiv 2111.09396)·
FDI detection/estimation for UAVs · ② Robust FTC for quadrotor (RAS 2024) · ③ stealthy attacks quadrotor
(ScienceDirect 2025)·attacks on desired trajectory·undetectable sensor/actuator attacks. 직접 선행: QUADFormer·EADR.

### 논문 서술 (템플릿)
> "Existing work supports the realism of UAV actuator/control-channel attacks [①]. Quadrotor control
> allocation relates actuator commands to generalized forces/moments [②]. We express this attack effect in
> the generalized control-input space, constrained to the roll/pitch channels, as a **bounded** trajectory-
> deviation attack; the actual actuator commands are produced by the nominal allocator [③]."

### "safe" 아니라 "bounded" (★)
`|a_φ|,|a_θ| ≤ 0.8` 이 자동으로 safe 가 아니다(속도·자세·추력·포화·flight envelope 의존).
- ✅ **"bounded attack"** 로 정의 → 실험에서 **"the selected bounds did not cause loss of flight stability"** 를 보인다.
- **bounded ≠ guaranteed safe.** (windthreshold 등으로 "이 밴드에서 추락 안 함" 실증 → δ0.4-0.8 근거)

### 우리 구현과 일치 (코드 수정 불필요)
틸트 = `a_u=[0,a_φ,a_θ,0]` ✓ · δ0.4-0.8 = bounded ✓ · c[0] 주입 = allocator 가 `f_a=A(u_a)` 처리 ✓ ·
UKF 가 `u_c` 읽고 plant 는 `u_a` = 탐지 불변식(§0) ✓.

---

## 3. ★ 시뮬레이션 구현 — allocator 통일 (2026-08-08 개편)

**★★ 외부 wrench 방식 폐기.** 이제 sim 도 실기와 **완전히 같은 경로**로 주입한다:
companion → DDS `actuator_attack` → **PX4 SITL ControlAllocator c[0]** → actuator_motors
→ Pegasus 로터모델 → Isaac 물리. 즉 배분·포화·MOTOR_TAU·추력곡선 결합이 sim·실기에서 같다.

### 3-1. 왜 바꿨나 (외부 wrench 의 한계)
구 방식은 `run_sim` 이 공격 wrench 를 Isaac 강체에 **직접** 얹었다(모터 우회). 그래서:
- 모터 **포화**를 무시 — 급기동+강한 공격(=밴드 엣지)에서 공격을 과대 적용
- **MOTOR_TAU**(23ms)가 공격엔 안 걸림 — 회전동역학이 실기와 다름
- 추력곡선 **결합**(롤 δ가 총추력·요로 샘) 없음 — 순수 토크만
- world frame 이라 급기동에서 body frame 인 다이나믹스와 어긋남
→ 밴드는 이 차이가 큰 영역에서 측정되므로 **전이가 안 된다.** allocator 로 통일해 해결.
  (부수 효과: allocator 토크는 **body frame 네이티브** → 구 world-frame 불일치도 소멸.)

### 3-2. 새 경로 (검증됨)
```
companion (online_rl / rc_attack_trigger)
    → /fmu/in/actuator_attack (ActuatorAttack: active, torque[3], thrust; 정규화)
    → uXRCE-DDS → PX4 SITL uORB actuator_attack
    → ControlAllocator::apply_attack_dds()  c[0] 에 가산  (ControlAllocator.cpp)
    → actuator_motors → MAVLink → Pegasus 로터모델 → Isaac 힘
run_sim.py: 공격 외부 wrench 제거(바람만 유지). /attack_config 는 GT 라벨링 전용.
```

### 3-3. 불변식은 그대로 (UKF 의 u 는 깨끗)
allocator 는 **배분 직전 c[0]** 에 δ 를 더한다. UKF 가 읽는 `vehicle_torque_setpoint` 은
레이트 컨트롤러가 **상류에서** 발행하므로 손 안 탄다. 센서(gyro/GPS)는 공격받은 모터힘으로
움직인 기체를 반영. → innovation = z(u+δ) − ẑ(u). = 불변식 만족 (실기와 동일).

### 3-4. 트리거 (누가 δ 를 켜나) — §6
- 수동 리허설(F9): `rc_attack_trigger.py` 가 RC aux 읽어 ActuatorAttack 발행
- 학습/스윕: `online_rl_main._send_attack_cmd` 가 step 기반으로 ActuatorAttack 발행
  (물리 bias N·m → 정규화 = bias / 권한. `swrl_config.attack_*_authority`, PROVISIONAL)
- 실기: PX4 allocator 가 RC aux 를 **직접** 읽음 (companion 불필요) + DDS 도 가능

---

## 4. NIS 는 어떻게 생성·유지되나 (Q/R 고집)

`ukf_filter.py:step` (180-227행). NIS = `compute_nis_scaled`(67-77행):
```
nis_raw = rᵀ · Pzz⁻¹ · r / nz        r = z − z_bar (innovation),  Pzz = R_eff + 측정spread
```

### 4-1. innovation 이 왜 생기나
공격 δ 로 참상태가 모델 예측에서 벌어진다 → z(측정) ≠ ẑ(예측) → `r ≠ 0`. (§2)

### 4-2. innovation 이 왜 **유지**되나 — 고집 = 낮은 이득
칼만 이득 `K = Pxz·Pzz⁻¹` 는 예측공분산 `P_bar = Q + ff²·spread` 에서 나온다.
- **반응형 필터**(Q 높음/R 낮음 → K 큼): 추정을 측정으로 빠르게 끌어당김 → `r` 1~2스텝에
  붕괴 → 공격이 과도 스파이크 후 사라짐 (= "펄스 후 침묵" 문제, 물리#2).
- **고집 필터**(Q 낮음/R 높음 → K 작음): 추정이 모델 궤적에 머묾 → **공격이 지속되는 한
  `r` 이 크게 유지** → NIS 가 높게 유지된다.

즉 **의도적 비일관 필터**로 통계적 일관성을 버리고 **지속적 탐지신호**를 얻는다. 그래서
여기 NIS 는 χ² 통계량이 아니라 **탐지 feature** 다 (평균이 nz 로 정규화되지 않는다).

### 4-3. 고집의 실제 구현 (검증)
```
[110-115]  Q = diag(pos 1e-3, euler 5e-4, vel 5e-3, gyro 5e-4)    ← ★낮은 Q = 고집의 주 레버
[117-121]  R = diag(pos 0.5, vel 0.3, gyro 0.5)                   ← 높은 R = 보조 (GPS/vel 불신)
[93]       ff = 1.0                                               ← fading-memory 끔
```
`ukf_filter.py:81` 주석 그대로: **"고집은 low Q 로."** fading-memory(ff<1)는 P 를 재팽창시켜
이득↑ + NIS분모(Pzz)↑ 를 동시에 만들어 둘 다 손해라 끈다.

### 4-4. 두 개의 보정
- **멀티레이트 GPS 게이트**(`step` gps_fresh, 205-210행): GPS 는 10Hz 인데 루프는 50Hz.
  낡은 스텝엔 `R[0:6] *= 1e6` 로 GPS 이득 ≈0. ⚠ 구현 전엔 같은 GPS 를 5회 재사용(ZOH)해
  실효 R 이 1/5 로 작아져 **고집 설계와 반대**였다(2026-08-05 수정, CLAUDE.md).
- **maneuver-gated Q**(`q_gate`, 194-201행): 명령 토크 `|u[1:4]|` 만큼 gyro Q 인플레.
  명령된 각가속은 "설명"(FP↓), 명령 안 한 이탈(공격)만 NIS 로 남긴다.

---

## 5. 실기 구현 — PX4 control_allocator 의 wrench 입력단

`ControlAllocator.cpp` 가 sim 과 데칼코마니다. 레이트 컨트롤러가 발행한
`vehicle_torque/thrust_setpoint`(UKF 가 읽는 그 topic)을 받아 **배분 직전 벡터 c 에 담는다**:
```
[405-410]  c[0](0)=τx  c[0](1)=τy  c[0](2)=τz          ← 다이나믹스 제어입력 (body frame)
           c[0](3)=Fx  c[0](4)=Fy  c[0](5)=Fz
[428]      setControlSetpoint(c[0]) → allocate() → actuator_motors → ESC
```
여기 `c[0]` 은 sim 의 `compute_attack_forces` 가 더하는 그 wrench 와 **같은 물리량**이다.
`vehicle_torque_setpoint` topic 자체는 상류(레이트 컨트롤러)가 발행하므로, allocator 안에서
c 에 δ 를 더해도 **UKF 가 읽는 값은 깨끗**하다 = 불변식 만족. sim 과 수학적으로 동일.

### 구현 4단계
1. **커스텀 uORB 메시지** `actuator_attack.msg`
   ```
   uint64 timestamp
   bool   active
   float32[3] torque_bias      # N·m, body frame  = (δτ_x, δτ_y, δτ_z)
   float32    thrust_bias      # N,   body -z      = δF
   ```
2. **`dds_topics.yaml` subscription 추가** (젯슨→FC, 오프보드 setpoint 가 흐르는 경로):
   ```yaml
   subscriptions:
     - topic: /fmu/in/actuator_attack
       type: px4_msgs::msg::ActuatorAttack
   ```
3. **`ControlAllocator::Run()` 의 `c[0]` 조립 직후(410행 뒤) 가산**:
   ```cpp
   if (attack_gate_open() && _attack.active) {          // 게이트: RC + armed + 비행중
       c[0](0) += _attack.torque_bias[0];
       c[0](1) += _attack.torque_bias[1];
       c[0](2) += _attack.torque_bias[2];
       c[0](5) += _attack.thrust_bias;                  // thrust 는 c(5)=xyz[2] (body -z)
   }
   ```
4. **바이어스 크기의 출처 — §5-1 참조.**

### 5-1. 바이어스는 PX4 파라미터로 한다 (확정)
동결 밴드값은 sim 에서 이미 정해진 **상수**다(실기에서 스윕할 이유 없음). 그래서:
- δ 크기 = **PX4 파라미터** `ATK_T_ROLL / ATK_T_PITCH / ATK_T_YAW / ATK_F_THR`
  (지상에서 동결값으로 세팅. ulog params 에 남아 **자동으로 재현·기록**된다)
- **ramp 는 모듈이 내부에서 계산**(`compute_attack_ramp` 와 동일 식). 파라미터는 **목표 δ**,
  RC 스위치가 트리거, 모듈이 0→목표로 올린다. step 공격이면 ramp=0(즉시).
- 확장(나중에 실기에서 δ 를 바꿔야 하면): 파라미터 대신 §5 의 DDS topic 으로 젯슨이 넘긴다
  (= sim 의 companion 발행 모델과 문자 그대로 일치). 첫 테스트엔 파라미터가 가장 안전.

  대응관계:  **sim `/attack_config` 의 bias 필드  ≡  실기 PX4 파라미터 δ**

### 5-2. 안전 가드 (attack_gate_open)
δ 는 다음을 **전부** 만족할 때만 적용:
- armed **이고** 비행 중 (지상 발사 방지)
- RC aux 스위치 up (`manual_control_setpoint` aux1 > 0.5) — **하드웨어 중단 경로**
- 오프보드/자동 비행 모드 (수동 복귀 시 즉시 δ=0)
스위치 내림 / 모드전환 → 즉시 δ=0. SW 무관한 물리적 kill 을 조종사가 항상 쥔다.

---

## 6. 트리거 설계 — RC 게이트 + 사후 정렬

**"무엇을 주입하나(δ, ramp)"와 "언제 쏘나(트리거)"는 분리한다.** δ 정의는 공유
(`sweep_bias_vector` + `compute_attack_ramp`), 트리거만 맥락별로 다르다.

| 맥락 | 트리거 (온셋 이벤트) | 발행자 |
|---|---|---|
| 학습/스윕 (자동) | 에피소드 **step 인덱스** | `online_rl_main._send_attack_cmd` |
| Isaac 수동 리허설 | **RC/조이스틱 스위치 flip** | `rc_attack_trigger.py` (신규) → `/attack_config` |
| 실기 | **RC 스위치 flip** | PX4 모듈 게이트 (§5-2) |

### RC 스위치를 트리거로, 온셋 시각은 로그로 복원 (시간 기반 안 씀)
| 필요 | RC 스위치 | 순수 시간 기반 |
|---|---|---|
| 안전한 순간 발사(고도·위치·바람) | ✓ 조종사가 고름 | ✗ 상황 무관 발사 |
| **즉시 물리적 중단** | ✓ 하드웨어 δ=0 | △ 모드전환 필요, 느림 |
| 온셋 정밀도(TTD) | ✓ flip 을 ms 로 로깅 | ✓ |
| 재현성 | ✓ 사후 로그 정렬 | ✓ |

시간 기반의 유일한 이점("사람 없이 정해진 순간 발사")은 **사후 정렬로 대체**되므로 불필요하고,
"불안정해도 SW 무관하게 계속 쏨" 위험만 남는다. → 실기에선 RC 게이트가 정답.

**개념 통일**: 온셋 = 타임스탬프 찍는 이산 이벤트. sim=step 인덱스, 실기=RC flip.
onset 이 crisp(step, ramp=0)한 것도 양쪽 동일 — flip 이 곧 step 공격의 온셋.

### sim 수정 범위 (수동 리허설용)
`run_sim` 은 이미 `/attack_config` 를 소비하므로 **코드 0줄 수정.** 발행자만 추가:
`rc_attack_trigger.py` — 조이스틱 버튼(오프보드 버튼처럼 매핑) 또는 `manual_control_setpoint`
aux 를 읽어 up→`{active:true, bias:동결값}` / down→`{active:false}` 발행. 실기 RC 게이트와 1:1.

---

## 6-2. F9 — 실기 공격 주입 가능성 테스트 (RC 트리거)  ★구현됨 2026-08-08

목적: **스윕이 아니라 "공격 주입 경로가 실기에서 도는가"의 가능성 검증.**
아주 약한(권한 3~8%) 공격을 RC 로 켜고, NIS 가 뜨는지(식별 가능성)만 본다.

### ★ 크기는 정규화값 = "제어권한 대비 비율" (핵심)
allocator 의 `c[0]` 은 **정규화 토크/추력** [-1,1] 이다
(`vehicle_torque_setpoint.msg`: "normalized", `MulticopterRateControl.cpp:253` 에서 [-1,1] clamp).
따라서 δ 는 N·m 이 아니라 **정규화값 = 권한의 몇 %** 다. 이건 CLAUDE.md 가 밴드를
기록하라는 바로 그 표현이다 (동결 s=1.34 → 롤 권한의 31%). **N·m 환산 불필요.**
- sim·실기 **양쪽 다 정규화값**을 c[0] 에 넣는다 (allocator 통일 후 환산 불필요해짐).
  `ATK_TQ_MAX = 0.08` → full-knob 에서 롤 권한의 8%. rc_attack_trigger 도 정규화 그대로 발행.
- online_rl(스윕)만 기존 물리 bias(N·m)를 정규화로 환산해 발행(`attack_*_authority`, PROVISIONAL).
  밴드 재측정 시 정규화로 재정의하면 이 환산도 불필요.
- **F9 에선 값 자체가 안 중요하다.** 경로·절차가 도는지가 목적.

### 트리거: RC 스위치(게이트) + 노브(크기)  — "스위치는 뭐로?"
- **enable 스위치** = 송신기 여분 2단 스위치(예: SC/SD) → `RC_MAP_AUX1` → `aux1`.
  up(>0.5)=무장, down=**즉시 δ=0**(하드웨어 중단, SW 무관). 이게 안전판.
- **크기 노브** = 다이얼/슬라이더(예: S1/S2) → `RC_MAP_AUX2`(토크)/`AUX3`(추력) → `aux2/3`.
  0 에서 천천히 올리며 NIS 를 보고 "약하지만 식별 가능" 지점을 **비행 중 실측.**
- 축은 한 번에 하나: `ATK_TQ_AXIS` (F9-1 roll / F9-2 는 thrust 노브) → gyro vs vel 시그니처 분리.

### PX4 구현 (SITL 빌드·검증 완료 2026-08-08, 실기 빌드·플래시 남음)
`~/PX4-Autopilot` 에 패치됨 (`px4_field/ATTACK_FIRMWARE_BUILD.txt` 에 빌드 절차):
```
msg/ActuatorAttack.msg                DDS 공격 메시지 (active, torque[3], thrust; 정규화)
msg/CMakeLists.txt                    등록
dds_topics.yaml                       /fmu/in/actuator_attack 구독
ControlAllocator.cpp                  apply_attack / _rc(RC) / _dds(DDS) — c[0] 가산
ControlAllocator.hpp                  구독 2개(manual_control, actuator_attack) + 선언
module.yaml                           ATK_EN/AUX_SW/AUX_TQ/AUX_TH/TQ_AXIS/TQ_MAX/TH_MAX
                                      (기본 ATK_EN=0 → stock 동일)
```
게이트: `ATK_EN=1 ∧ armed`.  RC: aux[ATK_AUX_SW]>0.5.  DDS: actuator_attack.active.
✅ SITL 빌드 성공: bin/px4, ATK 파라미터 7개, actuator_attack uORB 생성 확인.
⚠ 실기는 `make px4_fmu-v6c_default` + 플래시. `ATK_EN=0` 이면 stock 과 완전 동일.

### sim/companion 구현 (완료·검증)
- **run_sim.py**: 공격 외부 wrench **제거**(바람만 유지). 물리 주입은 이제 PX4 allocator.
- **rc_attack_trigger.py**: RC aux → `/fmu/in/actuator_attack`(ActuatorAttack) 발행.
  `--stdin`(`t 0.3`/`h 0.2`/`on`/`off`/`q`) 폴백. 실행: `launch_isaac_manual.sh f9`.
- **online_rl_main.py**: `_send_attack_cmd` 이 ActuatorAttack(정규화) 도 발행(학습/스윕).
  `/attack_config`(String)은 run_sim GT 라벨링 전용으로 남김.
- **px4_msgs**: ActuatorAttack 재빌드 완료(GPU 서버). 젯슨도 재빌드 필요(위 TXT [7]).

### 안전
- RC 스위치 down = 즉시 δ=0 (실기 하드웨어 경로 / sim 노드 종료 시 off 발행)
- armed 아니면 무시(지상 발사 방지). ATK_TQ_MAX/TH_MAX 로 상한(full-knob 도 8%)
- 첫 테스트는 **고도 여유·바람 잔잔·roll 축·3~5%** 부터. 불편하면 노브 0 또는 스위치 off.

---

## 7. 요약 — 무엇이 어디서 동일한가  (2026-08-08 allocator 통일)

```
                 sim (학습)          sim (수동 리허설)        실기
 주입 지점        PX4 allocator c[0]   PX4 allocator c[0]      PX4 allocator c[0]
                 = 다이나믹스 제어입력  = 동일                  = 동일 (전부 body frame)
 발행 소스        online_rl DDS        rc_attack_trigger DDS   RC aux (+DDS 가능)
                 (ActuatorAttack)      (ActuatorAttack)        (allocator 직접 읽음)
 UKF 의 u        깨끗 (setpoint)       깨끗                    깨끗 (상류 발행)
 불변식 만족?     ✓                    ✓                       ✓
```

**구현 상태 (2026-08-08~09)**:
- ✅ PX4 SITL 빌드·검증 (allocator RC+DDS, msg, yaml, params 7개) — GPU 서버
- ✅ px4_msgs 재빌드 (ActuatorAttack) — GPU 서버
- ✅ run_sim 외부 wrench 제거 / rc_attack_trigger·online_rl → DDS 발행 (topic 검증)
- ✅ ★ **SITL 실동작 검증 (2026-08-09, Isaac headless + 오프보드 이륙)**:
    · 오프보드로 3m 호버 → DDS `/fmu/in/actuator_attack` 주입 → /gt/odometry 관찰
    · roll 토크 δ=+0.12(정규화) → **roll +4.85° 편향**, off 시 회복 → 주입이 기체에 먹힘 확인
    · thrust δ=+0.12 → **고도 −0.30m** = lift 감소(loe_thrust) → **부호 정확**(코드 그대로, 반전 불요)
    · PX4 가 roll 을 ~5°에서 잡고 발산 안 함 = 폐루프 보상 = 지속 NIS 생성에 딱 맞는 거동
    · 경로 전체 확인: 오프보드→DDS→apply_attack_dds→c[0]→actuator_motors→Pegasus→Isaac
- ✅ ★ **F9 Isaac 수동조종 리허설 검증 (2026-08-10, 조종기 수동 호버 + DDS 주입)**:
    · 조종기로 3m 호버 → rc_attack_trigger(--stdin DDS)로 단계 주입 → /gt/odometry 측정
    · roll 토크: 10%→+3.5° / 20%→+7.2° / 30%→+8.4° / 50%→+17.1°(드리프트 1.0m) — **크기 비례 확정**
    · thrust: 10%→−0.22m / 20%→−0.41m / 30%→−0.55m — **부호·비례 확정**
    · Position 모드가 약한 공격(≤8%)은 완전 상쇄 → "제자리" = 정상(탐지는 NIS로)
    · ★ **rc_attack_trigger dedup 버그 수정**: 값 고정 시 재발행 안 함 → BEST_EFFORT 유실.
      10Hz 연속발행으로 수정(검증). `--tq-max`/`--th-max` 가 DDS 주입 크기(권한 비율)를 정함.
- ✅ 문서·런처 (ATTACK_FIRMWARE_BUILD.txt, RUN.txt, ISAAC_F1_F8.txt, 본 문서)
- ⬜ 실기 `make px4_fmu-v6c_default` + 플래시 (STAGE 2) → 프로펠러 벤치(STAGE 3) → 최종주행(STAGE 4)

**미결/배치 항목**:
- 실기 빌드·플래시 + 젯슨 px4_msgs 재빌드 (ATTACK_FIRMWARE_BUILD.txt)
- F9 정밀조정 (약하지만 식별가능 임계 — NIS 파이프라인 돌 때 실측)
- 동결 밴드 δ **정규화 단위로** 재측정 → `attack_*_authority` 환산 제거 — 재정합 후

참조: `PLANT_AND_MODEL.md`(플랜트 3층), `SIM2REAL_ALIGNMENT.md`(회전축·G), `CLAUDE.md`(로드맵).
