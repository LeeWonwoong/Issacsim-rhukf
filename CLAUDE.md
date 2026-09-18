# CLAUDE.md — 프로젝트 컨텍스트 (sweep 실험 단계)

> ## ★★ 먼저 읽을 것 (2026-08-13)
> **물리값·정합 현황의 단일 출처는 `SIM_ALIGNMENT.md` (날짜 없는 상시 참조본) 다.** (구 `SIM_ALIGN_CHANGES_20260812.md` 대체)
> 이 문서(CLAUDE.md)에는 서사·로드맵·방법론과 함께 **폐기된 옛 수치**가 이력으로 섞여 있다.
> 아래 📦/[폐기] 표시 블록의 숫자는 인용 금지. 계수·밴드·NIS 값이 필요하면 위 문서를 볼 것.
>
> 특히 지금 틀린 옛 서술 4가지 — 본문에도 각각 정정 표시를 달아두었다:
> 1. "sim 로터 설정 = `iris.py`" → **아니다.** `run_sim.py` 의 `config_multirotor` 블록.
>    `run_sim` 은 `IrisConfig` 를 쓰지 않으므로 `iris.py` 수정은 **플랜트에 안 닿는다.**
> 2. "추력-토크 결합항 의도적 미포함" → **복원됨.** 07-29 복원, 08-12 실기 실측값으로 교체.
> 3. `flight_radius 3.5 / agg 2.8` → **2.8 / 2.24** (08-12, 속도 클램프 해소)
> 4. `C_torque_z 4.907/4.41` → **1.378** (08-12 실기 오토튠 Σb/dt 37.3 × Izz)

## 프로젝트 개요
UAV(쿼드로터) 제어입력(액추에이터) 공격 탐지. POMDP/QCD stopping 문제로 정식화,
UKF 잔차(NIS) 관측 기반 DDQN이 매 스텝 track/hover 이진 결정.
최종 기여: Adam 대비 커스텀 2차 최적화기 RHUKF-FV(FIR 철학)의 sparse-attack 샘플효율 우위.
센서 공격은 스코프 밖(센서 무결성 가정). 가장 가까운 선행: EADR(고정 CUSUM, 시나리오별 하드코딩 임계값
— 저자들이 future work으로 RL 지목), QUADFormer(센서 공격+transformer, TTD 미보고).

## ★ 공격 주입 설계 — `ATTACK_INJECTION.md` (2026-08-08 allocator 통일)
additive attack 을 **다이나믹스 제어입력(wrench: τ_roll/pitch/yaw, F_thrust)** 에 더한다 — 모터 u1~u4 아님.
불변식: **δ 는 UKF 가 u 로 읽는 `vehicle_torque/thrust_setpoint` 보다 하류에서 플랜트 wrench 에만** 더해진다
(setpoint 를 건드리면 UKF 가 δ 를 u 로 보고 잔차 0 → 탐지 불가).
- ★★ **외부 wrench 폐기(2026-08-08).** sim·실기 **둘 다 PX4 `ControlAllocator::apply_attack()` 의 c[0]**
  (배분 직전 정규화 wrench)에 가산. 그래야 배분·포화·MOTOR_TAU·추력곡선 결합이 sim=실기 (전이성).
  sim: c[0]→actuator_motors→Pegasus 로터모델→Isaac. 실기: c[0]→ESC. **같은 코드·같은 지점.**
- 두 소스: ① RC aux(manual_control_setpoint, 수동/F9) ② DDS `actuator_attack`(companion, 학습/스윕).
  둘 다 apply_attack 에서 c[0] 가산. 게이트 `ATK_EN=1 ∧ armed`.
- δ 는 **정규화값 = 제어권한 대비 비율**(c[0] 이 정규화[-1,1], N·m 아님. 밴드 표현과 일치). ATK_TQ_MAX 0.08=8%.
- 고집(NIS 유지): **낮은 Q**(주 레버, `ukf_filter.py:110`)+높은 R, fading-memory off. NIS=χ² 아닌 탐지 feature.
- 구현·검증(2026-08-08): PX4 SITL 빌드 성공(ATK 파라미터 7개+actuator_attack uORB), px4_msgs 재빌드,
  `run_sim` 외부wrench 제거(바람만), `rc_attack_trigger`/`online_rl` → ActuatorAttack DDS 발행.
- ★ **SITL 실동작 검증 완료(2026-08-09, Isaac headless)**: 오프보드 3m 호버 → DDS 주입 → /gt/odometry.
  roll δ=+0.12 → roll +4.85°(off 시 회복), thrust δ=+0.12 → 고도 −0.30m = **thrust 부호 정확**(반전 불요).
  PX4 가 roll 을 ~5°에서 잡고 발산 안 함 = 폐루프 보상(지속 NIS 거동). 경로 전체 확인.
  · 실기: `make px4_fmu-v6c_default`+플래시 남음 → `px4_field/ATTACK_FIRMWARE_BUILD.txt`
  · F9(스윕 아닌 가능성 테스트): `RUN.txt`[팩8], 리허설 `ISAAC_F1_F8.txt`§7-2, 런처 `f9`
  · 미결: 식별가능 임계 정밀조정, 밴드 정규화 재측정(`attack_*_authority` PROVISIONAL)

## 시스템
- Isaac Sim 4.5 + PX4 SITL + Pegasus, ROS2 Humble, 10Hz 제어루프
- `online_rl_main.py`(RL/스윕 노드) + `run_sim.py`(Isaac 서브프로세스)
- 관측: 4스텝 윈도우 [nis_vel, nis_gyro, action] = 12차원. 의도적 비일관 UKF(Q낮게/R높게)
  → NIS는 χ² 통계량이 아니라 탐지 feature. **압축 통일(2026-07-22): 전 채널 ε̃=ln(1+ε)/(1+ln(1+ε)) [offset=1.0]** (기존 vel=log0.5 폐기).
  pos NIS(res[0:3],R_pos=0.5)는 **로깅 전용**(정책 입력 아님). 16D 채택 여부는 우선순위2 분리도로 판정.
- 네트워크: **페어링용 [24,24]=962params@12D / 1058@16D** (2026-07-22, 기존 514[16,16]에서 상향). 최적화기 공정비교용 고정. agent: `--agent rhukf|adam`
- **환경 확정(2026-07-22): 바람=none40%/turbulence60%, wind_speed_range=(1.0,5.0)** (constant/gust 제거·캡처A 근거). sim_speed_factor=10(캡처·스윕 기본).

## ★★ 플랜트/모델 정합 3건 수정 (2026-07-28) — 이전 측정치 전부 무효 ★★
사용자 요청("런타임 질량 직접 쿼리")에서 시작해 UKF 모델과 시뮬레이터 플랜트의 불일치 3건을 실측·수정.
**아래 FROZEN 밴드·NIS 기준선·d′·데드라인·학습된 에이전트는 전부 재측정 대상.**

1. **총 비행질량 = 1.372kg 로 정합** (`run_sim._apply_body_calib`, `[MASS]` 로그로 매 기동 확인)
   - Iris USD 는 body 1.5kg + **rotor 4개(리볼루트 조인트 별개 강체) 0.1186kg = 실제 1.6186kg** 였다.
     "기본값 1.5kg"은 바디만의 값 — 로터를 빼먹은 것이 오해의 출발점.
   - 이제 body 1.253385kg 를 써서 **총합이 1.372kg**. 관성도 같은 원칙으로
     구 총관성×(1.372/1.6186) → 총 (0.029547, 0.026484, 0.053359). 로터 x/y 암 길이가 달라 Ixx≠Iyy.
   - ⚠ **실기체 AUW 는 1.326kg 으로 저울 실측 확정됨(2026-08-04)**. 1.372 는 그 전에 AUW 로
     믿었던 값이고, sim 플랜트·calibration.json·C_thrust(25.58)·관성이 전부 1.372 기준으로
     맞춰져 있다. **sim 재정합이 남아 있다**(−3.4%) — 실기 F1/F2/F3 계수와 함께 한 번에 할 것.
     실기 로그 분석(`check_ulog.py`/`fit_from_ulog.py`)의 `--mass` 는 **1.326** 을 쓴다.
   - 검증: 로터모델 예측 호버 u_norm=0.5274 vs 실측 0.52722 (0.03%).
2. **C_thrust / C_torque 재캘리브레이션** (`calibration/fit_static_from_rotor.py`)
   - 구 값은 `calibrate_sysld.py` 가 **가정질량 1.5 를 하드코딩**한 채 적합됐고 calibration.json 의
     drone 만 손으로 1.372 로 고쳐져 갈라져 있었다 → 이제 DRONE 은 항상 calibration.json 에서 읽는다.
   - `C_thrust 22.82 → 25.58` (정적/호버평형/해석/동역학 4경로 0.2% 내 일치)
   - `C_torque_xy 0.265 → x=3.568 / y=4.017`(축별 분리, ukf_filter 폴백 지원), `C_torque_z 3.655 → 4.907`
   - **구 C_torque_xy 는 약 14배 과소**. 구 모델의 최대 롤 각가속도 9 rad/s² vs 실측 피크 52 rad/s²
     → 기동을 원리적으로 예측 못 함 = 기존 "기동↔공격 aliasing" 의 상당 부분이 이 캘리브 오류.
   - ⚠ C_torque 는 drone.I 와 **짝** — I 를 바꾸면 반드시 재적합.
   - 방법 주의: 자세 rate loop 는 폐루프라 `ω̇~τ_cmd` 회귀가 편향된다(R²0.03~0.15, 계수 2배 요동).
     run_sim 이 기록한 **실제 로터 각속도**(env `ROTOR_LOG`)로 적용 토크를 직접 계산하는 경로를 쓸 것.
     로터 배치 비대칭(Σx=+0.027, Σy=−0.008)으로 요·총추력이 롤로 새므로 교차항 필수(R² 0.4→0.88).
3. **오일러각 프레임 버그 수정 (ENU/FLU → NED/FRD)** (`online_rl_main._quat_to_euler`)
   - Isaac GT 쿼터니언(ENU 관성/FLU 바디)에 표준 ZYX 공식을 적용한 각을 NED 기준인 양 소비했다.
     수정 = (φ, −θ, 90°−ψ). 피해가 두 군데:
     (a) UKF 가 추력벡터를 틀린 수평 방향으로 회전 → 기동 중 vel NIS 오염.
         실측: 구 관례 항력적합 [0.13, 0.03]/corr−0.30 → 수정 후 [0.499, 0.299]/corr−1.00
         = Pegasus 실제 설정 `LinearDrag([0.50,0.30,0.0])` 와 일치. **drag 확정 [0.5, 0.3, 0.0]**.
     (b) `_hover_yaw`(ENU) 를 PX4 TrajectorySetpoint(NED yaw)로 보내 **호버 전환마다 약 90° 요 슬루**를
         명령했다. 실측 25회: 전환 후 3s |Δψ| 중앙 **84.5°**(90pct 90.0°) vs 전환 전 0.7°.
         → dhover 데드라인·basin·"호버전환" 클래스 분리도가 전부 이 인공물 위에서 측정된 것.

4. **추력-토크 기하 결합 항** ⚠ **정정: 아래 '미포함' 서술은 폐기됐다.**
   07-29 에 복원됐고(빼면 평시 gyro NIS 바닥의 96% 를 이 항이 차지), 08-12 에 **실기 실측값**
   `K=[+0.00182, +0.01210, −0.00195]` N·m/N 으로 교체됐다(실기 11소티 호버 트림, 편차 14%).
   sim 플랜트도 `run_sim._apply_com_align` 이 바디 COM 을 옮겨 같은 K 를 기하로 재현한다.
   ↓ 아래는 07-28 시점의 (지금은 틀린) 판단 기록
   - `ukf_filter.to_physical_u` 는 `torque_thrust_coupling` 키가 **있으면** 적용, 없으면 무시(현재 없음).
   - 빼는 이유: 실기에선 배터리/페이로드 위치마다 COM 오프셋이 달라 이 항을 맞출 수 없다.
     sim 에서만 정확히 보정하면 **모델오차가 sim 에서만 0** 이 되어 sim↔real 간극을 벌린다.
     빼두면 명목 오프셋이 만드는 현실적 편향이 학습 데이터에 그대로 남아 정책이 견디도록 학습된다.
   - 복원: `validate_by_regime.py <outdir> --fit-coupling --write` (백업 `.bak_with_coupling`)
   - 분리도 영향은 작다: COM ±10mm 상당 편향의 NIS 기여 ≈0.018 (압축 후 분리폭 0.84 대비 ~1%).
   ↓ 아래는 발견 내용 기록
   - Iris 는 로터 중심이 바디 COM 에서 (x+6.7mm, y−1.9mm) 어긋나 **총추력이 상시 토크**를 만든다.
     PX4 는 트림으로 상쇄하지만 UKF 가 모르면 그 트림명령을 실제 토크로 오해 → ω̇ 예측 상시 편향.
   - 실측 편향(수정 전): ω̇y **−3.17 rad/s²**, ω̇x +0.55 — 호버·순항·급기동 **전 영역 공통**(=상수항).
     기하학적 예측(Σx/4·T, Σy/4·T)과 부호·크기 일치. 수정 후 전 영역 편향 ≈0.
   - K = [−0.00121, +0.00621, +0.00013] N·m per N. `validate_by_regime.py --fit-coupling` 로 재추정.

### ★ sim-to-real 설계 원칙 (2026-07-28 확정) ★
**최종 목표는 실기체 inference.** 그래서 판단 기준이 "모델 정확도"가 아니라 **"모델오차 수준의 현실성"** 이다.
정책이 먹는 건 NIS 이고, NIS 수준을 결정하는 건 모델오차의 크기이기 때문.
- 틀린 물리(질량·프레임·C_torque)는 반드시 고친다 — 실기에서도 제대로 캘리브할 것이므로 양쪽 다 맞아야 함.
- 반면 **실기에서 못 맞추는 미세항은 sim 에서도 맞추지 않는다** (예: 추력-토크 결합항). 안 그러면
  모델오차가 sim 에서만 0 이 되어 평시 NIS 기준선이 비현실적으로 낮아지고, 실기에서 상시 오탐이 난다.
- COM 오프셋 랜덤화는 **하지 않음**: 분리도 영향 ~1% 인데 교란축이 늘면 학습곡선 분산이 커져
  본 기여(RHUKF-FV vs Adam **샘플효율** 비교)를 흐린다. 항을 빼는 것만으로 노출은 이미 확보됨.
- 미결(순서 주의):
  · **플랜트 현실성(모터 1차 지연 τ≈30ms, 배터리 새그)** → 플랜트를 바꾸므로 **밴드 재측정 전에** 결정.
  · 캘리브 불일치 랜덤화(C_thrust±5%, C_torque±10%, mass±5%) → 플랜트 무관, 밴드 후 결정 가능.
  · 관측 정규화(비행 초반 기준선으로 NIS 정규화) → 관측 정의 변경, 학습 전 결정.
- **밴드는 절대값이 아니라 제어권한 대비 비율로도 기록할 것** (전이 가능한 표현):
  현 동결 s=1.34 → 토크 1.34 N·m = 롤 권한 4.36 의 **31%**, 추력 2.01 N = 무게 13.46 의 **15%**.
- **캘리브 절차 전이성**: 오늘 쓴 로터 각속도 기반은 sim 전용(실기는 ESC 텔레메트리 필요).
  실기용은 PX4 시스템ID **치프 주입**(개루프 여기 → 폐루프 편향 제거). sim 에 ground truth
  (3.568/4.017/4.907)가 있으므로 **그 절차가 정답을 복원하는지 sim 에서 먼저 증명**해둘 것.

### 검증 결과 (2026-07-28, 4패턴 × 3영역)
고정 계수의 **예측잔차**로 검증(재적합 아님). `validate_by_regime.py`:
```
                  Fx(항력) 편향   Fz(추력) 편향   ω̇y 편향
  hover           +0.0005 N      −0.034 N       −0.12~+0.01
  cruise(1~1.8m/s) ±0.003 N      −0.012~+0.013  +0.08~+0.15
  agile(|ω|>1)    −0.009 N       +0.9~+1.5 N    −0.3~−1.5
```
호버·순항은 전 패턴(waypoint/circle/figure8/aggressive)에서 편향 ≈0 → **캘리브레이션은 기동영역 무관**.
잔차는 급전이(agile) 구간에 몰리는데 원인은 추력곡선 비선형(T=4k(100+1000u)², 호버 secant 25.6 vs
국소기울기 41.7)과 명령/응답 타이밍. **2차 모델로 바꿔봤으나 RMS 가 오히려 악화**(agile 2.18→3.35 N,
명령-응답 시차가 2차항에서 1.6배 증폭) → 선형 C_thrust 유지 결정.

### 교차검증 (독립 경로: calibrate_sysld.py, 가속도계 기반)
```
  C_thrust  25.514  vs 25.580 (−0.26%)   ✓
  drag      [0.438, 0.326, 0.036] vs [0.50, 0.30, 0.00]  ✓ (z≈0 확인)
  C_tq_z     4.502  vs  4.907 (−8%)      ✓
  C_tq_xy    0.255  vs  3.79             ✗ ← 폐루프 편향. 구 값 0.265 가 바로 이 경로 산물.
```
→ 롤/피치만 두 경로가 갈리며, 물리 상한·실측 각가속도·유효관성 3중 검증이 로터 경로를 지지한다.
**교훈: 자세 rate loop 계수는 `calibrate_sysld` 로 뽑지 말 것**(요/추력/항력은 유효).

### 재캘리브레이션 도구 (신규)
- ~~`run_calib_mass.sh`~~ — **존재하지 않음(2026-08-05 확인). 문서만 있던 유령 스크립트.**
  실제로 같은 일을 하는 것은 `run_ctorque_capture.sh` (bias0·모터지연OFF·5패턴·ROTOR_LOG).
  사용: `OUT=results_calib_m1326 EP=2 nohup ./run_ctorque_capture.sh > calib.log 2>&1 &`
- `online_rl_main --log-sysid` — GT속도+IMU+명령을 **IMU 레이트(250Hz)** 로 저장.
  ※ 50Hz 로 뜨면 자세루프 토크명령이 앨리어싱된다(초기 실패 원인).
- `run_sim` env `ROTOR_LOG=경로.npz` — PX4 명령 + 실제 적용 로터 각속도
- `run_sim` env `MOTOR_TAU=0.03` — 모터 1차 지연(초) 주입. **기본 비활성(0)**.
  Pegasus 는 명령을 즉시 로터속도로 적용("no delay introduced")하므로 실기 ESC+모터 지연이 없다.
  켜면 플랜트가 바뀌어 **밴드 재측정 필요** → 실기 τ 실측 후 켜는 것을 권장. 비행 검증은 미실시.
- `calibration/fit_static_from_rotor.py` (C_thrust/C_torque 확정) / `fit_gains.py` (병진·항력)
- `verify_calibration.py <outdir>` — 요 슬루·항력·호버점 회귀검증
- `validate_by_regime.py <outdir...>` — 호버/순항/급기동 영역별 고정계수 예측잔차 (+`--fit-coupling`)
- `run_regime_check.sh` — waypoint/circle/figure8 패턴별 검증 캡처
- `probe_mass.py` — Isaac 런타임 질량/관성 직접 쿼리 (`~/isaacsim/python.sh probe_mass.py`)
- ⚠ 시스템 python3 는 numpy2/scipy 불일치 → 분석 스크립트는 `~/isaacsim/python.sh` 로 실행

### 실기 도구·문서 (px4_field/) — 2026-08-07 정리
- **`RUN.txt`** — 실기 비행 절차 **F1~F8**. 현장에서 보는 유일한 문서.
  ⚠ 번호가 바뀌었다: 구 F2=doublet → **신 F3**, 구 F3=drag → **신 F4**.
  F2=오토튠, F5~F8=waypoint/circle/figure8/aggressive (보조 F3-2 yaw_spin, F4-2 accel_line)
- **`ISAAC_F1_F8.txt`** — Isaac SITL 수동조종으로 F1~F8 전부 리허설. **나가기 전 필수.**
  `launch_isaac_manual.sh <f1..f8>` 가 엔진 기동 + 시퀀스 실행을 한 줄로 한다.
- `f5_pattern.py --pattern waypoint|circle|figure8|aggressive|yaw_spin|accel_line`
  — 실기 패턴 비행. 궤적식은 `online_rl_main._compute_setpoint` 에서 그대로 이식.
- **★ ulog 회수·판정 자동화 (2026-08-07)** — 현장에서 사람이 할 일이 없다.
  비행 스크립트가 끝나면 `offboard_common.postflight()` 가 스스로:
  ① disarm 대기(로그는 disarm 때 닫힌다 — 실기 `SDLOG_MODE=0`)
  ② `fetch_ulog.py` 로 **젯슨이 이미 물고 있는 USB MAVLink 링크**를 통해 회수
     (QGC 를 안 붙인다. FC USB 포트가 하나뿐이고 오토튠 때문에 젯슨이 상시 점유.
      MAV_0_CONFIG=101(TELEM1) / UXRCE_DDS_CFG=102(TELEM2) 라 DDS 와 충돌 없음)
  ③ 파일명에 비행 이름을 박아 저장 → `ulog/f5_circle_log47_20260807_142233.ulg`
  ④ `check_ulog.py` 자동 실행 → **종료코드로 PASS/FAIL 판정**(0/1/2)
  · `--no-fetch` 로 끈다. F2 오토튠만 자동이 안 붙는다(같은 시리얼 포트).
  · `check_ulog --type pattern` 신설: 오프보드 구간 / **속도 클램프** / 추종 RMSE /
    EKF 리셋. 클램프 판정이 핵심 — 클램프되면 sim↔실기 궤적 비교가 깨진다.
- `autotune_go.py` (G=C_torque/I) / `fit_autotune.py` / `fit_from_ulog.py` / `check_ulog.py`
- `watch_ekf.py`(EKF 리셋 감시) / `watch_offboard.py`(명령 vs 실제 추종)
- 구 문서는 `px4_field/old/` 와 `old/` 로 옮겼다 — 각 폴더의 `README.txt` 가 무엇이
  무엇으로 대체됐는지 적어둔다. **구 문서는 구 번호(F1/F2/F3)라 현장에서 보면 위험하다.**

## 📦 [보존/무효] 핵심 물리 지식 — 구 플랜트 측정치 (2026-08-05 전면 무효화)
> ⚠️ **아래 5개는 전부 재측정 대상이다. 근거로 인용하지 말 것.** 삭제하지 않고 이력으로 남긴다.
> 무효 사유: ①질량 1.6186→1.372→1.326 ②C_torque 14배 오차 수정 ③오일러 ENU/NED 프레임 버그
> ④GPS update 멀티레이트 게이트(아래) — ②③④는 NIS 정의 자체를 바꾼다.
> 특히 **2번(펄스 시그니처)은 CUSUM 대비 RL 우위의 축**이라 재확인 전까지 서사에 쓸 수 없다.
1. s<1.2 공격은 PX4가 중화(EADR과 일치) → 공격 정의에서 배제
2. step-류 공격의 잔차 시그니처 = 펄스: t=1~2 스파이크 → t=3~8 침묵(PX4 보상) → 후기 재상승.
   CUSUM은 펄스 누적 불가(지연 7~15스텝), 윈도우 패턴매칭(RL)은 t=1~2 포착 가능 — RL 정당화 근거.
3. 호버-공격 평형은 흡인영역(basin)이 좁음: 처음부터 호버면 생존, 기동+온셋 노출 후 전환하면
   서서히 발산(전환 후 40~150스텝 뒤 crash_drift/altitude). 전환 과도 자체는 무해(고도캡처 수정 후).
4. combined 모드에서 죽음의 주 경로는 추력 채널(crash_altitude) — 호버로 흡수 불가한 채널.
5. ramp 0.1s(=1스텝@10Hz)는 사실상 step. ramp가 대응 데드라인(d≈3스텝)보다 길어야 구제 가능 가설.

## ★★ 재정합 로드맵 (2026-08-05 확정) — 이 순서를 어기면 두 번 한다 ★★
과거 측정치는 전부 무효(위 📦 블록 + FROZEN 블록). 아래 [1]→[4] 순서로만 진행한다.
```
[1] 플랜트 확정   ← sim 이 실기처럼 "날게"
    ✅ 질량 → **1.340 kg** (2026-08-06 저울 실측, 젯슨↔FC USB-C 케이블 포함)
       · 케이블 없이는 1.326. 케이블은 오토튠(MAVLink 트리거)에 필요하고,
         **네 팩 전부 케이블을 단 채로** 난다(형상 통일 — 아래 G/F2 비교 때문).
         질량 +1.2% / COM 이동 ~1mm / 관성 +0.13% → 자체 영향은 무시 수준.
         ⚠ 진짜 위험은 진동: 뻣뻣한 케이블이 FC 로 진동을 전달하면 자이로 σ 가 오른다.
       · 1342g→1340g 은 저울 오차(0.15%). **1.340 사용.**
       · ⚠ calibration.json 은 아직 1.326 — 아래 항목들과 **한 번에** 반영할 것.
       · ⚠⚠ **정정(2026-08-06): "I 는 잴 필요 없다"는 절반만 맞다.**
         ukf_filter.py:172-174 의 각속도 전파에는 항이 **둘**이고 I 가 다르게 들어간다:
             s[9] += ( (I[1]-I[2])/I[0]·q·r  +  u[1]/I[0] ) · dt
                       └ 자이로스코픽             └ 제어 토크
           · 제어항 u/I : C_torque/I = G 로 **비율만 필요** → 오토튠으로 확보 ✅
           · 자이로항   : **I 성분 간 비(ratio)** 가 필요. C_torque 와 무관해 G 로 상쇄 안 된다.
         크기(현 I 기준, roll 계수 −0.910):
             |q|=|r|=0.5 → 0.23 rad/s² (제어항의 1.3%)
             |q|=|r|=1.0 → 0.91 (5.1%)
             |q|=|r|=2.0 → 3.64 (20.4%)   ← 급기동에서 무시 못 함
         = **I 의 절대 크기는 G 에 흡수되지만, 성분 비는 별도로 필요하다.**
         ⚠ 그리고 sim C_torque 를 고정한 채 I 를 실기 G 에 맞추면 **비물리**가 된다:
             G_DC 기준  I=(0.0060,0.0086,0.0523) → Izz/(Ixx+Iyy)=3.58
             Σb/dt 기준 I=(0.0185,0.0260,0.1439) → 3.24     (쿼드는 ~1 이어야 함)
           현재 I 는 0.952 로 물리적이다. 즉 **실기 C_torque 도 sim 과 다르다**(당연 — Iris 아님).
           → G 만으로는 C_torque 와 I 를 분리할 수 없다. 셋 중 하나가 필요:
             (a) I 직접 측정(2선 진자 등)  (b) CAD/부품 질량분포  
             (c) 평면성 Izz≈Ixx+Iyy 가정 + Ixx/Iyy 비 가정 → C_torque 역산
           당장은 **I 성분 비는 현재 값(Iris 스케일, 평면성 0.952)을 유지**하고,
           절대 크기만 G 에 맞춰 스케일하는 것이 안전하다.
    ✅ C_thrust — 실기 호버평형 실측 (2026-08-06, F1 3소티)
         u_hover  0.3333 / 0.3521 / 0.3548   (시간순. SoC 저하로 단조 증가)
         secant C = mg/u_hover = 39.44 / 37.33 / 37.05 N per unit
       · 편차 6% 는 RUN.txt 가 상정한 예산 안(SoC 30%p → C_thrust 오차 4.4%).
         → 플랜트는 고정하고 **학습 시 UKF C_thrust 에만 ±5% 랜덤화**로 흡수(기존 결정 유지).
       · 08-04 값(u_hover 0.3320, C 39.18)과 첫 소티가 거의 일치 → 재현성 확인.
    ✅ ★★ 모터 1차 지연 τ ≈ 23 ms — **세 경로가 독립적으로 일치** (2026-08-06 확정)
         ① 08-05 수직 스윕 상호상관                    τ = 23 ms
         ② 08-06 오토튠 ARX 2번째 극 a2≈0.67 (log_43)  τ = 23.1/22.5/20.0 ms (roll/pitch/yaw)
         ③ 08-06 수직 스윕 호버구속 적합 최적지연       τ = 25 ms
       · ⚠ 현재 SITL 은 τ≈3 ms = **사실상 지연 없음**(Pegasus 가 명령을 즉시 로터속도로 적용).
         이것이 sim↔real 회전동역학 차이의 **주범**이다.
       · → `run_sim` env MOTOR_TAU  ⚠ **정정(08-11): 0.023 이 아니라 0.014.**
         MOTOR_TAU 는 유효지연을 2배로 만든다(up+down): τ_eff ≈ 4ms(커플링) + 2×MOTOR_TAU.
         실기 a2=0.67 ↔ sim MOTOR_TAU=0.014 (a2 0.663). 플랜트 변경이므로 밴드 재측정 필요.
    ✅ ★ G (= C_torque/I) 실기 실측 — 오토튠 (2026-08-06, log_43, COMPLETE 완주)
         수렴 maxVar roll 1.8 / pitch 1.5 (sim 11 보다 좋음), a1+a2+1≈0 (적분기 정합 통과)
                        G_DC        Σb/dt(지연극 제외)
         roll          595.5            192.9
         pitch         467.2            154.8
         yaw            93.9             34.1     ⚠ maxVar 53.6 미수렴
       · **G_DC 를 sim 목표로 삼되, 반드시 MOTOR_TAU 를 켠 상태에서** 맞춘다.
         (τ 를 끈 채 G_DC 를 맞추면 즉응이 3배 과대해진다)
       · "실기 G 가 sim 의 4.8배" 는 **착시**다. sim 에 지연이 없어 G_DC 가 안 부푼 탓.
         지연 극을 뺀 Σb/dt 로 보면 roll 1.34배 / pitch 1.10배 / yaw 1.27배 — 정상 범위.
       · 교차검증: 롤 여기구간 실측 |ω̇|std/|u|std = 183 ≈ Σb/dt 192.9 ✓
         (단순 회귀 corr 0.15 = 문서화된 폐루프 편향 R² 0.03~0.15 그대로 → 오토튠을 쓴 이유)
       · ⚠ G_DC 는 1/(1−a2)=3.09 배로 증폭되므로 **a2 오차에 민감**하다. 다음 측정과 대조할 것.
    ✅ 추력곡선 2차항 — 수직 스윕 호버구속 적합 (2026-08-06, log_40 118~150s)
         T = mg + 34.8·(u−u_h) + 101.5·(u−u_h)²     R²=0.775  RMS=0.86 N  (τ=25ms 보정 포함)
         호버 국소기울기 dT/du = 34.8  vs  secant C = 37.3   →  비 0.93
       · sim 은 T=4k(100+1000u)² 로 secant 25.6 / 국소 41.7 → 비 **1.63**.
         **실기는 호버 근방에서 훨씬 평평하다.** → [2] 의 Γ(u→T) 형태 판정에 직접 쓸 자료.
       · 구속 없는 순수 선형은 R²=0.105 로 실패 → 2차항 자체는 필요하다.
    ⬜ C_torque / I 개별값 — **분리 불가이고 분리할 필요도 없다**(위 G 항목 참조).
       sim 플랜트는 C_torque/I 비가 G_DC 와 같도록만 맞추면 된다.
       (참고로 남기는 구값: x 3.568→4.163 y 4.017→4.030 z 4.907→4.994, 로터경로)
    ⬜ drag ← F3 (2026-08-06 미실시 — 배터리 소진)
    ⬜ 자이로 σ ← F1 3소티 실측 [0.072,0.069,0.023] / [0.079,0.060,0.021] / [0.134,0.116,0.032]
       · 기존 기록 [0.0548, 0.0612, 0.0250] 대비 **x축이 30~45% 높다.**
       · ⚠ 단 산출 방법(이동평균 제거 51탭)이 기존과 같은지 미확인 → 방법 통일 후 재판정.
       · 3번째 소티만 2배 나쁨 — 케이블 진동 가설과 함께 다음 출동에서 A/B 확인.
       · **NIS 바닥을 직접 정하므로 계수보다 우선.**
    ✅ ★ 제어기 제한 9개 — 실기 PX4_6.params → Isaac 주입 (2026-08-07, `run_sim.py:196`)
         MPC_XY_VEL_MAX 2.0(기본 12.0) / MPC_XY_CRUISE 2.0(5.0) / MPC_VEL_MANUAL 2.0(10.0)
         MPC_XY_VEL_ALL 2.0 / MPC_Z_VEL_ALL 3.0 / MPC_Z_VEL_MAX_DN 2.25 / MPC_Z_V_AUTO_DN 2.25
         MPC_TKO_SPEED 1.8 / MPC_LAND_SPEED 1.5.  기동 시 `[PX4LIM]` 로그로 확인.
       · ⚠⚠ **플랜트가 아니라 제어기를 바꾼다 — 그래도 폐루프가 바뀌므로 결과는 같다.**
         밴드·NIS 기준선·d′ 는 [3] 에서 **이 상태로** 재측정한다. 이전 값과 비교 금지.
       · 수평 최대속도가 6배 달랐다. 수동조종 감각(MPC_VEL_MANUAL)까지 이제 실기와 같다.
    ✅ ★ 전 패턴 속도·궤적 정합 (2026-08-07, `swrl_config.py` + `px4_field/f5_pattern.py`)
         **방향: 실기에서 안전하게 날 수 있는 값을 먼저 정하고 sim 이 따른다.**
         flight_radius 5.0 → 3.5 → ⚠ **2.8 (08-12 재하향)**
           3.5(접선 1.75)도 MPC_XY_VEL_MAX 2.0 대비 여유가 14% 뿐이라 **명령속도가 2.00 에
           상시 포화**했다(08-12 실기 실측). 기체가 뒤처져도 못 따라잡아 추종 RMSE 1.4~2.9 m.
           2.8 → 접선 1.40, 여유 30%.
         flight_omega 0.5 유지,  agg_radius 4.0 → 2.8 → ⚠ **2.24 (08-12, 같은 배율 0.8)**
         fig8_radius_scale = **1/√2** ← figure8 은 vx,vy 가 t=0 에서 동시 최대라
             |v|max = R·ω·√2 = 2.47 로 클램프됐다. 반경만 1/√2 배해 circle 과 맞춤.
             ★ sim·실기 **양쪽 동일 처리**. 한쪽만 고치면 궤적 비교가 깨진다.
         → 전 패턴 ≤ 1.76 m/s < 2.0 (여유 12%). 공간은 circle 7×7 m 가 최대.
[2] 관측·필터 확정
    ★★ **센서 노이즈 σ 와 필터 Q·R 은 서로 다른 축이다 — 절대 뭉개지 말 것.**
       · **실기 센서 σ(gyro 0.045~0.13, GPS vel [0.11,0.09,0.17~0.33]) → sim 에 "먹인다"**
         = 측정 현실성 = **플랜트([1]) 소속**. sim 측정이 실기보다 깨끗하면 NIS 바닥이
         비현실적으로 낮아져 실기에서 상시 오탐. R 과 무관하게 반드시 주입한다.
       · **UKF 의 Q·R 은 센서 통계에 맞추는 값이 아니다** = **탐지 설계값(고집)**.
         이건 shadow UKF 다. 공격받은 센서값을 필터가 흡수해버리면 잔차→0 = 탐지 불가.
         그래서 R↑·Q↓ 로 **정상 제어명령 기반 예측**을 더 믿게 만들어 δ 가 잔차로 드러나게 한다.
         현행: R_gyro 0.5 (실기 σ² 0.002~0.017 의 **30~250배**) / R_vel 0.3 / Q 5e-4 / ff=1.0.
         ∴ NIS 는 χ² 통계량이 아니라 탐지 feature (위 "관측" 절과 같은 얘기).
    ✅ GPS update 멀티레이트 게이트 (아래 절)
    ⬜ Γ(u→T,τ) 형태: 1차 시컨트 / 운용 LS / 어파인(C·u+T₀) / 2차 — **d′ 기준**으로 판정
    ⬜ Q·R 재튜닝  ← R 의 실효 의미가 바뀌므로 [2] 안에서 마지막. 이전 R 은 5× 과대가중 전제.
       **튜닝 목표(양방향) — 센서 정합이 아니라 잔차의 시간 거동이다:**
         ① 공격 지속 중에는 **잔차가 계속 남아야** 한다 (필터가 공격을 흡수·추종하면 안 됨)
         ② 공격이 끝나면 **잔차가 빠르게 감소해야** 한다 (복귀 판정 = 양방향 stopping 의 전제)
       ①만 세면 복귀가 안 되고, ②만 세면 공격이 흡수된다. 두 시상수를 함께 본다.
       판정 기준은 consistency/χ² 가 아니라 **d′ + 온셋/오프셋 응답시간**.
[3] 재측정 (전부 여기서. [2] 완료 전 착수 금지)
    평시 NIS 기준선(실기 f1_field.ulg 와 대조) / ★펄스 시그니처 / d′·4클래스 분리도 /
    결과성 밴드 ①②③ + dhover 데드라인 / CUSUM baseline(FPR 캘리브)
    ⚠ 재측정 사유에 **제어기 제한 9개 + 패턴 속도 재정합**(위 [1] 참조)도 포함된다.
      플랜트 변경만이 아니다 — 제어기가 바뀌어도 밴드는 다시 재야 한다.
    ⚠ 기동 중 NIS 기준선은 **실기 F5~F8 로그와 대조**한다(`px4_field/RUN.txt`).
      지금 있는 실기 기준선은 호버뿐인데, 오탐은 기동 중에 난다.
[4] 학습  펄스 기반 reward 재설계 → RHUKF-FV vs Adam
```

### GPS update 멀티레이트 게이트 (2026-08-05)
**구 구현**: `_tick` 50Hz → `_run_ukf_step()` 이 매번 9D 전체 update. 그런데 GPS(`/sim/sensor_gps`)는
10Hz(`run_sim.py:665`)라 **같은 측정을 5회 재사용(ZOH)** → GPS 채널 실효 R 이 약 1/5 로 축소.
= 필터를 GPS 에 더 순종적으로 만드는 것 = **"고집(R↑·Q↓)" 설계 의도와 정반대**였다.
(git 전수 확인: 최초 커밋부터 이 구조. 멀티레이트로 구현된 이력 없음.)

**신 구현**: `DynamicsUKF.step(z, u, gps_fresh=True)` + `stale_gps_inflate=1e6`.
GPS 가 낡은 스텝은 `R[0:6,0:6] *= 1e6` → 그 채널 이득 ≈0 (update 스킵과 수학적 동치).
코드 경로를 하나로 유지하려는 선택(측정 분리 방식 대비 검증이 쉽다).
```
predict + gyro update : 50Hz (루프 레이트)
GPS update            : 10Hz (센서 실제 레이트)
RL 관측               : 10Hz (gps_updated 게이트, 기존과 동일)
```
검증: stale 스텝에서 Δpos 1.4e-4·Δvel 은 **인플레 1e4~1e10 에서 불변**(=GPS 기여 0, 잔량은 predict).
gyro 보정은 정상 유지. 기대 효과: vel 채널 고집 회복 → **vel NIS 상승 → 약했던 vel d′(2.2) 개선**.

## 프레임워크 성립 3조건 (동시 필요)
① 결과성: 무대응(track) 시 추락  ② 탐지 가능성: NIS 온셋 엣지 존재
③ 대응 가능성: 탐지지연 d≤3스텝 전환으로 생존 (dhover 생존율 ≈ hover 셀)
현재 목표: 3조건이 겹치는 (공격채널, ramp, bias 범위) 확정 = "공격 세팅 동결"

## 스윕 인프라 (이미 구현됨)
- `--sweep`: 고정정책 순회. 정책: track(A) / hover(B, 처음부터 원점호버) /
  dhover{d}(C, 공격+d스텝에 현재위치·고도 캡처 호버 전환) — d ∈ sweep_hover_delays=(1,2,3,4,5,8)
- CLI: `--sweep-mode {combined,torque,thrust}` `--sweep-values 1.3,1.5,...`(쉼표구분!)
  `--ramp 0.3` `--speed N` `--outdir DIR`
- 집계: `python3 sweep_aggregate.py <dir>` → (1)생존율/밴드 (1b)dhover 데드라인 (2)정상NIS
  (3)분리도d' (4)CUSUM baseline
- 완료된 스윕: results_pilot3(combined, ramp0.1, 밴드[1.33,1.42]이나 dhover 전멸 = 조건③ 실패)
- E1=results_pilot4(combined, ramp0.3) / E2=results_torque_r0{0,1,3}_fine(torque, ramp0/0.1/0.3) — 판독완료(아래)

## 📦 [전면 폐기] 과거 밴드·공격세팅 측정치 — 방법론만 남긴다 (2026-08-07)

> ⚠️ **아래 블록의 모든 수치는 폐기 대상이다. 근거로 인용하지 말 것.**
> 밴드 [1.34,1.40] Nm, 생존율, dhover 데드라인, d′, ft_ratio 1.5 — **전부 버린다.**
>
> **폐기 사유** (플랜트가 통째로 바뀌었다):
> ```
> 질량      1.6186 → 1.372 → 1.326 → 1.340 (실측)
> G         sim 124.9 vs 실기 595.5 — 회전 권한이 5배 달랐다
> MOTOR_TAU 0 → 0.023 (실기 τ 23ms, 세 경로 확인)
> 관성 I    Iris 질량비 → L² 보정 (대각 511mm vs 250mm)
> 추력곡선   THR_MDL_FAC 0 → 0.9 검토 중 (곡률 1.68 → 0.92, 실기 0.93)
> 오일러 프레임 / GPS 멀티레이트 / GPS 8→10Hz
> ```
> 밴드는 **제어권한 대비 비율**로 정의되는데, 그 제어권한 자체가 5배 달라졌다.
> N·m 절대값은 물론이고 비율 표현도 재측정해야 한다.

### ★ 살려서 쓸 것 — 방법론 (수치가 아니라 절차)

```
① 프레임워크 성립 3조건 — 이 정의는 유효하다
     결과성   : 무대응(track) 시 추락
     탐지가능성: NIS 온셋 엣지 존재
     대응가능성: 탐지지연 d≤3스텝 전환으로 생존 (dhover 생존율 ≈ hover 셀)
   세 조건이 겹치는 (공격채널, ramp, bias 범위) 를 찾는 것이 "밴드".

② 스윕 프로토콜
     정책 3종 순회: track(A) / hover(B, 처음부터 원점호버)
                    dhover{d}(C, 공격+d스텝에 현재위치·고도 캡처 호버 전환)
     d ∈ (1,2,3,4,5,8),  셀당 20+ 에피소드(지형파악은 4~8 로 충분)
     집계: sweep_aggregate.py → (1)생존율/밴드 (1b)dhover 데드라인
                                 (2)정상NIS (3)분리도d′ (4)CUSUM baseline

③ 판정 순서 — 결정트리
     torque-only 가 낮은 ramp 에서 열리면 torque 우선,
     안 열리면 combined 로 ft_ratio 를 올려가며 탐색.
     밴드 = track 추락 ∧ hover 생존 인 구간.
     ★ 그 안에서 dhover d≤3 생존율이 hover 셀과 같아야 조건③ 통과.

④ 패턴별로 따로 볼 것 (2026-07-23 교훈)
     aggressive 가 최악(가장 취약) 케이스. figure8/hover 는 같은 bias 를 흡수해
     비결과성이 된다. 밴드를 전 패턴 균일 주입하면 ①이 깨진다.
     → 패턴별 (onset, 상한, 천장) 3튜플로 기록할 것.

⑤ 데이터 무결성 습관
     · --sweep-values 는 쉼표 구분 (공백이면 뒤 값이 조용히 버려진다 — 실제 사고 2회)
     · 시작 30초 내 로그에서 ramp/bias목록/셀수 확인
     · summary CSV 의 survived ↔ crash_reason 일치 전수 검사
     · 결과 폴더는 실험별로 분리

⑥ 그림
     plot_fine_staircase.py → deadline_staircase.png (한글폰트 Noto CJK 적용됨)
```

### 이하 폐기된 수치 (이력 보존용, 인용 금지)


#### [폐기] E1/E2 판독 결과 (2026-07-06)
- **E2 torque 승리 → 공격채널=torque 확정.** 결정트리("torque가 낮은 ramp에서 열리면 torque 우선")대로.
- E2(torque fine, ramp 0/0.1/0.3 공통): 결과성 밴드=단일점 **1.30Nm** (1.25 둘다생존 / 1.35↑ hover도붕괴).
  1.30에서 track 0.12~0.38, hover 1.0, **dhover d=1~8 전부 1.0 → 조건③ 최초 통과.** ramp 무관(ramp0도 열림 = abrupt 서사 유지).
- E1(combined, ramp0.3): 밴드는 넓음[1.33,1.44]·hover=1.0지만 **dhover 붕괴(1.33서 d2=0.5/d3=0.38, 1.36↑ 전멸) → 조건③ 실패.**
  사인=crash_altitude(추력채널, 호버로 흡수불가·물리#4). pilot3 실패 재현.
- **⚠ combined 기각은 철회됨 (2026-07-07)** — 아래 "최종 근거"는 vel 채널 미확인 + ramp0.3 상태의 판단이었음.
  후속 검증(results_combined_final, ft1.5·ramp0.0)에서 **밴드 [1.34,1.40] 전 구간 dhover d≤3=1.0(조건③ 통과)** 확인 → combined 채택.
  E1의 dhover 붕괴는 ramp0.3(추력 서서히 주입 → 지연전환 시 이미 basin 이탈) 및 ft_ratio 과대(당시 ft20~35)의 산물로 재해석됨.
- ~~★ combined 기각 최종 근거 (2026-07-07): "탐지 신호 ∩ 대응 유지" = 공집합.~~ **(철회)**
  ~~combined ft20~35 구간에서 vel/gyr NIS는 0.2~1.0으로 유의하게 활성화(=탐지가능성 ②는 열림)되지만,~~
  ~~동일 구간에서 dhover3 생존율이 0으로 붕괴한다.~~
  → ft1.5(T≈2N)·ramp0.0에서는 이 교집합이 **열린다**(밴드 내 d≤3=1.0 ∧ vel d′=2.2). 위 FROZEN 갱신 블록 참조.
- crash lag: 밴드내(1.30) dhover는 전부 timeout=실제 생존(summary survived=1). 실사망은 1.35(밴드밖·hover도0)
  에서만 crash_flip/drift, 전환후 lag 13~50스텝 = 지연 basin 발산(물리#3). **전환 과도 자체는 무해 재확인.**
  ※ 데이터 무결성 확인(2026-07-06): summary CSV 1600행 전수 검사 survived↔crash_reason 불일치 0.
    _end_sweep_episode/sweep_aggregate 정상. (어제 crash-lag 임시스크립트가 detail의 terminal timeout을
    death로 오집계한 것이 유일한 아티팩트 — 데이터/집계 파이프라인 버그 아님.)
- 통합 그림: `plot_fine_staircase.py` → deadline_staircase.png (한글폰트 Noto CJK 적용됨).

#### [폐기] 공격 세팅 동결 (2026-07-07) — 더 이상 FROZEN 아님
> ⚠️ **2026-07-28 무효화**: 아래 밴드/생존율/NIS 수치는 전부 (질량 1.6186kg + C_torque 14배 오차 +
> 오일러 프레임 버그) 상태에서 측정된 것이다. 플랜트가 15% 가벼워졌고(T/W 2.61→3.07) 호버 전환의
> 90° 요 슬루가 사라졌으므로 **밴드는 반드시 재측정**해야 한다. 아래는 이력으로만 읽을 것.
**채널 = COMBINED ft_ratio=1.5 (torque:thrust = 1:1.5, T≈2N@s=1.34), 밴드 = [1.34, 1.40] Nm, ramp 0.0.**
압축 = ~~채널분리: vel=log(x+0.5) / gyro=log1p~~ **→ 통일(2026-07-22): 전 채널 offset=1.0, ε̃=ln(1+ε)/(1+ln(1+ε))** (env/ukf_filter.py `compute_nis_scaled`). ⚠️ 압축기 변경으로 기존 관측 baseline·d′ 무효화 — 재캡처(우선순위1 deadline)로 재산출 중.
검증 스윕 = results_combined_final (combined ft1.5, ramp0.0, bias 1.34~1.42, 27 cells × 20ep = 540ep):
```
  bias | track hover | dh1  dh2  dh3 | 판정
 1.340 |  0.05  1.00 | 1.00 1.00 1.00 | ★밴드 (①②③ 성립, crash=altitude)
 1.360 |  0.00  1.00 | 1.00 1.00 1.00 | ★밴드
 1.380 |  0.00  1.00 | 1.00 1.00 1.00 | ★밴드
 1.400 |  0.00  1.00 | 1.00 1.00 1.00 | ★밴드
 1.420 |  0.00  0.00 | 0.95 0.90 1.00 | hover도 붕괴(밴드밖 상, 회복불가) → ≥1.42 제외
```
- **combined 채택 근거**: vel d′ 1.5→2.2 (thrust 소량 + log0.5 압축강화로 vel 채널 활성화), gyro d′ 3.0→2.5 (소폭 하락하나 건재).
  순이득 vel(+0.7) > gyro 손실(−0.5). combined 기각(과거)은 **철회됨** — 당시 판단은 vel 채널 미확인 상태의 결론이었음.
- **gyro "약화"는 아티팩트로 판명(2026-07-07)**: 겉보기 gyr d′ 480→2~3은 ①vel컬럼 오독(2~3은 vel), ②집계공식 raw→log1p 변경,
  ③ mean의 에피소드길이 희석의 합. 동일 기준 재계산 시 combined gyr d′(log1p)=12~17 vs torque 18~22, **s=1.34 겹침점 95pct gyr NIS는 동등(353 vs 376)** → 신호 저하 아님.

##### [폐기] 직전 torque-only 동결 (2026-07-06)
최종 스윕 = results_torque_final (torque, ramp0.0, bias 1.26~1.34, 42 cells × **20ep** = 840ep). 판독:
```
  bias | track hover | dh1  dh2  dh3 | 판정
 1.260 |  1.00  1.00 | 1.00 1.00 1.00 | 둘다생존(밴드밖 하)
 1.280 |  0.90  1.00 | 1.00 1.00 1.00 | 결과성 onset(track 약간 붕괴)
 1.300 |  0.45  1.00 | 1.00 1.00 1.00 | ★밴드 (①②③ 성립)
 1.320 |  0.25  1.00 | 1.00 1.00 1.00 | ★밴드 (dh4=0.85 dh5=0.70이나 d≤3=1.0)
 1.340 |  0.00  0.15 | 0.15 0.35 0.00 | hover도 붕괴(밴드밖 상, 회복불가)
```
- **결과성 밴드 = [1.300, 1.320] Nm** (track추락 ∧ hover생존), 중심 1.31.
- **3조건 전부 성립**: ①결과성(track 0.45→0.25) ②탐지가능(gyr d′≈480~500) ③대응가능(dhover d≤3=1.0=hover).
- 탐지 데드라인: 밴드 내 d≤3 생존율 = hover 셀(1.0)과 동일 → step 공격 채택 OK. 1.34는 d=1부터 붕괴(회복불가, 밴드밖).
- (SUPERSEDED) torque-only 동결값: bias_scale_range=[1.30,1.32], thrust=0. → combined ft1.5 밴드[1.34,1.40]로 대체됨.
- 그림: `plot_fine_staircase.py` → deadline_staircase.png 에 FINAL 20ep 패널(밴드 음영) 추가됨.

##### [폐기] config 정렬 (2026-07-07)
- swrl_config.py 를 combined ft1.5로 정렬: `bias_ft_ratio=0.0 → 1.5`(thrust=1.5·s 복원), `bias_scale_range=(1.28,1.32) → (1.34,1.40)`(combined 핵심밴드).
  `sample_bias_box=True`(tube 샘플러) 유지 → 이제 (torque_xy=s, torque_z=0.2·s, thrust=1.5·s) = 동결 combined 채널. `bias_yaw_ratio=0.2` 유지.
  샘플러 단위검증 완료: s∈[1.34,1.40] → thrust≈2N, th/tq_xy≈1.5(±jitter).
- **★ 압축 채널분리는 학습 관측에도 이미 반영됨(코드검증 완료)**: `env/ukf_filter.py:compute_nis_scaled(offset)` 이 vel=0.5/gyro=1.0로 호출됨.
  - 학습 경로 `_rl_step_10hz` (online_rl_main.py:824-825) 와 sweep 경로 `_sweep_step_10hz` (1176-1177) **동일 함수·동일 offset** → window_buffer→agent obs.
  - 데이터검증: combined_final detail의 nis_v_scaled = log(raw+0.5)/(·) 일치(Δ<1e-4), log1p면 Δ=2.23(불일치) → vel은 확실히 log0.5.
- **⚠ 관측 차원은 절대 건드리지 않음**: `[nis_vel, nis_gyro, action]` × window4 = dimS 12 구조 유지. 압축 함수(offset)만 채널분리.
  config 정렬은 **공격 샘플러만** 바꾼다(bias_ft_ratio/scale_range) — obs_scale·dimS·window_size 불변.

#### [폐기 수치 / 교훈은 위 ④] 전 패턴 공격 정합화 (2026-07-23)
- **버그 수정**: `attack_flight_patterns=['aggressive']` → **flight_patterns 전체(waypoint/circle/figure8/aggressive)**.
  실제 공격은 **모든 기동 궤적**에 랜덤 세기로 주입되는 것이 의도된 설계였음(사용자 확정). 기존 aggressive 강제는 버그.
  이제 공격/평시 에피소드 **동일 패턴 분포** (swrl_config.py:117, sampler:388; 검증=공격에피 4패턴 균등 ~830/4000ep).
- **⚠ 동결 밴드 [1.34,1.40]은 aggressive 전용 — 전 패턴 미성립(2026-07-23 results_deadline 분석):**
  ```
   패턴      track생존(①=낮아야)   dhover0(③=높아야)    판정
   figure8   1.00(전 bias)         1.00              ① 실패: 공격 무해(track 안죽음)
   hover     1.00(전 bias)         1.40서 0.00       ① 실패 + hover전환이 오히려 사망(soft-hold 이슈)
   waypoint  1.37서 0.25           1.37서 1.00       1.37만 부분 성립(dh3=0.17로 데드라인 빡빡)
  ```
  → **aggressive = 최악(가장 취약) 케이스**. 순한 패턴은 제어여유 ↑ → 같은 bias 흡수 → 밴드가 **위로 이동(더 센 공격 필요)** 하거나 아예 무해.
  전 패턴 공격 정합화 시 **[1.34,1.40] 균일 주입은 figure8/hover에선 비결과성 공격**(①붕괴) 문제. → 밴드 패턴별 재조정 vs 비결과성 수용, 결정 필요.

## [폐기·대체됨] 동결 후 순서 — 재정합 로드맵 [1]~[4] 로 대체됨
E3: 밴드 3점 × {waypoint, circle, figure8, aggressive} 전 패턴 스팟체크(전 패턴 공격 정합화로 승격 — 밴드 유지/이동 확인)
E4: bias=0 × wind 스팟체크(교란원 구성: aggressive 유지 vs 바람 추가 판단)
관측 점검: nis_separability.py 4-클래스(호버전환/급기동/공격/정상) 확장 → Bayes 상한 → 관측 동결
학습 수정(동결 후에만): gamma 0.8→0.9, n-step=3(memory.py), timeout 부트스트랩 분리
  (done_env=추락만 True), fn 초반가중(온셋 3~5스텝 -0.8→-1.2~1.5), terminal -10 유지
본실험: RHUKF-FV vs Adam vs CUSUM(FPR 캘리브레이션), 공격밀도(burst 수) 축 학습곡선

## 프레이밍 결정사항 (논문 서사, 바꾸지 말 것)
- 탐지 프레이밍 유지(위협평가 프레이밍 기각): 공격분포를 결과성 밴드로 한정해 "탐지=위협"이 설계상 참
- 호버 = 공격불가지론적 fallback + 검증기동(능동센싱). "recovery"라 부르지 않음
- 이진 행동 = stopping-time POMDP 표준. 복귀 학습 = 고전 QCD 대비 확장(양방향 stopping)
- ramp = 실험 완화가 아니라 "탐지-대응이 유의미한 공격 클래스"의 정의(실측 근거로 설정)
- gamma 낮춤으로 즉각성 확보하지 않음(즉각성은 조밀한 per-step 페널티 담당, gamma는 시야)
- 급기동/외란 = aliasing의 원천 = RL 존재 이유. 최종 평가에서 제거 금지(ablation 축으로만)

## 주의사항
- --sweep-values는 쉼표 구분(공백이면 parse_known_args가 조용히 뒤 값 버림 — 실제 사고 2회)
- 스윕 시작 30초 내 로그에서 ramp/bias목록/셀수 확인 습관
- 결과 폴더 실험별 분리. 에피소드 4는 지형파악용, 논문수치는 셀당 20+
- 모든 학습 설정 변경은 RHUKF-FV/Adam 동일 적용
