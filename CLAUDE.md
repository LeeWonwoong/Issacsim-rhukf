# CLAUDE.md — 프로젝트 컨텍스트 (sweep 실험 단계)

## 프로젝트 개요
UAV(쿼드로터) 제어입력(액추에이터) 공격 탐지. POMDP/QCD stopping 문제로 정식화,
UKF 잔차(NIS) 관측 기반 DDQN이 매 스텝 track/hover 이진 결정.
최종 기여: Adam 대비 커스텀 2차 최적화기 RHUKF-FV(FIR 철학)의 sparse-attack 샘플효율 우위.
센서 공격은 스코프 밖(센서 무결성 가정). 가장 가까운 선행: EADR(고정 CUSUM, 시나리오별 하드코딩 임계값
— 저자들이 future work으로 RL 지목), QUADFormer(센서 공격+transformer, TTD 미보고).

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
   - 이제 body 1.253385kg 를 써서 **총합이 1.372kg**(사용자 실기체 AUW). 관성도 같은 원칙으로
     구 총관성×(1.372/1.6186) → 총 (0.029547, 0.026484, 0.053359). 로터 x/y 암 길이가 달라 Ixx≠Iyy.
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

4. **추력-토크 기하 결합 항 — 발견했으나 의도적으로 미포함 (sim2real 결정, 2026-07-28)**
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
- `run_calib_mass.sh` — bias0·wind none·aggressive 캡처 (zu_log + sysid_log + rotor_log)
- `online_rl_main --log-sysid` — GT속도+IMU+명령을 **IMU 레이트(250Hz)** 로 저장.
  ※ 50Hz 로 뜨면 자세루프 토크명령이 앨리어싱된다(초기 실패 원인).
- `run_sim` env `ROTOR_LOG=경로.npz` — PX4 명령 + 실제 적용 로터 각속도
- `run_sim` env `MOTOR_TAU=0.03` — 모터 1차 지연(초) 주입. **기본 비활성(0)**.
  Pegasus 는 명령을 즉시 로터속도로 적용("no delay introduced")하므로 실기 ESC+모터 지연이 없다.
  켜면 플랜트가 바뀌어 **밴드 재측정 필요** → 실기 τ 실측 후 켜는 것을 권장. 비행 검증은 미실시.
- `SIM2REAL_DATA_SPEC.md` — 실기 데이터 수집 스펙(궤적/ulog 메시지/ROS2 토픽/ESC 텔레메트리)
- `calibration/fit_static_from_rotor.py` (C_thrust/C_torque 확정) / `fit_gains.py` (병진·항력)
- `verify_calibration.py <outdir>` — 요 슬루·항력·호버점 회귀검증
- `validate_by_regime.py <outdir...>` — 호버/순항/급기동 영역별 고정계수 예측잔차 (+`--fit-coupling`)
- `run_regime_check.sh` — waypoint/circle/figure8 패턴별 검증 캡처
- `probe_mass.py` — Isaac 런타임 질량/관성 직접 쿼리 (`~/isaacsim/python.sh probe_mass.py`)
- ⚠ 시스템 python3 는 numpy2/scipy 불일치 → 분석 스크립트는 `~/isaacsim/python.sh` 로 실행

## 핵심 물리 지식 (스윕으로 실측 확정된 것)
1. s<1.2 공격은 PX4가 중화(EADR과 일치) → 공격 정의에서 배제
2. step-류 공격의 잔차 시그니처 = 펄스: t=1~2 스파이크 → t=3~8 침묵(PX4 보상) → 후기 재상승.
   CUSUM은 펄스 누적 불가(지연 7~15스텝), 윈도우 패턴매칭(RL)은 t=1~2 포착 가능 — RL 정당화 근거.
3. 호버-공격 평형은 흡인영역(basin)이 좁음: 처음부터 호버면 생존, 기동+온셋 노출 후 전환하면
   서서히 발산(전환 후 40~150스텝 뒤 crash_drift/altitude). 전환 과도 자체는 무해(고도캡처 수정 후).
4. combined 모드에서 죽음의 주 경로는 추력 채널(crash_altitude) — 호버로 흡수 불가한 채널.
5. ramp 0.1s(=1스텝@10Hz)는 사실상 step. ramp가 대응 데드라인(d≈3스텝)보다 길어야 구제 가능 가설.

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

## E1/E2 판독 결과 (2026-07-06, 완료)
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

## ★ 공격 세팅 최종 동결 (2026-07-07 갱신, FROZEN) ★
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

### 참고: 직전 torque-only 동결 (2026-07-06, SUPERSEDED by combined ft1.5)
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

### config 정렬 (combined ft1.5 동결 반영, 2026-07-07 완료)
- swrl_config.py 를 combined ft1.5로 정렬: `bias_ft_ratio=0.0 → 1.5`(thrust=1.5·s 복원), `bias_scale_range=(1.28,1.32) → (1.34,1.40)`(combined 핵심밴드).
  `sample_bias_box=True`(tube 샘플러) 유지 → 이제 (torque_xy=s, torque_z=0.2·s, thrust=1.5·s) = 동결 combined 채널. `bias_yaw_ratio=0.2` 유지.
  샘플러 단위검증 완료: s∈[1.34,1.40] → thrust≈2N, th/tq_xy≈1.5(±jitter).
- **★ 압축 채널분리는 학습 관측에도 이미 반영됨(코드검증 완료)**: `env/ukf_filter.py:compute_nis_scaled(offset)` 이 vel=0.5/gyro=1.0로 호출됨.
  - 학습 경로 `_rl_step_10hz` (online_rl_main.py:824-825) 와 sweep 경로 `_sweep_step_10hz` (1176-1177) **동일 함수·동일 offset** → window_buffer→agent obs.
  - 데이터검증: combined_final detail의 nis_v_scaled = log(raw+0.5)/(·) 일치(Δ<1e-4), log1p면 Δ=2.23(불일치) → vel은 확실히 log0.5.
- **⚠ 관측 차원은 절대 건드리지 않음**: `[nis_vel, nis_gyro, action]` × window4 = dimS 12 구조 유지. 압축 함수(offset)만 채널분리.
  config 정렬은 **공격 샘플러만** 바꾼다(bias_ft_ratio/scale_range) — obs_scale·dimS·window_size 불변.

### ★ 전 패턴 공격 정합화 (2026-07-23) ★
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

## 동결 후 순서 (합의된 로드맵)
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
