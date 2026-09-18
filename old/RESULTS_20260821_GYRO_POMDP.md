# 결과 종합 — gyro aliasing 규명 · 가감속/바람 · Q/R · POMDP (2026-08-21)

> 이 세션 전체 실험 기록. 단일 출처. 물리값·정합은 `SIM_ALIGNMENT.md` 참조.

## 0. 출발 질문
"급기동/circle/figure8에서 gyro 추정오차(NIS)가 스파이크로 나타나야 할 텐데 왜 안 나오나?
다이나믹스·캘리브가 잘못됐나?" → 정밀 규명 → 프레임워크(POMDP) 성립 조건 도출.

## 1. gyro가 기동에 안 튀는 이유 (정밀진단)
**다이나믹스·캘리브 정상.** gyro 예측(`ukf_filter._f:176`)이 명령토크 u/I를 씀 → 플랜트도 같은
명령토크를 로터로 실행 + I·C 동일(calibration.json) → **UKF는 "예측"이 아니라 같은 입력을 같은
모델로 한 스텝 굴리는 것** → 맞을 수밖에. 고집(R↑)은 예측이 이미 맞으니 공짜.

검증(문제없음): 다이나믹스 ω 반응 정상 / UKF I=플랜트 I(103행 vs 440·467행) / C_thrust40.3·
C_torque 3.26,3.1,1.378·mass1.34 / MOTOR_TAU 0.014=실기23ms(2배적용) / THR_MDL 0.78.

**핵심 물리**: gyro 잔차 ∝ 명령토크(각가속). **등속회전=정상상태 토크0=완벽예측=깨끗.
가감속=토크≠0=모터지연 물려 잔차.**

## 2. 가감속 vs 등속 — 전 패턴 (핵심 실험)
속도변조 노브 구현(`online_rl_main._compute_setpoint`, env `SPEED_MOD_AMP`/`FREQ`, 기본0=현재동일).
2모드 × 4패턴 × 바람(0/9/12) benign 캡처.

**ws0(순수기동) gyro NIS 피크(p90) 등속 vs 가감속 배율:**
```
circle     1.0   ← 매끄러운 등속선회 = 완전 깨끗
figure8    1.0   ← 마찬가지
aggressive 1.4   ← 반원전환 약간
waypoint   5.0   ← ★코너 정지·급선회서 5배 스파이크
```
→ **gyro aliasing ∝ 기동 sharpness. 회전량 아니라 회전의 급변(가감속·급선회).**
매끄러운 circle/figure8은 속도변조 넣어도 약함(1.2~1.3). waypoint 코너만 강함.

**바람 효과(ws0→ws12 피크):** gyro 0.2→1.0(버펫팅), **vel 0.4→11(27배, 바람 주채널)**.

## 3. Q/R로 gyro aliasing 못 늘림 (offline Q스윕 증거)
```
Q_gyro    평시등속  평시기동  기동/등속비  공격 d′
5e-3(현행)  1.97    107.1    54.3 ★최고   1.56 ★최고
1e-3       11.0    430.7    39.0        1.33
1e-4      145.3   3497.4    24.1        0.86
```
Q↓ → 바닥 74배 폭증 + **분리도 악화(54→24)** + **공격 d′ 하락(1.56→0.86)**.
R↓는 고집약화로 탐지 추가악화. **→ 현재 Q=5e-3이 최적. 건드리지 말 것.**
잔차는 필터게인으로 못 키움 — 모델오차(기동 급변·바람)로만 커짐.

## 4. 기각된 레버 (전부 안 함)
속도 1.5배(등속이라 무효+밴드재측정) / 모터지연↑(이미 정합) / I·C불일치(모델오차 주입 안함) /
Euler채널(측정불가, z=pos/vel/gyro만) / cal-error 5%(무력) / Q·R↓(악화).

## 5. 프레임워크 그림 (자연 노이즈, 주입 없음)
```
             평시(맑음·매끄러움)   현실(바람+날카로운기동)   공격(강)
gyro p90        0.2 (깨끗)          ~1.0                  100+
vel  p90        0.4                 ~10                   (표류로 실림)
```

## 6. POMDP 분석 (진행 중)
POMDP = 공격 NIS와 평시 NIS가 **겹쳐야**(관측 애매) 성립. 안 겹치면 임계값(MDP).
- **공격이 너무 세면(100+) trivial** = POMDP 아님. per-step d′만 봐야.
- zu_noisy(구aggr): 평시기동107 ≈ 공격144, **per-step d′ 1.56(겹침)** vs windowed d′ 16.8(분리)
  → 윈도우가 분리 = RL 명분. 단 구aggr 불연속 아티팩트로 부푼 값.
- **핵심 긴장**: consequential(추락)⟹PX4권한초과⟹강함⟹탐지쉬움. 약하면 PX4보상→무해.
- **해법**: 약한 공격을 **바람+기동 중**(마진 얇음)에 주입 → 약해도 결과성 + gyro가
  기동/바람 노이즈에 묻힘=POMDP + 고집UKF는 잔차 남김(탐지가능). **노이즈=POMDP의 핵심.**

## 7. 다음 실험 (밴드 스윕 × 바람+충격+기동) — 설계 중
목표: "결과성 ∧ per-step애매 ∧ windowed분리" 겹치는 공격세팅 = POMDP 성립점.
- 패턴 waypoint(가감속 강함, 바람에 강건) · 바람 ws~9(benign 생존, 추락은 공격이 유발)
- **+ 랜덤 충격(반복 임펄스)**: benign 전이가 공격 온셋과 닮아 애매성↑ = POMDP
- 공격 torque bias 스윕(0.3~1.5) → 각 bias: 결과성(track추락?)·공격/평시 NIS겹침·windowed분리

## 산출물
- 표: `results_sm2_summary.txt`  · plot: `results_sm2_{bar,timeseries,wind}.png`
- offline 도구: `scratchpad/offline_ukf_sweep.py`(I/C/Q/R), `analyze_speedmod2.py`
- 메모리: `gyro-clean-maneuver-diagnosis-20260821`, `speedmod-accel-decel-gyro-20260821`

## 8. ★ 실측/확정값 참조 (설계 전 반드시 확인 — 2026-08-21 추가)
**공격 밴드 (swrl_config.py):**
- `attack_delta_range = (0.4, 0.8)` = 틸트 δ **권한비(정규화)**. 이게 실측 밴드.
- `attack_tq_authority_nm = 4.36` : δ→N·m 환산. 샘플러 `_bb`: gx=δ×4.36.
- tilt 브랜치: `tqx = gx/4.36` → 주입 정규화 δ.
- ⚠ **sweep로 δ 주입 시 value = δ×4.36** (δ0.4→1.74, 0.6→2.62, 0.8→3.49). torque Nm으로 δ 직접 넣으면 틀림.

**바람 (확정 2026-07-22 · 강풍 2026-08-19):**
- 외란: none 40% / wind_turbulence 60% (`disturbance_weights [0.4,0.6]`).
- 2-tier: 약풍 75% `wind_nominal_range (0.5,2.0)` / 강풍 25% `wind_strong_range (6.0,12.0)`.
- ws6=1.1N, ws12=4.5N (ws15는 추락 7N=53%). **ws9는 강풍 tier 안(유효, 추락X).**
- turbulence = OU(Ornstein-Uhlenbeck), ti=0.5 tb=2.0. **시작에 스파이크 아님(0에서 램프업).** 스파이크/gust 바람은 실측근거 없음 → 안 씀.

**센서/필터 (SIM_ALIGNMENT.md):** R_gyro 0.2(실기σ²0.0044의 45배, 고집) / Q_gyro 5e-3(격자최적) / 압축 min(log(1+√NIS),3).
