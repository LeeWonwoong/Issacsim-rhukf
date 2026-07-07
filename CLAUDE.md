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
  → NIS는 χ² 통계량이 아니라 탐지 feature. log1p 압축.
- 네트워크 514 파라미터 고정(최적화기 공정비교용). agent: `--agent rhukf|adam`

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
  사인=crash_altitude(추력채널, 호버로 흡수불가·물리#4). pilot3 실패 재현. → combined 기각.
- **★ combined 기각 최종 근거 (2026-07-07): "탐지 신호 ∩ 대응 유지" = 공집합.**
  combined ft20~35 구간에서 vel/gyr NIS는 0.2~1.0으로 유의하게 활성화(=탐지가능성 ②는 열림)되지만,
  **동일 구간에서 dhover3 생존율이 0으로 붕괴**한다(hover 셀=1.0이나 지연전환 케이스는 basin 이탈=조건③ 실패).
  즉 vel이 유의하게 켜지는 곳에서는 d≤3 전환이 이미 물리적으로 늦다(추력채널 crash_altitude는
  전환 관성으로 흡수 불가·물리#4). **"vel 유의 활성화 ∩ dhover3 유지"가 combined에서 명확히 공집합** →
  탐지 데드라인이 대응 데드라인보다 뒤에 놓임 = 3조건 동시성립 불가. torque-only는 이 교집합이 열림(밴드 내 d≤3=1.0).
- crash lag: 밴드내(1.30) dhover는 전부 timeout=실제 생존(summary survived=1). 실사망은 1.35(밴드밖·hover도0)
  에서만 crash_flip/drift, 전환후 lag 13~50스텝 = 지연 basin 발산(물리#3). **전환 과도 자체는 무해 재확인.**
  ※ 데이터 무결성 확인(2026-07-06): summary CSV 1600행 전수 검사 survived↔crash_reason 불일치 0.
    _end_sweep_episode/sweep_aggregate 정상. (어제 crash-lag 임시스크립트가 detail의 terminal timeout을
    death로 오집계한 것이 유일한 아티팩트 — 데이터/집계 파이프라인 버그 아님.)
- 통합 그림: `plot_fine_staircase.py` → deadline_staircase.png (한글폰트 Noto CJK 적용됨).

## ★ 공격 세팅 동결 (2026-07-06, FROZEN) ★
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
- **동결값**: 공격채널 = **torque(-only)**, ramp = **0.0**(abrupt, ramp무관 실측), **bias_scale_range = [1.30, 1.32] Nm**(핵심밴드).
  - 학습 샘플링 권장: 하단 onset(1.28, track0.90) 마진 포함해 **[1.28, 1.32]** 로 스프레드(그래디언트 확보), **≥1.34 제외**(hover 붕괴=회복불가는 "탐지=위협" 프레이밍 위반).
- 그림: `plot_fine_staircase.py` → deadline_staircase.png 에 FINAL 20ep 패널(밴드 음영) 추가됨.

### config 정렬 (동결 반영, 2026-07-07 완료)
- swrl_config.py 를 torque-only로 정렬: `bias_ft_ratio=5.0 → 0.0`(thrust 성분 제거), `bias_scale_range=(1.0,1.33) → (1.28,1.32)`(핵심밴드+하단 onset 마진).
  `sample_bias_box=True`(tube 샘플러) 유지 → 이제 (torque_xy=s, torque_z=0.2·s, thrust=0) = 동결 torque 채널. `bias_yaw_ratio=0.2` 유지(동결 sweep의 torque 모드가 yaw=0.2·b 포함).
- 옛 combined-ray 주석(line 98~99, 102: s∈[1.2,1.3])도 torque-only 밴드[1.28,1.32]로 갱신.
- **⚠ 관측 차원은 절대 건드리지 않음**: `[nis_vel, nis_gyro, action]` × window4 = dimS 12 구조 유지.
  torque-only 공격에서도 **vel_NIS는 정상 대비 4~7배 상승하는 보조 탐지 신호**이므로 vel 채널 제거 금지.
  config 정렬은 **공격 샘플러만** 바꾼다(bias_ft_ratio/scale_range) — obs_scale·dimS·window_size 불변.

## 동결 후 순서 (합의된 로드맵)
E3: 밴드 3점 × {aggressive, circle} 패턴 스팟체크(밴드 이동 확인)
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
