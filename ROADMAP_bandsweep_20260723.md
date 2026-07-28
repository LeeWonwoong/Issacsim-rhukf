# ROADMAP — 전 패턴 공격 밴드 재보정 (2026-07-23)

## 왜 (배경)
- **버그 수정**: `attack_flight_patterns=['aggressive']` → flight_patterns 전체(waypoint/circle/figure8/aggressive).
  실제 공격은 모든 기동 궤적에 랜덤 세기로 주입 = 의도된 설계(사용자 확정). swrl_config.py:117, sampler:388.
- **문제**: 동결 밴드 [1.34,1.40]은 **aggressive 전용**. results_deadline 분석(2026-07-23):
  - figure8: track 전 bias 생존 → 공격 무해(①붕괴). onset > 1.40.
  - waypoint: 1.37만 부분성립(track 0.25 / dh0 1.00, 단 dh3=0.17로 데드라인 빡빡).
  - hover*: track 생존 + 1.40서 호버전환 사망(soft-hold 이슈). (*hover는 flight_patterns에 없음=공격패턴 아님, 대응/캡처용)
  - **aggressive = 최악(가장 취약) 케이스**. 순한 패턴은 제어여유로 같은 bias 흡수 → 밴드 위로 이동/무해.
- **함의**: [1.34,1.40] 균일 주입 시 figure8/circle 공격이 비결과성 → "공격분포를 결과성 밴드로 한정해 탐지=위협이 설계상 참" 보증 붕괴.

## 결정된 것 (합의)
1. **패턴별 결과성 밴드 재스윕** (Option A). 공격은 전 패턴 유지.
2. **밴드 폭 = 물리범위 `[onset, hover한계]`를 패턴별로 꽉 채워 넓게.**
   - 근거: 밴드 안이면 전 구간 ①∧③ 성립 → 넓혀도 `탐지=위협` 안 깨짐(하한 아래만 금지).
   - severity 다양성 = robustness + 현실성 + 샘플효율 서사. 좁게 = overfit + 일반화주장 약화.
   - 하한(near-threshold)이 RHUKF-FV vs Adam 판별구간 → 하한 커버 + 평가지표가 하한 주목.
3. **스윕 외란 = 바람 0** (`disturbance=none`). 결과성을 외란과 격리. 바람은 학습/관측 단계에서 복귀.
4. combined ft1.5, ramp0.0 (동결 채널·주입방식 유지). 공격 온셋 고정(@step100, 밴드측정 격리).

## STEP 1 — 패턴별 밴드 스윕
- **패턴**: waypoint, circle, figure8 (+ aggressive 대조로 [1.34,1.40] 재현 확인).
- **bias 그리드 (기존 데이터 기반 예상, 실측으로 좁힘)**:
  | 패턴 | 예상 밴드 | 1차 스캔 그리드 |
  |---|---|---|
  | waypoint | 좁고 낮음 (~1.35–1.40) | 1.33,1.35,1.37,1.39,1.41 |
  | circle | 미지 | 1.34,1.40,1.46,1.52 (넓게 스캔→좁힘) |
  | figure8 | 높음 (onset>1.40) | 1.40,1.48,1.56,1.64,1.72 |
  | aggressive | [1.34,1.40] 재현 | 1.32,1.36,1.40 |
- **정책**: track(①결과성 대조) + dhover{0,3,5} (③대응). ep/cell = **20**(논문수치).
- **CLI 예시**(패턴별 개별 실행, 그리드 다름):
  ```
  python3 online_rl_main.py --headless --sweep --sweep-mode combined \
    --sweep-pattern figure8 --sweep-values 1.40,1.48,1.56,1.64,1.72 \
    --ramp 0.0 --sweep-wind-type none --sweep-wind-speed 0 --speed 10 \
    --outdir results_band_figure8
  ```
- **1차는 거친 그리드로 onset/상한 위치 파악 → 2차 세밀 그리드**(0.02 간격)로 밴드 확정.

## STEP 1 판정기준 (패턴별 밴드 정의)
- **하한** = track 생존율이 유의 하락 시작(결과성 onset; track ≲ 0.5 되는 첫 bias).
- **상한** = dhover0(=hover) 생존율 유지 마지막 bias(그 위=hover도 붕괴=회복불가).
- **밴드 = [하한, 상한]** 이면서 **dhover d≤3 = hover(1.0)** 구간(조건③).
- 각 패턴 밴드를 aggregate 표로: `pattern | 하한 | 상한 | 폭 | d≤3 데드라인`.

## STEP 2 — 샘플러 코드 확장 (밴드 확정 후)
- 현재 `bias_scale_range` 단일 → **패턴별 dict** 로 확장 (`bias_scale_range_by_pattern`).
- `sample_episode_scenario`: 패턴 뽑은 뒤 해당 패턴 밴드에서 `s ~ U(lo,hi)` (기존 ±10% jitter/클립 로직 유지, 클립 상한도 패턴별).
- 단위검증: 4패턴 각각 밴드 내 샘플·thrust=1.5·s 확인.

## STEP 3 — 관측 재산출 (P2, 추가실행 최소)
- 압축 통일(offset=1.0)로 기존 d′/baseline 무효 → 재산출 필요분과 합침.
- **전 패턴 공격 vs 전 패턴 기동** 분리도: raw NIS / 압축 NIS / +pos NIS(12D vs 16D).
- 판정: 침묵구간 분리도 개선폭 = 16D(pos 포함) 채택 여부. (band 확정 후 캡처 데이터로 분석만)
- 바람(none40/turb60, 1–5) 이 단계에서 복귀 → 공격 vs 바람 분리도 포함.

## STEP 4 — 동결 & 본실험
- 환경·관측 동결 → 본 페어링 **seed 3개** (Adam γ0.85 + RHUKF 동일 config).
- 공격밀도(burst 수) 축 학습곡선. RHUKF-FV vs Adam vs CUSUM.

## 미해결 / 주의
- **hover-패턴 1.40 dhover0 사망(soft-hold)**: results_deadline_softhold 비교 이미 존재 → soft-hold가 구제하는지 확인 후 대응정책 확정(hover는 공격패턴 아니라 우선순위 낮음).
- **호버=자세안정**: 제어적 사실(setpoint→roll/pitch/rate 0)이나 CLAUDE.md 프레이밍 "recovery라 안 부름"과 충돌 소지 → **메커니즘 관찰로 유지, 서사 승격 보류**(승격 시 stopping-POMDP와 양립방법 별도 결정).
- **per-pattern 밴드 heterogeneity**: 공격분포가 패턴마다 달라짐 → 로깅에 pattern별 bias_scale 기록 유지(분석 분리).
- `--sweep-values`는 **쉼표 구분**(공백이면 뒤 값 조용히 버림 — 실사고 2회).
