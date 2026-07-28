# TASK: STEP 1 — 패턴별 결과성 밴드 1차 거친 스캔 (경량판, 2026-07-23)

## 컨텍스트
- 참조: `HANDOFF_20260723.md`(§5~6), `ROADMAP_bandsweep_20260723.md`, `CLAUDE.md`.
- 배경: attack_flight_patterns aggressive 전용 버그를 전 패턴으로 수정 완료(swrl_config.py:117, sampler:388).
  기존 밴드 [1.34,1.40]은 aggressive 전용으로 판명 → 패턴별 밴드를 새로 측정.
- **이번 1차 스윕이 기존 results_deadline 스팟체크를 대체하는 재측정임** (온셋 고정 + 바람 0으로 조건이 더 깨끗함).
- 이번 작업 = 스윕 실행 + 판독까지만. 샘플러 코드 수정(STEP 2, `bias_scale_range_by_pattern`)은
  밴드 확정 후 별도 승인 받고 진행할 것.

## 설계 원칙 (경량화 — 사용자 지시)
- **1차 = 위치 잡기 전용**: 셀당 **ep=5**, 정책은 **track + dhover0 두 개만**.
  (하한 = track 꺾임, 상한 = dhover0 유지. dh3/dh5는 2차에서.)
- 2차 = 판정·논문수치: 1차로 좁힌 onset 주변 3~4점 × 4정책(track, dh0/3/5) × ep=15+.
- **온셋 = step 100 고정** (랜덤 온셋 금지 — 분산 제거, 저 ep 판독의 전제).
- **바람 = none/0 고정** (사용자 확정. 결과성 격리. 바람은 STEP 3에서 복귀).

## 사전 체크 (실행 전 필수)
1. `swrl_config.py`: attack_flight_patterns = flight_patterns 전체(4패턴) 반영 확인.
2. `env/ukf_filter.py:compute_nis_scaled`: 전 채널 통일 압축 `ln(1+ε)/(1+ln(1+ε))` offset=1.0 확인.
3. 스윕 config에서 공격 온셋 고정(step 100) 확인 — 랜덤 U(50,150)이면 고정으로 전환.
4. `pkill -f run_sim` 후 잔여 프로세스 확인. 결과 폴더는 신규 생성(기존 덮어쓰기 금지).

## 실행 — 4패턴 순차 (1차 그리드, combined ft1.5, ramp=0.0)
```bash
# figure8: 1.34~1.40 무해 관측(단 12ep 근거라 약함) → 1.40 포함해 위로 스캔
python3 online_rl_main.py --headless --sweep --sweep-mode combined \
  --sweep-pattern figure8 --sweep-values 1.40,1.48,1.56,1.64,1.72 \
  --ramp 0.0 --sweep-wind-type none --sweep-wind-speed 0 --speed 10 --outdir results_band_figure8

# circle: 데이터 없음 → 넓게
python3 online_rl_main.py --headless --sweep --sweep-mode combined \
  --sweep-pattern circle --sweep-values 1.34,1.42,1.50,1.58,1.66 \
  --ramp 0.0 --sweep-wind-type none --sweep-wind-speed 0 --speed 10 --outdir results_band_circle

# waypoint: 1.37 부분성립 관측 → 좁게
python3 online_rl_main.py --headless --sweep --sweep-mode combined \
  --sweep-pattern waypoint --sweep-values 1.33,1.35,1.37,1.39,1.41 \
  --ramp 0.0 --sweep-wind-type none --sweep-wind-speed 0 --speed 10 --outdir results_band_waypoint

# aggressive: 대조군(기존 20ep 밴드 [1.34,1.40] 재확인)
python3 online_rl_main.py --headless --sweep --sweep-mode combined \
  --sweep-pattern aggressive --sweep-values 1.32,1.34,1.36,1.38,1.40 \
  --ramp 0.0 --sweep-wind-type none --sweep-wind-speed 0 --speed 10 --outdir results_band_aggressive
```
- ep/cell 및 정책셀(track, dhover0만)은 스윕 config/CLI에서 위 설계 원칙대로 지정.
- hover는 flight_patterns에 없음(대응정책/캡처용) → 공격 스윕 대상 아님.
- **재스캔 규칙**: 어떤 패턴이든 전 셀 생존(그리드가 onset 아래) 또는 전 셀 사망(위)이면
  그리드를 해당 방향으로 한 스텝 폭만큼 이동해 1회만 재스캔.

## 게이트 (각 스윕 시작 30초 내 배너 확인 — 미일치 시 즉시 중단)
- ramp=0.0s / bias 목록 5개 전부 / 셀수 기대치 / pattern 일치 / wind=none / ep=5.
- ⚠ `--sweep-values`는 **쉼표 구분** 필수(공백이면 뒤 값 조용히 유실 — 실사고 2회). 배너 bias 개수로 검증.

## 집계·판독
- `python3 sweep_aggregate.py <dir>` → 패턴별 표: `bias | track생존 | dhover0 | crash_reason`.
- 1차 판독(ep=5 주의: 생존율은 0.2 단위 — 경향만 읽고 확정 금지):
  - onset 후보 = track 생존이 꺾이기 시작하는 구간(예: 1.00 → ≤0.6).
  - 상한 후보 = dhover0 이 1.00을 유지하는 마지막 bias.
- crash_reason 특이(altitude 외 flip/drift 등) 발견 시 기록.

## 산출물 (완료 보고 형식)
1. 패턴별 1차 표 4개 + onset/상한 후보 구간.
2. **2차 세밀 그리드 제안**: 패턴별 onset 주변 3~4점(0.02 간격), 4정책, ep=15+ — 예상 소요시간 포함.
3. `bias_scale_range_by_pattern` 초안 (제안만, 코드 반영 금지).
4. 이상 관찰 목록(게이트 불일치, 재스캔 이력, crash_reason 특이).

## 금지/주의
- 관측 구조 불변: 12D(`[nis_vel, nis_gyro, action]`×window4). obs_scale·dimS·window_size 수정 금지.
- 학습 설정(gamma, n-step, reward) 수정 금지 — 동결 후 단계.
- 호버를 "recovery"라고 부르지 말 것.
- 논문 수치는 2차에서만. 1차 ep=5 수치를 결론에 인용하지 말 것.
