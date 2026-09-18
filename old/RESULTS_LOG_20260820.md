# 결과 기록 — SWRL(RHUKF-FV) 프레임워크 확정 (2026-08-20~)

> 이 파일은 실험 결과·결정을 **계속 누적 기록**한다. 최신이 위.
> 물리·정합 단일출처: `SIM_ALIGN_CHANGES_20260812.md`. 서사·로드맵: `CLAUDE.md`.

---

## 확정 config (2026-08-20 현재)
| 항목 | 값 | 검증 |
|---|---|---|
| 공격 | torque 틸트 FDI, δ0.4-0.8, ON4-10/OFF4-8, const0.3 | 밴드 실측 |
| 관측 압축 | `min(log(1+√NIS), 3)` = 마할라거리 log1p clip3, 범위[0,3] | 온셋 d′ 2.48≈tanh√ |
| 리워드 | FN 딜레이 그래디언트(−0.4~−3.87 온셋가중)·TP/TN+0.5·FP−1·term−2.5 | v4 delay2·F1 0.88 |
| 강풍 | 약풍0.5-2(75%) / 강풍6-12(25%) | 15는 추락, ws10까지 survivable |
| 센서 σ | gyro[.066,.070,.044]·GPS수평0.9×·GPS수직0.219, UKF단, SENSOR_NOISE_SCALE | 실기 실측 |
| UKF R/Q | R_gyro0.2·R_vel0.1·Q_gyro5e-3 | 노이즈 입력서도 최적 불변(재검증) |
| RHUKF | r_init1.0·p_delta0.02·q1e-4·huber3·tau0.005·ui1·online_moving | 수렴T_Var 1.2=저-K |
| net | [24,24] 962params, 12D obs, window4 | 페어링 고정 |

---

## 핵심 발견 (누적)

### 2026-08-20 ★ gyro 채널이 기동에 안 튄다 (프레임워크 재설계 트리거)
- **패턴별 benign gyro NIS p90 = 0.1~0.37 (전 패턴, aggressive 포함).** 기동이 gyro aliasing 안 만듦.
  - 앞서 "gyro p90 120"은 **공격 회복구간 오염**이었지 기동 노이즈 아님 (windthreshold bias0로 확인).
  - plot: maneuver_gyro_check.png — omega 변해도 gyro NIS 바닥, 공격 때만 튐.
- **원인**: UKF 캘리브 = sim 플랜트 캘리브 동일(모델오차 0) + 순한 기동(ω̄0.2, MPC2.0캡). → gyro 부자연스럽게 깨끗.
- **문헌**: 기동 중 innovation 상승은 정설(IMM 필터·INS·기동탐지). 우리 sim이 안 그런 게 비정상.
- **해법(정당)**: `UKF_CALIB_ERR`로 현실적 캘리브 오차(±5%) 주입 → 기동서 gyro innovation 상승 = 자연 aliasing.
  "실기 탐지필터는 완벽캘리브 불가 → 현실적 오차 반영"으로 프레이밍. CLAUDE.md 로드맵 계획 축.
- **다음**: ctq/cth/m 각 5% 캡처 → clean과 비교, 기동 alialsing 생기는지 + 윈도우가 잡는지.

### 2026-08-20 센서 σ·Q/R — 재측정했으나 불변
- 실기 센서 σ 주입: **gyro NIS에 거의 무영향** (R_gyro0.2 >> σ²0.0044 + 바람 지배). 검증됨.
- T_Var "1.25→16"은 착각(수렴 vs 초반). 같은 단계선 15.08 vs 16.11 (+7%). → r_init1.0 유지 맞음.
- Q/R 노이즈 입력서 재튜닝: **최적 R0.2/Q5e-3 = 현행** 불변. REPORT §4-6 유효.

### 2026-08-20 패턴별 바람 crash 임계 (windthreshold benign)
- 전 패턴 **ws10까지 0% 추락**. ws12: aggressive/f8/wp 50%, hover 25%, circle 0%. ws15: 대량.
- → 자연 강풍 challenge는 **ws≤10** survivable.

### 2026-08-19 Adam v4 (관측 압축 확정 근거)
- v1~v4 비교: 압축 log(1+√NIS)clip3(v4)가 delay2.0·d≤2 100%·F1 0.88·FP0.001. 압축 포화가 병목이었음.
- v5 페어(RHUKF vs Adam, 동일): 둘 다 F1~0.88·delay2, 샘플효율 RHUKF 0.7@48 vs Adam 57(미미). → clean은 안 갈림.

---

## 전략 방향 (2026-08-20)
- **메인 기여 = SWRL(RHUKF-FV) 옵티마이저**, 탐지 아님(탐지는 임계값 trivial).
- RHUKF 우위 = **노이지 학습환경 강건성**. 단 **인위적 주입 X**, 프레임워크 내 자연 발생.
- 자연 노이즈원: ① 기동→gyro aliasing(현실적 모델오차 UKF_CALIB_ERR 필요) ② 강풍→vel aliasing.
- 추락 제외(survivable 밴드·ws≤10). gyro 중간 aliasing→윈도우가 잡음→정탐/오탐 높게·RHUKF>Adam 확정 목표.

## 진행 중 / 다음
- [진행] §6-2 fresh 캡처 (현재 config, clean gyro, aggressive×ws0-15) — "before(clean)" 기준.
- [다음] UKF_CALIB_ERR 5% 캡처 — "after(모델오차)" → 기동 gyro aliasing 대조.
- [대기] 프레임워크 확정 후 RHUKF vs Adam 본실험.
