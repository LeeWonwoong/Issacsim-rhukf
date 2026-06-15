# α-SWEEP 사용법 (결과성 밴드 + 탐지가능성 + CUSUM baseline)

## 무엇을 확인하는가
online RL이 "정직하게" 성립하려면 **강제호버라는 행동이 결과를 바꿔야** 한다.
sweep은 그 전제를 데이터로 검증한다:

1. **결과성 밴드** — `track 추락 ∧ hover 생존`인 α 구간이 실재하는가?
   (없으면 행동이 no-op → 분류로 강등 → online RL 프레이밍 무효)
2. **탐지가능성** — 그 밴드에서 raw NIS가 정상과 갈리는가?
3. **CUSUM baseline** — 고정임계/CUSUM의 탐지지연·FAR (= RHUKF-RL이 이겨야 할 숫자)
4. **maneuver-FP** — 정상 aggressive 기동만으로 NIS가 튀는가? (→ `ukf_q_gate_gyro` 필요 여부)

## 실행
```bash
# 1) sweep 돌리기 (학습 OFF, 고정정책 순회). headless 권장.
python online_rl_main.py --sweep --headless
#    → results/sweep_detail.csv, results/sweep_summary.csv 생성

# 2) 집계/분석 (Isaac 불필요, 서버에서 단독)
python sweep_aggregate.py results
```

## 셀 구성 (swrl_config.py 에서 조정)
```python
sweep_alphas   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]   # 밴드 근처는 0.05로 정밀화
sweep_pattern  = 'aggressive'   # track 셀: 명령토크 최대 = 최악조건
sweep_episodes = 4              # 셀당 반복(RNG 노이즈)
sweep_attack_start = 30         # step@10Hz, 이후 ramp 1.0s
```
- 셀 = baseline 2개(무공격 aggressive/hover) + α마다 (track, hover).
- track 셀: aggressive 비행 + 공격 → 토크결손으로 자세붕괴 → 낮은 α에서 추락.
- hover 셀: 원점 호버(명령토크≈0 → 토크LoE≈0) → 추력LoE TWR마진(α≲1−1/TWR)까지 생존.
- 둘이 갈리는 사이 = 결과성 밴드.

## sweep 결과 → 본학습 반영
| sweep 산출 | 반영 위치 |
|---|---|
| 결과성 밴드 [lo, hi] | `curriculum_fixed_min`, `curriculum_end_max`, `eval_scenarios` 강도 |
| 정상 NIS 99pct | CUSUM/고정임계 baseline 임계값 |
| baseline-aggressive NIS 높음 | `ukf_q_gate_gyro` 켜기 (예: 0.05~0.2) |
| track→crash_flip/altitude 확인 | terminal 교사 성립 → `use_logical_done=False` 유지 확정 |
| track이 drift만(추락 X) | logical_done 재고 또는 강도/패턴 상향 |

## 이번 빌드의 주요 변경 (대화 종합)
- **고집 필터**: `DynamicsUKF` R을 실측 노이즈에 정합(0.5→0.01/0.02/0.03) + ff 1.02→1.0.
  고집은 **low Q**로 주고 **R을 키우지 않음**(R↑은 NIS 분모를 키워 재희석). 필터는 PX4 *명령* u를
  쓰므로 "명령 vs 실제" 불일치(=공격)가 지속 잔차로 남음. + maneuver-gated Q 훅(기본 off).
- **공격**: `loe_thrust` 제거(보상되어 무해) → `loe_combined`만. ramp 0.6→1.0s(10스텝 대응여유).
- **탐험**: `eps_action_probs` 90/10 → 50/50 (공격∧hover 쌍 커버).
- **logical_done = False 유지**: 강제호버+결과성 공격이면 행동이 플랜트를 바꿔 이미 MDP.
  logical_done은 행동이 세상을 안 바꿀 때(per-step O/X) 쓰는 크러치라 불필요(이중처벌 방지).
- **RHUKF 학습기 하이퍼는 그대로**(CartPole/LunarLander 검증분 — 과제만 고침).

## 트러블슈팅: headless에서 "Heartbeat lost" 무한 HARD_RESET
- 원인: 헤드리스는 렌더 스로틀이 없어 `run_sim` 루프가 실시간보다 폭주 → PX4 SITL lockstep 붕괴 → GT odometry 미발행 → heartbeat 타임아웃.
- 조치(이 빌드 적용됨):
  1. `run_sim.py run()` 루프를 **실시간 페이싱**(sim_time이 wall-clock 앞서면 sleep). GUI의 60fps 스로틀과 동일 효과를 헤드리스에서 재현.
  2. `online_rl_main.py` heartbeat_timeout **20→40초**(기동/리셋 여유).
- 그래도 GT가 아예 안 오면 `run_sim`가 헤드리스에서 죽은 것 → 로그 확인:
  ```bash
  cat ./results/sim_process.log
  ```
  (PX4 SITL 미연결/포트 충돌/USD 에러 등이 여기 찍힘.)
