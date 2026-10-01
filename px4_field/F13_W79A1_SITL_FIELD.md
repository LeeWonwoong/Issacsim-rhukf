# 학습된 SWIRL(w79·A-1) 비행 중 추론 검증 — SITL → 실기 (2026-10-01)

모델 = **SWIRL w79·A-1 시드 42** (사전 등록: 학습 후반 181–200 후회 중앙값 시드, PREREG_rwA1_0930.md)
- `px4_field/models/swirl_w79A1_s42.npz` — export_policy.py 로 내보냄 (θ_T · silu · 18입력 = 창 6 × [vel, gyro, 직전 행동] · log1p_sqrt·clip 4·div 4)
- `px4_field/models/swirl_w79A1_s42_config.yaml` — 학습 런의 config.yaml 그대로 (UKF 노브·failsafe·공격 프로파일의 출처)
- 오프라인 검증(10-01): numpy 정책 vs 학습 망 Q 최대 차 1.5e-5·greedy 100% 일치 / 실기 관측 생성기가 Isaac 기록 입력을 1120스텝 차 0 으로 재현

모든 명령은 저장소 루트 `~/projects/Issacsim-rhukf` 기준. ROS 환경: `source /opt/ros/humble/setup.bash; source ~/colcon_ws/install/setup.bash`

---

## 0. 시작 전 확인 (로컬 Isaac 학습 대기열이 멈춰 있어야 함)

```bash
tail -3 results/claudecodefortest/isaac_w79/orchestrator.log      # "일시 정지 시작" 이 보여야 함
pgrep -af "run_sim.py|bin/px4|train.py"                            # 아무것도 없어야 함
```
- 정지는 60분 + "run_sim·px4·f13 가 2분 연속 없을 때" 자동 해제된다. **검증이 길어지면** `touch results/claudecodefortest/HOLD_PLAN_W79` (끝나면 `rm` 해야 재개).
- 잔재 px4 가 있으면: `pkill -9 -f bin/px4` (3초 뒤 진행).

## 1. SITL 한 비행: 정상 → 바람 → 바람+공격 → 공격 → 정상

패턴 시계(오프보드 패턴 시작 = 0 s, circle 4바퀴 ≈ 66 s) 기준 일정:

| 구간 | 시간 | 설정 |
|---|---|---|
| 정상 | 0–15 s | 무풍 |
| 바람 | 15–30 s | `--sitl-wind 15,35,8` → 8 m/s 난류(학습과 같은 wind_turbulence) |
| 바람 + 공격 | 30–35 s | `--attack-fixed 0.5,30,100` → δ 0.5 (램프 0.3·0.6·0.9 → 0.5 유지), 100스텝 = 10 s |
| 공격 | 35–40 s | 바람 꺼짐(35 s), 공격 계속 |
| 정상 | 40–66 s | 무풍·무공격 |

**터미널 A — Isaac + PX4 SITL + 정책 노드 (무인 자동 이륙)**
```bash
cd ~/projects/Issacsim-rhukf
HEADLESS=0 WIND_MOMENT_ARM=0.05 bash px4_field/launch_isaac_manual.sh f13 1.0 \
  --model models/swirl_w79A1_s42.npz \
  --knobs models/swirl_w79A1_s42_config.yaml \
  --attack-cfg models/swirl_w79A1_s42_config.yaml \
  --attack dds --attack-fixed 0.5,30,100 --attack-max 0.8 \
  --sitl-wind 15,35,8 \
  --pattern circle --laps 4 \
  --min-hover-steps 0 \
  --sitl-auto 2.5 --fs-url udpin:0.0.0.0:14540 --no-fetch
```
- `WIND_MOMENT_ARM=0.05`: 학습 런의 sim_env 와 같게(바람력이 COM 위 5 cm 에 작용 → 회전 모멘트). 빠지면 같은 8 m/s 라도 학습보다 덜 흔든다.
- `HEADLESS=0` 이면 Isaac 화면이 뜬다(무거우면 `HEADLESS=1`).
- `--sitl-auto 2.5`: SITL 전용 자동 arm·OFFBOARD·2.5 m 이륙(**실기 금지**). 조종기로 직접 하려면 이 옵션을 빼고 QGC/조종기로 이륙 → 오프보드 스위치 ON.
- `--min-hover-steps 0`: 학습과 같은 결정(실행 hover 최소 유지 없음). 실기 안전 규칙(기본 5)과 비교하려면 5 로 한 번 더.
- 노드 로그에 `SITL 바람 일정`, `[WIND] t_pat 15.0s → 바람 8.0`, `공격 계획: 사건 1 fixed_persist@300`, `[WIND] … → 바람 0.0` 이 차례로 보여야 한다.
- 끝: 패턴 완료 후 자동 착륙·정리, 또는 Ctrl+C (엔진·px4 정리됨).

**터미널 B — 실시간 그림 (관측값·행동·Q 차·δ·3D 궤적·탑뷰)**
```bash
cd ~/projects/Issacsim-rhukf/px4_field
python3 plot_policy_log.py --latest field_logs --live
```
- 왼쪽: 3D 궤적(파랑 track · 주황 공격 중 track · 빨강 hover)과 탑뷰(명령 설정점 대비, 무공격 추종 RMSE).
- 오른쪽 시간축 4줄: gyro·vel 관측값(정책 입력) / 공격 δ / 정책 행동·실제 hover / Q(hover)−Q(track).
- 배경: **파랑 = 바람**, 주황 = 공격, 빨강 = hover. 0.5 s 마다 갱신.

**비행 뒤 — 그림 저장·로그 확인**
```bash
cd ~/projects/Issacsim-rhukf/px4_field
python3 plot_policy_log.py --latest field_logs --out field_logs/w79A1_s42_sitl_wind_attack.png
ls -t field_logs/f13_policy_circle_*.csv | head -1                 # 10 Hz CSV: NIS·관측·Q·행동·δ·위치·설정점·wind·정책 입력 o0–o17
```

**기대(학습 결과 기준)**: 정상·바람 구간 Q 차 ≈ −1 (track), 공격 시작 후 1–3스텝 안 hover 선언, 공격 중 Q 차 ≈ +1, 공격 끝나고 1–2스텝 안 track 복귀. 바람 구간에서 gyro 관측값이 0.3–0.5 로 오르지만 hover 하지 않는 것이 강풍 aliasing 강건성.

### 변형 (원하면 한 번씩)
| 목적 | 바꿀 인자 |
|---|---|
| 약한 공격(강풍 속 어려운 경우) | `--attack-fixed 0.25,30,100` |
| 실기 안전 규칙 그대로 | `--min-hover-steps 5 --attack-max 0.5` |
| 학습 궤적 다른 것 | `--pattern figure8` / `aggressive` / `scurve` (시간이 짧으면 `--laps` 늘림) |
| 비교 기준 (Adam) | Adam 모델 export 후 `--model` 만 교체 (아래 3) |

## 2. 실기 (젯슨) — 첫 비행은 shadow

```bash
# 노트북: 젯슨으로 배포 (models/ 폴더 통째로 — 새 npz·config 포함)
JETSON=quad@<젯슨IP> bash px4_field/deploy_jetson.sh

# 젯슨: ① shadow — 정책은 판단·기록만, hover 를 걸지 않음
cd ~/swirl_field/px4_field
python3 f13_policy.py --model models/swirl_w79A1_s42.npz --knobs models/swirl_w79A1_s42_config.yaml \
  --attack-cfg models/swirl_w79A1_s42_config.yaml --pattern circle --attack rc --fs-url /dev/ttyACM0 --shadow

# 젯슨: ② 실제 hover (shadow 결과 확인 뒤) — 조종기 aux 로 공격, δ ≤ 0.5
python3 f13_policy.py --model models/swirl_w79A1_s42.npz --knobs models/swirl_w79A1_s42_config.yaml \
  --attack-cfg models/swirl_w79A1_s42_config.yaml --pattern circle --attack rc --atk-tq-max 0.5 --fs-url /dev/ttyACM0

# 젯슨 또는 노트북(로그 동기화 시): 실시간 그림
python3 plot_policy_log.py --latest field_logs --live
```
- 현장 규칙: δ 상한 0.5 (강공격 0.7+ 금지), 첫 비행 `--shadow`, 실행 hover 최소 5스텝(기본), `--sitl-auto`·`--sitl-wind` 쓰지 않음(실기에서는 바람을 만들 수 없다 — 자연풍).
- 비행 뒤 `postflight` 가 ulog 를 회수·판정한다(`--no-fetch` 로 끔).

## 3. 다른 모델로 바꾸기 (같은 절차, 모델만 교체)
```bash
~/isaacsim/python.sh px4_field/export_policy.py results/claudecodefortest/isaac_w79/<런>/final_model.pt px4_field/models/<이름>.npz
cp results/claudecodefortest/isaac_w79/<런>/config.yaml px4_field/models/<이름>_config.yaml
```
- export 는 런 폴더 config.yaml 의 관측 규약(압축·clip·div·창·특징)과 대조해 다르면 멈춘다(10-01 추가).
- Adam 모델은 `--net target` (기본) 그대로 — 평가 규칙 동일.
