# 실기 데이터 수집 핸드오프 — 2026-08-04

> 브라우저 Claude 용 자립 문서. 저장소 접근 없이도 이어서 논의할 수 있도록
> 사실·근거·미결을 전부 담았다. 코드 인용은 `파일:행` 으로 표시.

---

## 0. 한 줄 요약

UKF 모델 계수(C_thrust · G=C_torque/I · drag)를 **실기체에서** 뽑기 위한
오프보드 데이터 수집(F1/F2/F3) 준비 단계.

## 0-1. ★ 3단계 로드맵 (사용자 확정)

```
  1단계  프로펠러 제거 지상확인        ← 통과 (창1/창2 정상)
         하드웨어 배선·오프보드 진출입·시퀀스 순서

  2단계  Isaac Sim 에서 조종기 수동조종으로 확인    ← ★ 지금 할 일
         "조종기로 띄우고 → 오프보드 스위치 → 스크립트가 시퀀스 비행"
         전 과정을 가상에서 재현. 실비행 거동을 미리 본다.
         사용자가 별도 프로젝트에 Isaac Sim + 조종기 수동조종을 이미 갖고 있고,
         그것을 이 프로젝트에 코드화해 반영하는 것이 목표.

  3단계  실외 비행 (F1 → F2 → F3)
         계수 추출. FIELD_PROCEDURE.txt
```

병행 작업: **DDS yaml 토픽 추가**(§3 #2). 하기로 확정됨.

---

## 1. 하드웨어 · 소프트웨어 실물 구성

| 항목 | 값 | 확인 방법 |
|---|---|---|
| FC | **PX4 v1.15.4**, HW `PX4_FMU_V6C` (V6C002002), STM32H7 | nsh `ver all` |
| FC 빌드일 | 2025-12-10 | 같음 |
| OS | NuttX 11.0.0 | 같음 |
| 기체 AUW | 1.372 kg (sim 정합값) / FIELD_PROCEDURE 는 1.326 kg 표기 ⚠불일치 | — |
| 온보드 컴퓨터 | Jetson, Ubuntu + **ROS2 Humble** | — |
| px4_msgs | **2.0.1** (colcon_ws, v1.18 계열과 MESSAGE_VERSION 일치) | package.xml |
| PX4 소스 트리 | `~/PX4-Autopilot` = **v1.18.0-alpha1-208** (main) | git describe |
| DDS 브리지 | MicroXRCEAgent (uXRCE-DDS) | — |
| ROS_DOMAIN_ID | 현재 **0** (설정 안 돼 있었음, 잘 동작 중) | — |

> ⚠ **핵심 불일치**: FC 는 v1.15.4 인데 소스 트리는 v1.18.0-alpha1 이다.
> 이 간극이 오늘 조사한 모든 토픽 문제의 뿌리다.

---

## 2. 오늘 확정된 사실

### 2-1. 없는 토픽 3개의 정체

`px4_field/check_ros2_topics.sh` 가 3개를 ✗ 로 표시했다. 원인이 서로 다르다.

**(a) `sensor_gps` — 이름이 틀렸다**
Isaac Sim 쪽 이름이다(`run_sim.py:153` 이 `/sim/sensor_gps` 로 발행).
실기 PX4 는 **`/fmu/out/vehicle_gps_position`**, 타입은 **똑같이 `SensorGps`**.
v1.15.4 기본 yaml 62행에 이미 있다 → 그래서 GPS 는 실제로 보이고 있었다.
→ 스크립트 이름만 정정함(수정 완료).

**(b)(c) `vehicle_thrust_setpoint` / `vehicle_torque_setpoint`**
v1.15.4 의 `dds_topics.yaml` 을 직접 열어 확인:

```
 62:  - topic: /fmu/out/vehicle_gps_position      ← 있음
 71:  - topic: /fmu/out/vehicle_status            ← 있음
151:  - topic: /fmu/in/vehicle_thrust_setpoint    ← in 만 (오프보드 입력용)
154:  - topic: /fmu/in/vehicle_torque_setpoint    ← in 만
157:  - topic: /fmu/in/actuator_motors            ← in 만
 47:  # - topic: /fmu/out/vehicle_angular_velocity  ← 주석 처리
```

**우리가 필요한 건 `/fmu/out` 방향**(= PX4 컨트롤러가 명령한 값을 읽는 것)인데
v1.15.4 에는 publications 에 없다. **그냥 추가하면 된다.**

### 2-2. `_v<N>` 접미사 문제는 해소됨

PX4 는 `MESSAGE_VERSION != 0` 인 메시지의 DDS 토픽명에 `_v<N>` 을 자동으로
붙인다(`utilities.hpp:35`, `dds_topics.h.em:83` — yaml 이 아니라 msg 정의에서 옴).

- v1.18 트리 기준: `VehicleStatus`=4, `VehicleAttitudeSetpoint`=1, `VehicleLocalPosition`=1
- **v1.15 는 메시지 버저닝 도입 전** → 접미사 없음

→ **v1.15.4 트리에서 토픽만 추가하면 이름이 바뀌지 않고 아무것도 안 깨진다.**
(v1.18 로 점프하면 위 3개가 `_v4`/`_v1` 로 바뀌어 `offboard_common.py:112,115,118` 수정 필요.
지금은 그 경로를 택하지 않는다.)

### 2-3. ★ 계수 추출은 DDS 가 아니라 ulog 경로다

`fit_from_ulog.py:150-155` 가 전부 `.ulg` 에서 읽는다:
```python
d_thr = topic(ulog, 'vehicle_thrust_setpoint')
d_tq  = topic(ulog, 'vehicle_torque_setpoint')
```
그리고 그 토픽들은 PX4 **기본 로깅 프로파일**에 이미 있다
(`src/modules/logger/logged_topics.cpp`, 전부 `add_default_topics()` 안):

| 토픽 | 기본 로깅 레이트 | 행 |
|---|---|---|
| `sensor_combined` | — | 123 |
| `vehicle_gps_position` | 10 Hz | 145 |
| `sensor_gps` | 1 Hz ⚠느림 | 222 |
| `vehicle_thrust_setpoint` | 50 Hz | 238 |
| `vehicle_torque_setpoint` | 50 Hz | 239 |

**→ DDS 토픽 3개가 없어도 F1/F2/F3 계수 추출에는 전혀 지장이 없다.**
DDS 노출은 나중에 젯슨에서 UKF 를 **실시간**으로 돌릴 때(배포 단계) 필요하다.

### 2-4. 지상 벤치의 한계 (프로펠러 제거)

✅ 검증되는 것: 오프보드 진입/이탈, setpoint 발행 게이트, 시퀀스 순서·타이밍,
   자세명령 → 모터 4채널 분배의 **부호·축·순서**

❌ 원리적으로 안 나오는 것 — 기체를 책상에 올리든 고정하든 동일:
```
C_thrust = m·g / u_hover     ← 실제로 무게를 지탱해야 u_hover 가 생김
G_i      = ω̇_i / cmd_i       ← 실제로 각가속도가 있어야 함
drag     = f(v)              ← 실제로 움직여야 함
```
정지 상태에선 셋 다 0 이라 분모가 0 이거나 신호가 없어 회귀가 성립하지 않는다.

관련 PX4 동작: `mc_rate_control.cpp:220` 이 land detector 플래그를
`RateControl::update()` 에 넘겨 **적분기만 동결**시킨다. P·D 는 살아 있으므로
자세 setpoint 변화 → 모터 명령 분배는 벤치에서도 실제로 관측된다.
단 자세가 안 변해 오차가 계속 남으므로 **크기는 비현실적으로 커진다**
(부호·축·순서만 볼 것).

### 2-5. 미해결 현장 증상 — 인계 후 왼쪽 스틱 무반응

시퀀스 완주 후 Ctrl-C 없이 조종기를 수동으로 되돌리면 **왼쪽 스틱(스로틀)이 안 먹는다.**

가설: `/fmu/in/vehicle_attitude_setpoint` 는 PX4 내부 uORB `vehicle_attitude_setpoint`
인데 **Stabilized 의 수동 자세 FlightTask 도 같은 토픽에 발행**한다.
bench 스크립트가 쏘는 게 정확히 그 토픽(`thrust_body` 고정값)이라, 두 발행자가
10 Hz 로 교대하면 스로틀이 스틱값과 고정값 사이를 오간다.
(2026-08-03 에 `trajectory_setpoint` 로 겪은 것과 같은 종류의 uORB 충돌)

판정 방법 — 수동으로 되돌린 **직후**:
```
ros2 topic hz /fmu/in/vehicle_attitude_setpoint
```
· 계속 10 Hz → 스크립트가 범인 (`_may_stream()` 게이트가 안 닫힘)
· 조용함 → 스크립트 무관

잠정 규칙: **조종기를 수동으로 되돌리기 전에 스크립트를 먼저 Ctrl-C.**

관련 코드: `offboard_common.py:249` 의 `_may_stream()` 은
`offboard or user_intent==OFFBOARD or stream_armed` 인데 **`stream_armed` 가
영구 래치**다(Enter 한 번 누르면 해제 코드가 없음). 후보 수정:
```python
elif self.state == 'ENGAGED':
    if not self.offboard:
        self.stream_armed = False    # ← 추가
```

---

## 3. 미결 항목

| # | 항목 | 결정에 필요한 것 | 단계 |
|---|---|---|---|
| 1 | **토픽 접미사 자동 감지** (§3-A) | 실기 v1.15.4 ↔ Isaac SITL v1.18 양립 코드 | **2단계 선결** |
| 2 | **DDS yaml 토픽 추가** — 하기로 확정 | v1.15.4 체크아웃 → yaml → 빌드 → 플래시 | 병행 |
| 3 | 왼쪽 스틱 증상 (§2-5) | 되돌린 직후 `hz` 결과 | 3단계 전 |
| 4 | AUW 표기 불일치 | 1.372 (CLAUDE.md) vs 1.326 (FIELD_PROCEDURE) 실측 확정 | F1 분석 전 |
| 5 | `check_ulog.py` GPS 소스 우선순위 | `sensor_gps`(1Hz) 대신 `vehicle_gps_position`(10Hz) 우선으로 | F1 분석 전 |

### #2 DDS yaml — 정확한 절차 (하기로 확정됨)

```bash
cd ~/PX4-Autopilot
git checkout -b field-v1.15.4 v1.15.4     # FC 와 같은 버전으로
# dds_topics.yaml 의 publications 에 추가:
#   - topic: /fmu/out/vehicle_thrust_setpoint
#     type: px4_msgs::msg::VehicleThrustSetpoint
#   - topic: /fmu/out/vehicle_torque_setpoint
#     type: px4_msgs::msg::VehicleTorqueSetpoint
#   - topic: /fmu/out/actuator_motors          # 벤치 모터 관찰용
#     type: px4_msgs::msg::ActuatorMotors
#   (47행 vehicle_angular_velocity 주석 해제도 유용)
make px4_fmu-v6c_default
# QGC → Vehicle Setup → Firmware → Advanced → 커스텀 .px4
```
⚠ 주의: `~/PX4-Autopilot` 의 main(v1.18) 브랜치에는 Isaac Sim SITL 빌드
(`build/px4_sitl_default`)가 물려 있다. 브랜치를 바꾸면 SITL 재빌드가 필요하다.

---

## 3-A. ★ 2단계(Isaac Sim) 통합 과제 — 브라우저 Claude 가 맡을 일

### 목표
사용자가 별도 프로젝트에 갖고 있는 **Isaac Sim + 조종기 수동조종** 환경에서,
`px4_field/` 의 오프보드 시퀀스(F1/F2/F3)를 그대로 돌려본다.
"조종기로 이륙 → 오프보드 스위치 ON → 스크립트가 시퀀스 비행 → 스위치 OFF 인계"
전 과정을 가상에서 재현하는 것. 그 결과를 이 프로젝트에 코드화해 반영한다.

### ⚠ 먼저 해결해야 할 문제 — 토픽 이름 버전 불일치 (실측 확인함)

두 환경의 PX4 버전이 다르고, **v1.16+ 는 versioned 메시지의 DDS 토픽명에
`_v<N>` 을 자동으로 붙인다**(`utilities.hpp:35`, `dds_topics.h.em:83`).

| 환경 | PX4 | 토픽명 |
|---|---|---|
| 실기 FC | v1.15.4 | `/fmu/out/vehicle_status` (접미사 없음) |
| Isaac SITL | v1.18.0-alpha1 | `/fmu/out/vehicle_status_v4` |

생성된 SITL 헤더에서 실물 확인:
`build/px4_sitl_default/uORB/topics/vehicle_status.h:152` →
`static constexpr uint32_t MESSAGE_VERSION = 4;`

**영향받는 토픽** (v1.18 트리 기준):

| 메시지 | VERSION | Isaac SITL 토픽명 | 쓰는 곳 |
|---|---|---|---|
| `VehicleStatus` | 4 | `/fmu/out/vehicle_status_v4` | `offboard_common.py:115` |
| `VehicleAttitudeSetpoint` | 1 | `/fmu/in/vehicle_attitude_setpoint_v1` | `offboard_common.py:112` |
| `VehicleLocalPosition` | 1 | `/fmu/out/vehicle_local_position_v1` | `offboard_common.py:118` |
| `TrajectorySetpoint` · `VehicleCommand` · `VehicleOdometry` · `VehicleAttitude` · `OffboardControlMode` · `VehicleThrustSetpoint` · `VehicleTorqueSetpoint` · `ActuatorMotors` · `SensorCombined` | 0 | 변화 없음 | — |

> 참고: `online_rl_main.py` 가 Isaac SITL 에서 잘 도는 이유는 **우연히
> MESSAGE_VERSION=0 인 토픽만 쓰기 때문**이다(`online_rl_main.py:347-353`).
> `vehicle_status` 를 안 쓴다. 반면 **필드 스크립트는 위 3개를 전부 쓴다.**
> 그래서 지금 상태로 Isaac SITL 에 붙이면 그 3개가 조용히 안 붙는다.

### 권장 해법: 접미사 자동 감지

**이 저장소에 이미 같은 패턴의 선례가 있다** — `online_rl_main.py:263-290` 의
`_detect_px4_namespace()` 가 ROS 그래프에서 `*/fmu/out/vehicle_odometry` 를
찾아 네임스페이스를 자동 판정한다. 같은 방식으로 버전 접미사도 판정하면
**한 코드가 실기(v1.15.4)와 Isaac SITL(v1.18) 양쪽에서 그대로 돈다.**

구현 스케치 (`offboard_common.py` 에 추가):
```python
def _resolve(self, base):
    """ROS 그래프에서 base 또는 base_v<N> 중 실재하는 이름을 고른다."""
    names = [n for n, _ in self.get_topic_names_and_types()]
    if base in names:
        return base
    for n in names:
        if n.startswith(base + '_v'):
            return n
    return base            # 아직 없으면 원래 이름 (나중에 뜰 수도)
```
주의: 노드 생성 직후에는 그래프가 아직 비어 있을 수 있다.
→ 첫 감지를 몇 초 재시도하거나, `--px4-version {auto,1.15,1.18}` CLI 로
   수동 지정할 수 있게 두는 편이 현장에서 안전하다.

### 조종기 수동조종을 SITL 에 넣는 경로 (셋 중 택1)

1. **QGC Joystick** — 실제 송신기를 USB/동글로 노트북에 붙이고 QGC 의
   Joystick 설정에서 활성화. QGC 가 MAVLink `MANUAL_CONTROL` 로 보낸다.
   모드 스위치 매핑까지 QGC 에서 할 수 있어 **실비행과 가장 비슷하다. 권장.**
2. **`/fmu/in/manual_control_input`** — ROS2 로 직접 `ManualControlSetpoint`
   발행. v1.15.4 yaml 139행, v1.18 yaml 143행 양쪽에 이미 있다.
   사용자의 기존 Isaac 프로젝트가 이 방식이면 그대로 재사용.
3. **QGC 모드 드롭다운** — 조종기 없이 마우스로 Offboard 전환.
   가장 간단하지만 "스위치로 인계" 흐름을 검증하지 못한다.

> 스크립트의 스트림 개방 트리거는 `vehicle_status.nav_state_user_intention`
> 이다(`offboard_common.py:172, 247-249`). 이 필드는 **모드 선택의 출처와
> 무관하게** 채워지므로 위 세 경로 어느 것이든 동작한다.

### 2단계에서 확인할 것

- [ ] 오프보드 진입 순간 기체가 튀지 않는가 (origin 스냅샷이 제대로 되는가)
- [ ] F1: 20초간 제자리를 지키는가
- [ ] F2: doublet 마다 롤/피치가 실제로 흔들리고 **고도를 얼마나 잃는가**
      → `FIELD_PROCEDURE` 4-A 가 실비행으로 재려던 값을 미리 얻는다
- [ ] F3: 속도 setpoint 경로 (`position=nan`, `velocity=값`) — **벤치에서는
      검증 불가한 유일한 경로**(bench 가 자세로 치환하므로)
- [ ] 스위치 OFF 로 인계했을 때 조종기가 정상 동작하는가 (§2-5 증상 재현 여부)

### 참고: Isaac 없이 순수 SITL 로도 된다
```bash
cd ~/PX4-Autopilot && make px4_sitl gz_x500     # gz 설치돼 있음
```
`build/px4_sitl_default/bin/px4` 는 이미 빌드돼 있다(2026-07-30, main=v1.18).
Gazebo 로 먼저 돌려보면 Isaac 통합 전에 스크립트 쪽 문제를 걸러낼 수 있다.

---

## 4. 파일 지도

### `px4_field/` — 실기 데이터 수집
| 파일 | 역할 |
|---|---|
| `offboard_common.py` | 오프보드 노드 공통 골격. 상태기계 WAIT→ENGAGED→DONE, setpoint 게이트 |
| `f1_hover.py` | F1 정지호버 → C_thrust, 결합항 k_norm, 자이로 σ, GPS 속도 σ |
| `f2_doublet.py` | F2 자세 doublet(개루프 여기) → G_i = C_torque_i / I_i |
| `f3_drag.py` | F3 직선왕복(등속) → drag_x, drag_y |
| `check_ulog.py` | ulog 사전검사 (PASS/FAIL, 호버 스로틀 추출) |
| `fit_from_ulog.py` | 계수 최종 적합 |
| `check_ros2_topics.sh` | 필요 토픽 발행/구독 점검 |
| `GROUND_CHECK.txt` | **실내·프로펠러 제거 지상검증 절차** (STEP 1/2/3) |
| `FIELD_PROCEDURE.txt` | **실외 비행 절차** (F1→검사→F2 4-A→검사→F2 4-B→F3) |
| `DDS_TOPICS_SETUP.md` | 토픽 노출 설정 |
| `_test_flow.py` | 오프라인 흐름 검증 13단계 (전부 통과) |

### 설계 원칙 (offboard_common.py 헤더)
1. **setpoint 는 오프보드일 때만 발행.** `/fmu/in/trajectory_setpoint` 는 조종기
   수동비행 FlightTask 와 같은 uORB → 동시 발행 시 충돌 (2026-08-03 현장 실측)
2. 스트림 개방 트리거 = `vehicle_status.nav_state_user_intention == OFFBOARD`
   (스위치를 켜면 setpoint 가 없어 `nav_state` 는 안 바뀌지만 이 필드는 즉시 바뀜)
3. 오프보드 진입 순간 origin(x,y,z)·yaw0 스냅샷 → 이후 전부 이 기준
4. 이륙/착륙은 사람이 한다. 스크립트는 떠 있는 기체를 잠깐 넘겨받을 뿐
5. **1 스크립트 실행 = 1 오프보드 구간.** 살려두면 오프보드 재진입 시 시퀀스가
   t=0 부터 재시작(F2 는 doublet 재발사) + 마커 CSV 가 Ctrl-C 전까지 안 써짐

### 비행 순서 의존성 (바꾸면 안 됨)
```
F1 → check_ulog 로 호버 스로틀 u_hover 추출 → F2 --thrust 에 그 값 입력
                                              → F2 4-A(짧게) → 로그검사 → F2 4-B(본)
                                              → F3
```
F2 는 attitude 모드라 **PX4 가 고도를 안 잡는다**(스크립트가 PD 로 보정).
`--thrust` 가 어긋나면 저고도에서 위험 → F1 을 먼저 뜨는 것이 안전 요건.

### 축 순서 (F2)
롤 N회 전부 → 피치 N회 → 요 N회. **섞으면 안 된다** —
설계행렬 조건수가 치솟아 축별 계수 분리 실패(sim 에서 조건수 1412 전례).

---

## 5. 지금 바로 쓰는 명령어

```bash
# 터미널 세팅 (창마다)
source /opt/ros/humble/setup.bash && source ~/colcon_ws/install/setup.bash

# 실내 지상검증 STEP 2 (가장 중요) — 프로펠러 제거 확인!
cd ~/px4_field && python3 f2_doublet.py --bench --need-alt 0 --n 2 --settle 2 --bench-thrust 0.22

# 창2: setpoint 나가는가
ros2 topic hz /fmu/in/vehicle_attitude_setpoint

# 창3: 모터가 실제 반응하는가 (QGC → Analyze → MAVLink Console)
listener actuator_motors

# 토픽 점검
cd ~/px4_field && ./check_ros2_topics.sh

# FC 가 실제로 뭘 내보내는지
ros2 topic list | grep /fmu/out/ | sort
```

### QoS 주의
- `ros2 topic hz` 는 `qos_profile_sensor_data`(BEST_EFFORT) **하드코딩**.
  `--qos-reliability` 옵션이 **없다** (붙이면 unrecognized argument)
- `ros2 topic echo` 는 Humble 에서 발행자 QoS 를 자동 협상하지만,
  echo 를 먼저 띄우면 reliable 로 잡히므로 `--qos-reliability best_effort` 를
  붙이는 편이 안전

---

## 6. 오늘 변경된 것

| 파일 | 변경 |
|---|---|
| `px4_field/GROUND_CHECK.txt` | 실내 전용 전면 재작성. [A]벤치 한계 [B]ulog 경로 [5]펌웨어 상황 신설 |
| `px4_field/check_ros2_topics.sh` | `sensor_gps` → `vehicle_gps_position`, 누락 원인 (a)이름/(b)펌웨어 구분 안내 |
| `px4_field/DDS_TOPICS_SETUP.md` | 토픽명 정정 + ulog 기본 로깅 확인 결과 |
| `px4_field/f1_hover.py` `f2_doublet.py` `f3_drag.py` | `--bench-thrust` 플래그 (기본 0.10 유지) |
| `px4_field/offboard_common.py` | 진입 로그에 bench 추력 표시 |
| `.gitignore` | `*.ulg` 추가 (실기 로그는 디스크에만) |
| 삭제 | 7/29 실행로그 4, 구 문서 3, 고아 스크립트 3, calib 백업 4 |
