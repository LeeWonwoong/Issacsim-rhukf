# 실기체 DDS 토픽 추가 — 노트북에서 빌드·플래시

> 목적: 제어입력 `u`(추력·토크 setpoint)를 ROS2 로 내보내게 한다.
> 젯슨에서 UKF 를 **실시간**으로 돌릴 때 필요하다.
>
> ⚠ 계수 추출(F1/F2/F3)에는 **필요 없다** — 그건 ulog 경로다
> (`fit_from_ulog.py:150-155`, 두 토픽 모두 PX4 기본 로깅에 50Hz 로 이미 있음).
> 즉 **이 작업 없이도 실외 비행은 그대로 진행 가능**하다.

---

## 0. 기계 역할 — 셋이 다르다

```
 ┌──────────────────────────────┐
 │ ① 노트북 (Ubuntu)            │  ★ 여기서 빌드 + 플래시 + QGC
 │    현장에 들고 가는 기계      │     펌웨어 .px4 를 만드는 곳
 └──────────┬───────────────────┘
            │ USB (플래시할 때만)
 ┌──────────▼───────────────────┐
 │ ② FC (Pixhawk FMU-v6C)       │  PX4 v1.15.4
 │                              │  ★ dds_topics.yaml 이 박히는 유일한 곳
 └──────────┬───────────────────┘
            │ 시리얼/USB (비행 중 상시)
 ┌──────────▼───────────────────┐
 │ ③ 젯슨 (드론 탑재)           │  ROS2 + MicroXRCEAgent + px4_field/*
 │                              │  ★ 펌웨어 작업 없음. 아무것도 안 고쳐도
 │                              │    새 토픽이 그냥 나타난다
 └──────────────────────────────┘
```

**젯슨에서는 빌드하지 말 것** — ARM + 적은 RAM 이라 몇 시간 걸린다.
**FC 는 플래시 대상일 뿐** — 빌드하는 곳이 아니다.

`.px4` 는 아키텍처 무관한 펌웨어 이미지라 어느 PC 에서 만들든 동일하다.
노트북 툴체인이 말썽이면 다른 PC 에서 빌드해 `.px4` 파일(약 2MB) 하나만
USB 로 옮겨도 된다.

---

## 1. 추가할 토픽 — 딱 2개

v1.15.4 기본 yaml 을 실제로 열어 확인한 결과(2026-08-04):

| 토픽 | 상태 | 용도 |
|---|---|---|
| `sensor_combined` | ✅ 이미 있음 | **IMU**(자이로+가속도). UKF 관측 z[6:9] |
| `vehicle_gps_position` | ✅ 이미 있음 | UKF 관측 z[0:6] |
| `vehicle_attitude` | ✅ 이미 있음 | 자세 |
| `vehicle_odometry` · `vehicle_local_position` | ✅ 이미 있음 | 위치·속도 |
| `vehicle_status` | ✅ 이미 있음 | nav_state (오프보드 감지) |
| **`vehicle_thrust_setpoint`** | ★ **추가** | **UKF u[0]** |
| **`vehicle_torque_setpoint`** | ★ **추가** | **UKF u[1:4]** |

제외한 것:
- `actuator_motors` — 제어 할당기 **출력**이라 UKF 모델에 안 들어간다.
  프로펠러 뺀 벤치에서 모터 분배를 눈으로 볼 때만 쓸모 있다. 실기에선 불필요.
- `vehicle_angular_velocity` — 자이로는 `sensor_combined` 에 이미 들어 있어
  **사실상 중복**이다. 넣어도 되지만 필요는 없다.

---

## 2. 노트북 준비 (1회)

```bash
# 필요 디스크 ~10GB
sudo apt update && sudo apt install -y git

# FC 와 같은 버전을 클론 (서브모듈 34개 포함, 10~20분)
git clone --branch v1.15.4 --recursive \
    https://github.com/PX4/PX4-Autopilot.git ~/PX4-v1154

# PX4 툴체인 (arm-none-eabi-gcc 등).
# --no-sim-tools = 하드웨어 빌드만. Gazebo/JSBSim 등을 건너뛰어 훨씬 가볍다.
cd ~/PX4-v1154 && bash ./Tools/setup/ubuntu.sh --no-sim-tools
```

★ `ubuntu.sh` 가 사용자를 `dialout` 그룹에 넣는다 → **로그아웃 후 재로그인.**
   (USB 시리얼 접근 권한. 안 하면 나중에 막힐 수 있다)

QGroundControl 은 이미 노트북에 있다(FIELD_PROCEDURE [0] 체크리스트).

**툴체인 확인:**
```bash
arm-none-eabi-gcc --version | head -1     # 버전이 찍히면 OK
```

---

## 3. yaml 편집

```bash
vi ~/PX4-v1154/src/modules/uxrce_dds_client/dds_topics.yaml
```

6행이 `publications:` 이고 7행이 빈 줄이다. **그 사이에** 삽입:

```yaml
publications:
  # UKF 제어입력 u — 젯슨 실시간 추론용 (2026-08-04 추가)
  - topic: /fmu/out/vehicle_thrust_setpoint
    type: px4_msgs::msg::VehicleThrustSetpoint
  - topic: /fmu/out/vehicle_torque_setpoint
    type: px4_msgs::msg::VehicleTorqueSetpoint

  - topic: /fmu/out/register_ext_component_reply     ← 원래 있던 줄
    type: px4_msgs::msg::RegisterExtComponentReply
```

**들여쓰기 엄수**: `- topic:` 은 스페이스 **2칸**, `type:` 은 **4칸**.
YAML 이라 **탭을 쓰면 빌드가 깨진다.**

### 문법 검증 — 빌드 전 30초로 잡아낸다
```bash
python3 -c "
import yaml
d = yaml.safe_load(open('$HOME/PX4-v1154/src/modules/uxrce_dds_client/dds_topics.yaml'))
p = [x['topic'] for x in d['publications']]
for t in ['/fmu/out/vehicle_thrust_setpoint', '/fmu/out/vehicle_torque_setpoint']:
    print(('OK  ' if t in p else 'MISS'), t)
print('총 publications:', len(p))
"
```
둘 다 `OK` 여야 한다. `yaml.scanner.ScannerError` 가 나면 들여쓰기/탭 문제다.

---

## 4. 빌드

```bash
cd ~/PX4-v1154 && make px4_fmu-v6c_default
```

첫 빌드 20~60분(노트북 코어 수에 따라). 결과물:
```
~/PX4-v1154/build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4
```

**반영 확인** — 생성된 헤더에 실제로 들어갔는지:
```bash
grep -c '"/fmu/out/vehicle_thrust_setpoint"' \
  ~/PX4-v1154/build/px4_fmu-v6c_default/src/modules/uxrce_dds_client/dds_topics.h
```
`1` 이상이면 반영된 것이다.

---

## 5. 플래시

**① 먼저 파라미터를 백업한다.**
QGC → Parameters → 우상단 **Tools → Save to file** → `params_before_20260804.params`

> 같은 v1.15.4 이므로 파라미터는 유지되는 것이 정상이다. 하지만 세팅이
> 끝난 기체이므로 보험은 든다.

**② 플래시**
1. FC 를 노트북에 USB 로 연결
2. QGC → Vehicle Setup → **Firmware**
3. USB 를 뽑았다 다시 꽂으라는 안내가 뜨면 그렇게 한다
4. 우측 패널 **Advanced settings** → 드롭다운 → **Custom firmware file...**
5. `~/PX4-v1154/build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4` 선택
6. 완료 후 FC 재부팅

**③ 파라미터 확인** — 백업 파일과 대조. 다르면 QGC → Tools → Load from file.

특히 확인: `UXRCE_DDS_DOM_ID`, `MPC_THR_HOVER`, `COM_RC_IN_MODE`,
그리고 기체 캘리브레이션(가속도계/자이로/나침반).

---

## 6. 확인 — 젯슨에서

```bash
# 젯슨에서 FC 와 시리얼 연결
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600

# 다른 창에서
ros2 topic list | grep -E "thrust_setpoint|torque_setpoint"
cd ~/px4_field && ./check_ros2_topics.sh      # 전체 점검
```

> 젯슨 쪽은 **아무것도 고치지 않는다.** FC 펌웨어가 토픽을 내보내기 시작하면
> ROS2 그래프에 그냥 나타난다.

---

## 7. 되돌리기

QGC → Firmware 에서 **표준 PX4 v1.15.4** 를 다시 플래시하면 원복된다.
필요하면 백업한 `.params` 를 Load from file 로 되돌린다.
어느 시점에도 기체가 못 쓰게 되는 상태는 없다.

---

## 8. 배경 — 왜 이렇게 하는가

### 왜 재플래시가 반드시 필요한가
`dds_topics.yaml` 은 FC 가 런타임에 읽는 설정 파일이 **아니다.**
빌드할 때 `generate_dds_topics.py` 가 `dds_topics.h` 로 코드 생성해
**펌웨어 바이너리에 컴파일해 박는다.** FC 안에 이 파일은 존재하지 않는다.

**재플래시 없이 값만 보려면** QGC → Analyze → **MAVLink Console**:
```
listener vehicle_thrust_setpoint
listener vehicle_torque_setpoint
```
uORB 값을 그대로 찍어준다. 조종기 조종 중에도, 오프보드 중에도 실시간으로.
**ROS2 토픽이 아닐 뿐 값은 완전히 동일하다.**

### 왜 버전을 v1.15.4 로 맞추는가
FC 에 올라간 펌웨어가 v1.15.4 다(빌드일 2025-12-10, `ver all` 실측).
같은 버전이므로 **버전 점프가 아니다** — 파라미터·기체 세팅·토픽 이름이
전부 유지되고, 늘어나는 것은 DDS 토픽 2개뿐이다.

v1.16+ 는 `MESSAGE_VERSION != 0` 인 메시지의 토픽명에 `_v<N>` 을 붙인다
(`utilities.hpp:35`, `dds_topics.h.em:83`). v1.18 로 올리면
`vehicle_status` → `vehicle_status_v4` 등으로 이름이 바뀐다.
**그래서 버전을 올리지 않는다.**

(`offboard_common.py` 의 `_make_resolver()` 가 양쪽을 자동으로 흡수하므로
 스크립트 자체는 어느 버전이든 돌지만, 굳이 변수를 늘릴 이유가 없다.)

### Isaac SITL 은 별개다
`~/PX4-Autopilot`(이 워크스테이션, v1.18)은 Isaac 전용이고 **FC 와 무관**하다.
거기엔 `actuator_motors` 와 `vehicle_angular_velocity` 도 넣어뒀다 —
프로펠러 뺀 벤치·시뮬레이션 디버깅용이라 실기엔 안 넣는다.
