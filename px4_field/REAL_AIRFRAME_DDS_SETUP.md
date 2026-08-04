# 실기체 DDS 토픽 추가 — 노트북에서

> 목적: 제어입력 `u`(추력·토크 setpoint)를 ROS2 토픽으로 받는다.
>
> 서두를 필요는 없다 — 계수 추출은 **ulog 경로**라
> (`fit_from_ulog.py:150-155`, 기본 로깅 50Hz `logged_topics.cpp:238-239`)
> 플래시 전에도 실외 비행 F1/F2/F3 는 그대로 가능하다.

---

## 0. 추가할 토픽 — 딱 2개

v1.15.4 기본 yaml 을 실제로 파싱해 확인한 결과(2026-08-04):

| 토픽 | 상태 | 용도 |
|---|---|---|
| `sensor_combined` | ✅ 이미 있음 | **IMU**(자이로+가속도). UKF 관측 z[6:9] |
| `vehicle_gps_position` | ✅ 이미 있음 | UKF 관측 z[0:6] |
| `vehicle_attitude` · `vehicle_odometry` · `vehicle_local_position` · `vehicle_status` | ✅ 이미 있음 | — |
| **`vehicle_thrust_setpoint`** | ★ **추가** | **UKF u[0]** |
| **`vehicle_torque_setpoint`** | ★ **추가** | **UKF u[1:4]** |

**넣지 않는 것**
- `actuator_motors` — 제어 할당기 **출력**이라 UKF 모델에 안 들어간다.
- `vehicle_angular_velocity` — 자이로가 `sensor_combined` 에 이미 있어 중복.

> ⚠ **FLASH 여유가 1% 뿐이다** (2개 추가 후 98.85% = 1943500 B / 1920 KB).
>   여기서 토픽을 더 넣으면 빌드가 깨질 수 있다.

### px4_msgs 는 손대지 않는다
`VehicleThrustSetpoint`/`VehicleTorqueSetpoint` 는 px4_msgs 에 **이미 있다.**
yaml 편집은 기존 메시지 타입을 DDS 목록에 적는 것일 뿐이다.
**`colcon build` 를 다시 돌릴 필요가 없다.**

---

## 1. ★ 첫 단계 — 노트북 PX4 트리의 버전 확인

노트북에 이미 `PX4-Autopilot` 이 있으면 **클론하지 않는다.** 다만 버전이
FC 와 같아야 한다. 먼저 확인:

```bash
cd ~/PX4-Autopilot && git describe --tags && git log -1 --format=%H
```

FC 가 `ver all` 로 보고한 값(2026-08-04 실측):
```
PX4 version : 1.15.4
PX4 git-hash: 99c40407ffd7ac184e2d7b4b293f36f10fe561ef
```

### 결과에 따라 갈린다

**(A) `v1.15.4` / 해시가 `99c40407...` → 그대로 쓴다.** §2 로.

**(B) 다른 버전이다 → 두 선택지**

  **B-1. 같은 저장소에서 워크트리로 v1.15.4 를 꺼낸다 (다운로드 없음, 권장)**
  ```bash
  cd ~/PX4-Autopilot
  git worktree add ~/PX4-v1154 v1.15.4        # 네트워크 미사용
  cd ~/PX4-v1154 && git submodule update --init --recursive
  git describe --tags                          # v1.15.4 확인
  ```
  태그는 이미 로컬 히스토리에 있으므로 받을 게 없다. 기존 체크아웃도
  건드리지 않는다. 지우려면 `git worktree remove ~/PX4-v1154`.

  **B-2. 워크스테이션에서 만든 `.px4` 를 USB 로 옮긴다 (빌드 불필요)**
  이미 빌드해뒀다:
  ```
  ~/PX4-v1154/build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4   (1.8 MB)
  ```
  `.px4` 는 아키텍처 무관한 펌웨어 이미지라 어느 PC 에서 만들든 동일하다.
  이걸 노트북으로 옮기면 **§2~§3 을 건너뛰고 바로 §4 플래시**로 간다.

> **왜 버전을 맞추는가**: v1.16+ 는 `MESSAGE_VERSION != 0` 인 메시지의
> 토픽명에 `_v<N>` 을 붙인다(`utilities.hpp:35`). v1.18 로 올리면
> `vehicle_status` → `vehicle_status_v4` 로 바뀌고, 마이너 3단계 점프라
> 파라미터·캘리브레이션·제어기 기본값도 바뀔 수 있다.
> 같은 v1.15.4 면 **비행 거동이 하나도 안 바뀌고** 토픽 2개만 늘어난다.

---

## 2. yaml 편집

```bash
vi <PX4트리>/src/modules/uxrce_dds_client/dds_topics.yaml
```

6행이 `publications:` 이고 7행이 빈 줄이다. **그 사이에** 삽입:

```yaml
publications:
  # UKF 제어입력 u — 젯슨 실시간 추론용 (2026-08-04 추가)
  #   u[0]   = vehicle_thrust_setpoint.xyz[2]
  #   u[1:4] = vehicle_torque_setpoint.xyz
  - topic: /fmu/out/vehicle_thrust_setpoint
    type: px4_msgs::msg::VehicleThrustSetpoint
  - topic: /fmu/out/vehicle_torque_setpoint
    type: px4_msgs::msg::VehicleTorqueSetpoint

  - topic: /fmu/out/register_ext_component_reply     ← 원래 있던 줄
    type: px4_msgs::msg::RegisterExtComponentReply
```

**들여쓰기 엄수**: `- topic:` = 스페이스 **2칸**, `type:` = **4칸**.
YAML 이라 **탭을 쓰면 빌드가 깨진다.**

### 문법 검증 — 빌드 전 30초로 잡는다
```bash
python3 -c "
import yaml, os
f = os.path.expanduser('~/PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml')
p = [x['topic'] for x in yaml.safe_load(open(f))['publications']]
for t in ['/fmu/out/vehicle_thrust_setpoint', '/fmu/out/vehicle_torque_setpoint']:
    print(('OK  ' if t in p else 'MISS'), t)
print('총 publications:', len(p))
"
```
둘 다 `OK` 여야 한다. `ScannerError` 면 들여쓰기/탭 문제다.

---

## 3. 빌드

```bash
cd <PX4트리> && make px4_fmu-v6c_default
```

첫 빌드 20~60분. 툴체인(`arm-none-eabi-gcc`)이 없으면:
```bash
bash ./Tools/setup/ubuntu.sh --no-sim-tools    # 하드웨어 빌드만
```
끝나면 `dialout` 그룹 적용을 위해 **로그아웃 후 재로그인**.

**반영 확인** — 생성된 헤더의 발행 목록에 들어갔는지:
```bash
grep -c "ORB_ID(vehicle_thrust_setpoint)" \
  <PX4트리>/build/px4_fmu-v6c_default/src/modules/uxrce_dds_client/dds_topics.h
```
`1` 이상이면 반영된 것이다. (v1.15.4 템플릿은 토픽 문자열이 아니라
`ORB_ID(...)` 로 들어가므로 `/fmu/out/...` 로 grep 하면 안 잡힌다)

산출물: `build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4`

---

## 4. 플래시

**① 먼저 파라미터를 백업한다.**
QGC → Parameters → 우상단 **Tools → Save to file**

> 같은 v1.15.4 라 유지되는 것이 정상이지만, 세팅이 끝난 기체이므로 보험은 든다.

**② 플래시**
1. FC 를 USB 로 연결
2. QGC → Vehicle Setup → **Firmware**
3. USB 를 뽑았다 다시 꽂으라는 안내가 뜨면 그렇게 한다
4. 우측 패널 **Advanced settings** → **Custom firmware file...**
5. `px4_fmu-v6c_default.px4` 선택 → 완료 후 재부팅

**③ 파라미터 확인** — 백업과 대조.
특히 `UXRCE_DDS_DOM_ID`, `MPC_THR_HOVER`, `COM_RC_IN_MODE`, 캘리브레이션.

---

## 5. 확인 — 토픽 받아보기

FC 와 ROS2 를 잇는 다리(MicroXRCEAgent)가 있어야 한다.

```bash
# 노트북에서 USB 로 확인할 때
MicroXRCEAgent serial --dev /dev/ttyACM0 -b 921600

# 젯슨에서 (기체 탑재 시)
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600
```

다른 창에서:
```bash
ros2 topic list | grep -E "thrust_setpoint|torque_setpoint"
ros2 topic echo /fmu/out/vehicle_thrust_setpoint --qos-reliability best_effort
```

> 젯슨 쪽 코드는 **아무것도 고치지 않는다.** FC 가 토픽을 내보내기 시작하면
> ROS2 그래프에 그냥 나타난다.

### 플래시 전에 값만 보고 싶다면
QGC → Analyze → **MAVLink Console**:
```
listener vehicle_thrust_setpoint
listener vehicle_torque_setpoint
```
출력 예 (SITL 실측):
```
TOPIC: vehicle_torque_setpoint
    timestamp: 17228000 (0.000000 seconds ago)
    xyz: [-0.00385, 0.00299, 0.00003]
```
uORB 값을 그대로 찍어준다. **ROS2 토픽이 아닐 뿐 값은 완전히 동일하다.**

---

## 6. 터미널 환경 — `colcon build` 는 넣지 말 것

`.bashrc` 에서 **`colcon build` 를 자동 실행하면 안 된다.** 터미널을 열
때마다 워크스페이스를 다시 빌드해 수십 초씩 잡아먹고, 빌드가 실패하면
셸 자체가 이상해진다. 소싱(source)만 하면 된다.

**있으면 제거 (한 줄):**
```bash
sed -i '/colcon build/d' ~/.bashrc && source ~/.bashrc
```

**올바른 세팅 — 3줄. `[ -f ]` 가드가 있어 파일 없는 기계에서도 조용히 넘어간다:**
```bash
printf '\n# ROS2 / PX4\nexport ROS_DOMAIN_ID=0\n[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash\n[ -f ~/colcon_ws/install/setup.bash ] && source ~/colcon_ws/install/setup.bash\n' >> ~/.bashrc && source ~/.bashrc
```

- 순서: 언더레이(`/opt/ros/humble`) **먼저**, 오버레이(`colcon_ws`=px4_msgs) **나중**
- `~/colcon_ws` 는 px4_msgs 를 빌드해둔 워크스페이스다. **노트북에 없으면**
  위 가드가 알아서 건너뛴다. 필요하면 노트북에도 한 번만 만든다:
  ```bash
  mkdir -p ~/colcon_ws/src && cd ~/colcon_ws/src
  git clone https://github.com/PX4/px4_msgs.git
  cd ~/colcon_ws && colcon build --packages-select px4_msgs   # ★ 이 한 번만
  ```
- `ROS_DOMAIN_ID` 는 **터미널 · MicroXRCEAgent 띄운 창 · PX4 파라미터
  `UXRCE_DDS_DOM_ID` 세 곳이 전부 같아야** 한다. 지금 0 으로 잘 되고 있으면
  0 을 유지하는 편이 안전하다.

---

## 7. 되돌리기

QGC → Firmware 에서 **표준 PX4 v1.15.4** 를 다시 플래시하면 원복된다.
필요하면 백업한 `.params` 를 Load from file 로 되돌린다.
어느 시점에도 기체가 못 쓰게 되는 상태는 없다.

---

## 8. 왜 재플래시가 필요한가

`dds_topics.yaml` 은 FC 가 런타임에 읽는 설정 파일이 **아니다.**
빌드할 때 `generate_dds_topics.py` 가 `dds_topics.h` 로 코드 생성해
**펌웨어 바이너리에 컴파일해 박는다.** FC 안에 이 파일은 존재하지 않는다.
그래서 yaml 수정에는 재빌드 + 재플래시가 반드시 따라온다.

## 9. Isaac SITL 은 완전히 별개다

`~/PX4-Autopilot`(워크스테이션, v1.18)은 Isaac 전용이고 FC 와 무관하다.
2026-08-04 에 yaml 추가 + `make px4_sitl_default` 완료했고,
`actuator_motors` 와 `vehicle_angular_velocity` 도 거기엔 넣어뒀다
(프로펠러 뺀 벤치·시뮬레이션 디버깅용). 실기엔 넣지 않는다.
