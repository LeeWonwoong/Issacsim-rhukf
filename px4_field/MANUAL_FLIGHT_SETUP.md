# Isaac Sim 조종기 수동비행 — 2단계 리허설 설정

> 목적: **실외에 나가기 전에** "조종기로 이륙 → 오프보드 스위치 → 스크립트가
> 시퀀스 비행 → 스위치 OFF 인계" 전 과정을 가상에서 그대로 재현한다.
> 프로펠러 뺀 벤치(1단계)가 답하지 못하는 **비행 거동**을 여기서 본다.

출처: `AdaptiveFIR_SAC/datagen/MANUAL_FLIGHT.md` 의 조종기 경로를 이 프로젝트로
이식. **조종기 관련 코드는 한 줄도 없다** — 전부 QGC 설정 + PX4 파라미터다.

```
사람 → 조종기(USB HID) → QGroundControl Joystick
     → MAVLink MANUAL_CONTROL → PX4 SITL → Isaac(run_sim.py)
                                    ↑
                       px4_field/f1|f2|f3 (오프보드 시퀀스)
```

---

## 1. 입력장치 — 두 경로 모두 QGC Joystick 으로 통일

### A. 실제 조종기 (RadioMaster / FrSky / Jumper 등) — 권장
1. 조종기를 USB 로 PC 에 연결 → 조종기 화면에서 **"USB Joystick (HID)"** 선택
   (EdgeTX/OpenTX 는 연결 시 모드 선택 팝업이 뜬다)
2. QGC → **Vehicle Setup → Joystick → Enable joystick input** → **Calibrate**
3. 시뮬레이터 동글(트레이너 포트→USB)도 OS 에는 HID 조이스틱으로 잡히므로 동일

### B. USB 게임패드 (Xbox/PS/로지텍)
1. `ls /dev/input/js*` 로 인식 확인 (`jstest-gtk` 로 축 확인 가능)
2. 위 A-2 와 동일

> 어느 쪽이든 **코드 경로가 완전히 동일**하다. QGC 가 joystick 입력을
> MAVLink `MANUAL_CONTROL` 로 PX4 에 전달한다.

### PX4 파라미터
```
COM_RC_IN_MODE = 1      # Joystick/무RC. SITL 기본값이 보통 1
```
아니면 QGC → Parameters 에서 변경.

---

## 2. ★ 오프보드 스위치 — 이 프로젝트에만 필요한 조각

`AdaptiveFIR_SAC` 는 Position 모드로만 날았으므로 모드 전환 버튼이 없었다.
**우리는 "조종기 스위치로 오프보드 ON/OFF" 자체가 검증 대상**이다.

QGC → **Vehicle Setup → Joystick → Button Assignment** 탭에서:

| 버튼 | 할당 | 용도 |
|---|---|---|
| 아무 버튼 1 | **Offboard** | ★ 오프보드 진입/이탈. 실기 조종기의 그 스위치 |
| 아무 버튼 2 | **Position** | 인계 복귀 |
| 아무 버튼 3 | Arm / Disarm | 편의 |

**실제 조종기의 2단/3단 스위치**를 쓰면 실비행과 감각이 같아진다
(EdgeTX 에서 스위치를 채널에 매핑 → QGC 가 버튼으로 인식).

> ⚠ QGC 버튼은 **누르는 순간 모드 전환**이라 토글이 아니다. 실기 스위치처럼
> "올려두면 오프보드 유지"가 아니라 "누르면 오프보드로 감"이다.
> 인계하려면 Position 버튼을 눌러야 한다. 이 차이만 감안하면 검증 목적에는
> 충분하다 — 스크립트가 보는 것은 `nav_state` 와
> `nav_state_user_intention` 이지 스위치 물리 상태가 아니기 때문이다
> (`offboard_common.py:172, 247-249`).

---

## 3. 실행

```bash
bash px4_field/launch_isaac_manual.sh f1              # F1 호버
bash px4_field/launch_isaac_manual.sh f2 1.0 --n 3    # F2 doublet 3회
bash px4_field/launch_isaac_manual.sh f3              # F3 직선왕복
```

⚠ **speed 는 1.0 고정 권장.** RTF>1 이면 사람 반응이 시뮬레이션 시간 기준으로
느려져 조종성이 나빠진다.

런처가 하는 일: 잔재 프로세스 검사 → MicroXRCEAgent 기동 → `run_sim.py` 백그라운드
→ `/gt/odometry` 로 준비 판정 → QGC 안내 출력 → 시퀀스 스크립트 포그라운드 실행.
Ctrl-C 하면 엔진과 고아 px4 까지 정리한다.

### ★ `--bench` 를 붙이지 않는다
SITL 은 GPS·위치 추정이 유효하므로 **실비행과 동일한 경로**(위치·속도 setpoint)를
그대로 검증한다. 벤치에서 검증 불가했던 **F3 의 속도 setpoint 경로가 여기서
처음 확인된다.**

---

## 4. 확인할 것

- [ ] 오프보드 진입 순간 기체가 튀지 않는가 (origin 스냅샷이 제대로 되는가)
- [ ] 스위치 전에는 `ros2 topic hz /fmu/in/trajectory_setpoint` 가 조용한가
      (= 조종기와 안 싸움. `offboard_common.py` 설계원칙 1)
- [ ] **F1**: 20초간 제자리를 지키는가
- [ ] **F2**: doublet 마다 롤/피치가 실제로 흔들리고 **고도를 얼마나 잃는가**
      → `FIELD_PROCEDURE` 4-A 가 실비행으로 재려던 값을 미리 얻는다
- [ ] **F3**: `position=nan`, `velocity=값` 으로 나가는가 (벤치 미검증 경로)
- [ ] 축 순서 roll → pitch → yaw (F2)
- [ ] 인계 후 조종기가 정상 동작하는가 (`GROUND_CHECK.txt` §4 의 왼쪽 스틱 증상)

### 제어입력 관측
```bash
ros2 topic echo /fmu/out/vehicle_thrust_setpoint     # u[0]
ros2 topic echo /fmu/out/vehicle_torque_setpoint     # u[1:4]
ros2 topic echo /fmu/out/actuator_motors             # 모터 4채널
```

⚠ 위 셋은 `dds_topics.yaml` 의 `publications` 에 있어야 보인다.
**yaml 은 런타임 설정이 아니라 빌드 때 `dds_topics.h` 로 코드 생성되어
바이너리에 박힌다.** 바꿨으면 반드시 재빌드:

```bash
cd ~/PX4-Autopilot && make px4_sitl_default
```

`~/PX4-Autopilot`(v1.18) 의 yaml 에는 넷 다 추가돼 있다(2026-08-04).
**이 빌드는 SITL 전용이라 실기 FC 와 무관하다 — FC 는 손대지 않는다.**

---

## 5. 토픽 이름 주의 — 자동 처리됨

실기 FC(v1.15.4)와 Isaac SITL(v1.18)은 같은 토픽을 다른 이름으로 낸다.
PX4 v1.16+ 가 `MESSAGE_VERSION != 0` 인 메시지에 `_v<N>` 을 붙이기 때문
(`utilities.hpp:35`).

| 메시지 | FC v1.15.4 | Isaac SITL v1.18 |
|---|---|---|
| `VehicleStatus` | `/fmu/out/vehicle_status` | `/fmu/out/vehicle_status_v4` |
| `VehicleAttitudeSetpoint` | `…/vehicle_attitude_setpoint` | `…_v1` |
| `VehicleLocalPosition` | `…/vehicle_local_position` | `…_v1` |

**`offboard_common.py` 의 `_make_resolver()` 가 ROS 그래프를 보고 자동으로
맞춘다.** 한 코드가 양쪽에서 그대로 돈다. 실행 로그에 `토픽 해석: A → B` 로
찍히니 확인할 것.

그래프가 비어 있으면(엔진 미기동) 접미사 없는 이름으로 폴백하고 경고한다
→ **런처가 `/gt/odometry` 를 기다린 뒤 스크립트를 띄우므로 정상 경로에서는
문제되지 않는다.**

---

## 6. 실기 FC 에서 제어입력을 보려면 (재플래시 없이)

FC 는 v1.15.4 이고 그 yaml 에는 thrust/torque 가 `/fmu/in` 쪽에만 있다.
`/fmu/out` 으로 내보내려면 재빌드·재플래시가 필요한데 — **그럴 필요 없다.**

QGC → Analyze → **MAVLink Console** 에서:
```
listener vehicle_thrust_setpoint
listener vehicle_torque_setpoint
listener actuator_motors
```
uORB 값을 그대로 찍어준다. 조종기 조종 중에도, 오프보드 중에도 실시간으로.
**ROS2 토픽이 아닐 뿐 값은 완전히 동일하다.**

그리고 계수 추출은 어차피 ulog 경로다(`fit_from_ulog.py:150-155`).
thrust/torque 는 PX4 기본 로깅에 50Hz 로 이미 들어있다
(`logged_topics.cpp:238-239`). **실외 F1/F2/F3 는 지금 펌웨어로 그대로 진행 가능.**

ROS2 토픽이 반드시 필요한 시점은 **젯슨에서 UKF 를 실시간으로 돌릴 때(배포)**
하나뿐이고, 그때 v1.15.4 트리에서 토픽만 추가해 빌드하면 버전 점프가 아니라
**파라미터도 기체 세팅도 그대로 유지된다.**


③ 3줄 새로 추가

cat >> ~/.bashrc <<'EOF'

# ROS2 / PX4
export ROS_DOMAIN_ID=6
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
EOF

EOF 까지 통째로 복사해서 붙여넣고 엔터. (내가 아까 준 printf 는 이스케이프가 깨져서 안 됐던 거야. 이건 테스트했어.)

④ 적용

source ~/.bashrc
echo $ROS_DOMAIN_ID

6 이 나오면 끝.

---
노트북에 ~/colcon_ws 가 없으면 ③의 마지막 줄에서 "그런 파일이 없습니다" 가 떠. 그럼 그 줄만 빼:

sed -i '/colcon_ws/d' ~/.bashrc && source ~/.bashrc
