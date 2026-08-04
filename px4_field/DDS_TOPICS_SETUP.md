# uXRCE-DDS 토픽 노출 설정 (dds_topics.yaml)

`Publisher count: 0` = PX4 가 그 토픽을 DDS 로 내보내지 않는다는 뜻.
`ros2 topic list` 에 이름이 보이는 것은 근거가 아니다 — **구독자만 있어도 목록에 뜬다.**

## 1. 필요한 토픽 전체

`online_rl_main.py`(학습/배포) + `px4_field/*`(데이터 수집) 이 쓰는 것 전부.

### PX4 → ROS2  (`publications`)
| 토픽 | 용도 |
|---|---|
| `sensor_combined` | 자이로·가속도. **UKF 관측 z[6:9]**, 자이로 σ |
| `vehicle_gps_position` | GPS 위치·속도. **UKF 관측 z[0:6]**. 타입은 `SensorGps` |
| `vehicle_thrust_setpoint` | u[0]. C_thrust |
| `vehicle_torque_setpoint` | u[1:4]. G, 결합항 k_norm |
| `vehicle_odometry` | 위치·속도 (필드 스크립트) |
| `vehicle_local_position` | 위치·속도 — odometry 있으면 선택 |
| `vehicle_attitude` | 자세. drag 계산, UKF 초기화 |
| `vehicle_status` | nav_state/arming. 오프보드 감지 |

### ROS2 → PX4  (`subscriptions`)
| 토픽 | 용도 |
|---|---|
| `offboard_control_mode` | 오프보드 모드 선언 |
| `trajectory_setpoint` | 위치·속도 명령 |
| `vehicle_attitude_setpoint` | 자세 명령 (F2 doublet) |
| `vehicle_command` | arm/mode 명령 |

## 2. 파일 위치

```
PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml
```

## 3. 추가 방법

**기존 항목의 서식을 그대로 복사하세요.** PX4 버전에 따라 필드 구성이 다릅니다
(메시지 버전 관리가 들어간 최신 버전은 항목이 더 붙습니다).

대략 이런 모양입니다:
```yaml
publications:
  - topic: /fmu/out/vehicle_odometry
    type: px4_msgs::msg::VehicleOdometry

  - topic: /fmu/out/sensor_combined
    type: px4_msgs::msg::SensorCombined
  ...

subscriptions:
  - topic: /fmu/in/vehicle_attitude_setpoint
    type: px4_msgs::msg::VehicleAttitudeSetpoint
  ...
```

⚠ `publications` = **PX4 가 발행** (= `/fmu/out/`)
   `subscriptions` = **PX4 가 구독** (= `/fmu/in/`)
   헷갈리면 반대로 넣게 된다.

## 4. 재빌드 + 업로드

보드 타겟은 QGC → Vehicle Setup → Firmware 에서 확인.

```bash
cd ~/PX4-Autopilot
make <보드타겟>            # 예: px4_fmu-v6c_default, px4_fmu-v5_default
make <보드타겟> upload     # USB 로 연결한 상태에서
```

⚠ 재빌드 전에 현재 파라미터를 QGC 에서 **백업**해두세요 (Tools → Save to file).

## 5. 검증

```bash
./check_ros2_topics.sh
```
전부 ✓ 여야 한다. 발행되는데 `echo` 로 안 보이면 QoS 문제다:
```bash
ros2 topic echo <토픽> --qos-reliability best_effort
```

## 6. 재빌드가 부담스러우면 — 최소 세트

이번 현장 데이터 수집(F1/F2/F3)만 먼저 하려면 이것만 있으면 된다:
```
PUB: vehicle_odometry (또는 vehicle_local_position), vehicle_attitude, vehicle_status
SUB: offboard_control_mode, trajectory_setpoint, vehicle_attitude_setpoint
```
나머지(`sensor_combined`, `vehicle_gps_position`, `thrust/torque_setpoint`)는
**계수 추출을 ulog 로 하기 때문에 DDS 노출이 없어도 된다.**
(전부 PX4 기본 로깅 프로파일에 있음 — `logged_topics.cpp` 의 `add_default_topics()`:
 sensor_combined 123행 / vehicle_gps_position 10Hz 145행 / sensor_gps 1Hz 222행 /
 vehicle_thrust_setpoint·vehicle_torque_setpoint 50Hz 238-239행.
 ⚠ GPS σ 는 1Hz 인 `sensor_gps` 말고 10Hz 인 `vehicle_gps_position` 을 쓸 것)
→ 이번엔 3개만 추가하고, 나머지는 배포 단계에서 한 번에 정리해도 된다.
