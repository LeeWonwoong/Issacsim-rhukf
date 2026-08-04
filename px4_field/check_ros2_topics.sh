#!/usr/bin/env bash
# check_ros2_topics.sh — 실기 운용에 필요한 PX4 토픽이 실제로 발행/구독되는지 점검
#
#   ★ ros2 topic list 에 이름이 보이는 것은 근거가 안 된다.
#     구독자만 있어도 목록에 뜬다. 판정은 Publisher/Subscription count 다.
#
# 사용:  ./check_ros2_topics.sh
set -u

# topic|용도|필요한쪽 (PUB = PX4가 발행 /fmu/out,  SUB = PX4가 구독 /fmu/in)
ROWS=(
"/fmu/out/sensor_combined|자이로·가속도 (UKF 관측 z[6:9], 자이로 σ)|PUB|필수"
"/fmu/out/vehicle_gps_position|GPS 위치·속도 (UKF 관측 z[0:6]). 타입=SensorGps|PUB|필수"
"/fmu/out/vehicle_thrust_setpoint|u[0] 추력 (C_thrust)|PUB|필수"
"/fmu/out/vehicle_torque_setpoint|u[1:4] 토크 (G, k_norm)|PUB|필수"
"/fmu/out/vehicle_odometry|위치·속도 (필드 스크립트)|PUB|필수*"
"/fmu/out/vehicle_local_position|위치·속도 (odometry 로 대체 가능)|PUB|택1*"
"/fmu/out/vehicle_attitude|자세 (drag 계산·UKF 초기화)|PUB|필수"
"/fmu/out/vehicle_status|nav_state/arming (오프보드 감지)|PUB|필수"
"/fmu/in/offboard_control_mode|오프보드 모드 선언|SUB|필수"
"/fmu/in/trajectory_setpoint|위치·속도 명령|SUB|필수"
"/fmu/in/vehicle_attitude_setpoint|자세 명령 (F2 doublet)|SUB|필수"
"/fmu/in/vehicle_command|arm/mode 명령|SUB|권장"
)

printf "%-42s %-6s %-8s %s\n" "토픽" "필요" "개수" "용도"
printf '%.0s─' {1..110}; echo
missing=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r topic desc side need <<<"$row"
  info=$(ros2 topic info "$topic" 2>/dev/null)
  if [ -z "$info" ]; then
    n="-"; mark="✗"
  else
    if [ "$side" = "PUB" ]; then
      n=$(echo "$info" | grep -oP 'Publisher count: \K[0-9]+' | head -1)
    else
      n=$(echo "$info" | grep -oP 'Subscription count: \K[0-9]+' | head -1)
    fi
    n=${n:-0}
    if [ "$n" -gt 0 ]; then mark="✓"; else mark="✗"; fi
  fi
  [ "$mark" = "✗" ] && missing=$((missing+1))
  printf "%-42s %-6s %s %-6s %s\n" "$topic" "$need" "$mark" "$n" "$desc"
done
echo
if [ $missing -gt 0 ]; then
  echo "✗ ${missing}개 누락"
  echo "  * vehicle_odometry 와 vehicle_local_position 은 둘 중 하나만 있어도 됩니다"
  echo
  echo "  누락 원인은 둘 중 하나입니다 — 먼저 FC 가 실제로 뭘 내보내는지 보세요:"
  echo "      ros2 topic list | grep /fmu/out/ | sort"
  echo
  echo "  (a) 이름이 다름   예) sensor_gps 는 Isaac Sim 이름. 실기는 vehicle_gps_position"
  echo "                        (타입은 똑같이 SensorGps, 기본 yaml 에 이미 있음)"
  echo "                    예) 끝에 _v1 같은 접미사가 붙어 있으면 PX4 메시지 버저닝 문제"
  echo "  (b) 펌웨어 불일치 FC 에 올라간 빌드의 dds_topics.yaml 이 소스 트리와 다름."
  echo "                    vehicle_thrust_setpoint / vehicle_torque_setpoint 는"
  echo "                    PX4 기본 yaml 7·9행에 이미 /fmu/out 으로 등록돼 있고,"
  echo "                    uxrce_dds_client 는 세션이 열릴 때 시동 여부와 무관하게"
  echo "                    yaml 전 항목의 data writer 를 만듭니다(dds_topics.h.em:100)."
  echo "                    → 그래도 0 이면 그 yaml 로 빌드된 펌웨어가 아닙니다. 재빌드·플래시."
else
  echo "✓ 전부 확인됨"
fi
echo
echo "참고: 발행되고 있는데 echo 로 안 보이면 QoS 때문입니다 —"
echo "      ros2 topic echo <토픽> --qos-reliability best_effort"
