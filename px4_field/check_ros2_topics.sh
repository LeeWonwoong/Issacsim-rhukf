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
"/fmu/out/sensor_gps|GPS 위치·속도 (UKF 관측 z[0:6])|PUB|필수"
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
  echo "✗ ${missing}개 누락 — dds_topics.yaml 에 추가하고 PX4 재빌드 필요"
  echo "  * vehicle_odometry 와 vehicle_local_position 은 둘 중 하나만 있어도 됩니다"
else
  echo "✓ 전부 확인됨"
fi
echo
echo "참고: 발행되고 있는데 echo 로 안 보이면 QoS 때문입니다 —"
echo "      ros2 topic echo <토픽> --qos-reliability best_effort"
