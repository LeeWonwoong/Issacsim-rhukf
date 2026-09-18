# ActuatorAttack — 제어입력(wrench)에 주입하는 additive 공격 δ
#
#  UAV 다이나믹스 제어입력 = 정규화 토크/추력. ControlAllocator 가 배분 직전 c[0] 에 더한다.
#  companion(젯슨/스윕노드)이 /fmu/in/actuator_attack 으로 발행 → uXRCE-DDS → 이 uORB.
#  RC 경로(manual_control_setpoint aux)와 함께 apply_attack() 에서 소비된다.
#  설계: 저장소 루트 ATTACK_INJECTION.md.
#
#  ⚠ torque/thrust 는 **정규화값 [-1,1]** 이다 (= 제어권한 대비 비율). N·m/N 아님.

uint64 timestamp        # time since system start (microseconds)

bool active             # false 면 δ=0 (게이트)

float32[3] torque       # 정규화 토크 바이어스 [roll, pitch, yaw], 각 [-1,1]
float32 thrust          # 정규화 추력 바이어스 (body-z), [-1,1]

# TOPICS actuator_attack
