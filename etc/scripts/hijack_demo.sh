#!/usr/bin/env bash
# B 유인공격 데모 — 목표점 (N=7, E=6) 으로 지속 유도. 무대응(track) vs 대응(whover2).
#   burst 시간구조는 학습(A)과 동일, 방향만 closed-loop. waypoint 임무 중 hijack.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
ki(){ for p in $(pgrep -f "python.sh run_sim"); do kill -9 $p 2>/dev/null; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null; done; sleep 5; }
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 EP_MAX_STEPS=400
export SWEEP_ATK_START=150 SWEEP_ATK_END=350 SWEEP_ATTACK_TYPE=tilt
export HIJACK_TARGET="6,5" HIJACK_DELTA=0.5
ki; rm -rf results_hijack; mkdir -p results_hijack
CAPTURE_POLICIES=track,whover2 timeout 2400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns waypoint --capture-disturbances none:0 \
  --capture-biases 2.180 --episodes 1 --speed 1 --log-zu --outdir results_hijack > results_hijack/run.log 2>&1
echo rc=$?
ki; touch HIJACK_DEMO_DONE
