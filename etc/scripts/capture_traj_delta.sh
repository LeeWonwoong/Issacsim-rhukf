#!/usr/bin/env bash
# δ 스윕 공격대응 궤적 캡처 (2026-08-27): δ{0.1..0.8} × {track,whover2} × {waypoint,scurve}
#   공격창 15~25s(10s), 무풍, speed 1, 코너감속 ON. maneuver A/B 완료 후 자동 시작.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=maneuver_ab.log
echo "[$(date '+%H:%M:%S')] traj_delta: A/B 완료 대기" | tee -a $LOG
while [ ! -f MANEUVER_AB_DONE ]; do sleep 60; done
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5
export EP_MAX_STEPS=400 SWEEP_ATK_START=150 SWEEP_ATK_END=250 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES=track,whover2
echo "[$(date '+%H:%M:%S')] traj_delta START (20셀)" | tee -a $LOG
kill_isaac; rm -rf results_traj_delta; mkdir -p results_traj_delta
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns waypoint,scurve \
  --capture-disturbances none:0 \
  --capture-biases 0.436,0.872,1.744,2.616,3.488 --episodes 1 --speed 1 --log-zu \
  --outdir results_traj_delta > results_traj_delta/run.log 2>&1
echo "[$(date '+%H:%M:%S')] traj_delta DONE rc=$?" | tee -a $LOG
kill_isaac
touch TRAJ_DELTA_DONE
