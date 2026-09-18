#!/usr/bin/env bash
# calibcap — offline G/Q/R 재생용 (z,u) 스트림 캡처. SGD 완료 후 자동, 끝나면 AdamMSE 이어감.
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f TRAIN0909_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_calibcap; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] calibcap START" | tee -a isaac_calibcap.log
SWEEP_ATK_START=180 SWEEP_ATK_END=245 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,scurve,aggressive --capture-disturbances none:0 \
  --capture-biases 0,3.488 --episodes 3 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] calibcap END zu=$(ls -la $OUT/zu_log.npz 2>/dev/null|awk '{print $5}')" | tee -a isaac_calibcap.log
kill_all; touch CALIBCAP_DONE
