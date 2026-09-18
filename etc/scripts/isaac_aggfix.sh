#!/usr/bin/env bash
# 느린속도(1.0) + acc 4.0 조합에서 aggressive+ws8 안정성/밴드 재확인 (8월 acc 기본 3.0이 병목인지)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 MPC_ACC_HOR=4.0
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all
OUT=results_aggfix; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] aggfix START (scale1.0 acc4.0, aggr+waypoint × ws8)" | tee isaac_aggfix.log
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns aggressive,waypoint \
  --capture-disturbances wind_turbulence:8 \
  --capture-biases 3.052,3.488 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] aggfix END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_aggfix.log
kill_all; touch AGGFIX_DONE
