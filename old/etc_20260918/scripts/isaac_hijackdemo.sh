#!/usr/bin/env bash
# hijackdemo — "북으로 가다 동으로 끌려감": line(북진)+circle+waypoint, 월드 동쪽(90°) 공격, track vs whover3
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
export WORLD_ATK_DIR=90    # 월드 동쪽으로 끌기 (line 은 +N 북진 → 직각 납치)
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_hijackdemo; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] hijackdemo START" | tee isaac_hijackdemo.log
SWEEP_ATK_START=150 SWEEP_ATK_END=215 CAPTURE_POLICIES="track,whover3" \
timeout 5400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns line,circle,waypoint --capture-disturbances none:0 \
  --capture-biases 3.488,4.360 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] hijackdemo END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_hijackdemo.log
kill_all; touch HIJACKDEMO_DONE
