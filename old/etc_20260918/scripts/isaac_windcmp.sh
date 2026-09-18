#!/usr/bin/env bash
# windcmp — 실공간 궤적(r2.8·ω0.45)에서 바람 구성 비교 (GUI)
#   run1: arm0.05 × ws{8,10}   run2: arm0.07 × ws{6,8}
#   δ{0.75,0.8,0.85} × {track,dhover3} × 3패턴 × 3ep = 각 108(2ws면 108), 판정: track死∧dhover3生 창
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 MPC_ACC_HOR=4.0 DISPLAY=:1
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3"
LOG=isaac_windcmp.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
run_one(){ # $1=arm $2=ws목록 $3=태그
  kill_all
  export WIND_MOMENT_ARM=$1
  OUT=results_windcmp_$3; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] windcmp $3 (arm=$1 ws=$2) START" | tee -a $LOG
  timeout 14400 python3 -u online_rl_main.py --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,figure8,aggressive \
    --capture-disturbances $2 \
    --capture-biases 3.270,3.488,3.706 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] windcmp $3 END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
}
run_one 0.05 "wind_turbulence:8,wind_turbulence:10" a05
run_one 0.07 "wind_turbulence:6,wind_turbulence:8"  a07
kill_all; touch WINDCMP_DONE
