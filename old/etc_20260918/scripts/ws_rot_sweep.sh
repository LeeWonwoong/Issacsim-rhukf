#!/usr/bin/env bash
# ws 스윕 (arm=0.04 고정, 회전 모멘트 ws²비례): waypoint, ws{0,6,9,12} × tilt δ{0,0.6}.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.04 SWEEP_ATTACK_TYPE=tilt SWEEP_ATK_START=80
OUT=results_ws_rot; LOG=ws_rot.log
echo "════ ws 스윕(arm0.04) 시작 $(date '+%H:%M:%S') ════" | tee $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for pid in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
sleep 5; rm -rf $OUT; mkdir -p $OUT
timeout 12000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns waypoint \
  --capture-disturbances none:0,wind_turbulence:6,wind_turbulence:9,wind_turbulence:12 \
  --capture-biases 0.0,2.616 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "════ 종료 $(date '+%H:%M:%S') rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0) ════" | tee -a $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for pid in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
touch WS_ROT_DONE
