#!/usr/bin/env bash
# 시간창 grid: clean→바람→겹침→공격→복귀. 전패턴 × ws{4,6,8,10,12} × δ{0.4,0.6,0.75,0.8}. arm0.04.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.04 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
OUT=results_timewin; LOG=timewin.log
echo "════ 시간창 grid 시작 $(date '+%H:%M:%S') (clean이륙→바람100-300→공격180-380) ════" | tee $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for pid in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
sleep 5; rm -rf $OUT; mkdir -p $OUT
timeout 36000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns circle,figure8,waypoint,aggressive \
  --capture-disturbances wind_turbulence:4,wind_turbulence:6,wind_turbulence:8,wind_turbulence:10,wind_turbulence:12 \
  --capture-biases 1.744,2.616,3.27,3.488 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "════ 종료 $(date '+%H:%M:%S') rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0) ════" | tee -a $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for pid in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
touch TIMEWIN_DONE
