#!/usr/bin/env bash
# 구(arm0) vs 신(arm0.02) 밴드: 공격only(ws0)+공격+바람(ws6,12), 전패턴 × δ{0.4,0.6,0.8}. 시간창.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 SWEEP_ATTACK_TYPE=tilt EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
LOG=band_oldnew.log
echo "════ 밴드 구/신 시작 $(date '+%H:%M:%S') ════" | tee $LOG
run_arm () {  # $1=arm $2=태그
  export WIND_MOMENT_ARM=$1
  OUT=results_band_$2
  echo "[$(date '+%H:%M:%S')] arm=$1 ($2) START" | tee -a $LOG
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  sleep 5
  rm -rf $OUT; mkdir -p $OUT
  timeout 18000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns circle,figure8,waypoint,aggressive \
    --capture-disturbances none:0,wind_turbulence:6,wind_turbulence:12 \
    --capture-biases 1.744,2.616,3.488 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] arm=$1 DONE rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0)" | tee -a $LOG
}
run_arm 0 old
run_arm 0.02 new
pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch BAND_OLDNEW_DONE
