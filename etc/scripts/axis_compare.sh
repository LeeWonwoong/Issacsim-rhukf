#!/usr/bin/env bash
# 축모드 비교: 단일축(순수 roll) vs balanced(roll=pitch). 같은 magnitude δ. arm0.02 Q2e-3 시간창.
# 전패턴 × δ{0.3-0.8} × ws{0,6,9,12}. 방향이 밴드·NIS에 영향 주나 확인.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
LOG=axis_compare.log
echo "════ 축모드 비교 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG

run_axis () {  # $1=태그 $2=yaw_ratio $3=biases(δ×환산)
  local TAG=$1 YR=$2 BIAS=$3 OUT=results_axis_$1
  echo "[$(date '+%H:%M:%S')] $TAG (yaw_ratio=$YR) START" | tee -a $LOG
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  sleep 5
  rm -rf $OUT; mkdir -p $OUT
  timeout 43200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --torque-yaw-ratio $YR --capture-mode hijack \
    --capture-patterns circle,figure8,waypoint,aggressive \
    --capture-disturbances none:0,wind_turbulence:4,wind_turbulence:5,wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_turbulence:9,wind_turbulence:10,wind_turbulence:11,wind_turbulence:12 \
    --capture-biases $BIAS --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] $TAG DONE rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0)" | tee -a $LOG
}

# 단일축(roll): yaw_ratio=0 → (value,0). magnitude=value/A=δ → value=δ×4.36
run_axis single 0.0 1.308,1.744,2.18,2.616,3.052,3.488
# balanced(roll=pitch): yaw_ratio=1 → (value,value). magnitude=√2·value/A=δ → value=δ×3.083
run_axis balanced 1.0 0.9249,1.2332,1.5415,1.8497,2.1580,2.4663

pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch AXIS_COMPARE_DONE
