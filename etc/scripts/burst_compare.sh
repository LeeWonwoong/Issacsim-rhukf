#!/usr/bin/env bash
# burst 밴드 (지속20초 대비): SWEEP_BURST=rand:4,10,4,8 (ON~U(4,10)/OFF~U(4,8), 표준 FDI 간헐).
# 핵심: burst δ0.8이 attack-only서 track 추락시키나? (지속=추락 vs burst=약할수도)
# 밴드경계 집중: δ{0.6,0.7,0.8} × ws{0,8,12} × 4패턴 × ep2. 단일축+동시축. 2h 예산(모드당 timeout 3300s).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
export SWEEP_BURST="rand:4,10,4,8"
LOG=burst_compare.log
echo "════ burst 밴드 비교 시작 $(date '+%m-%d %H:%M:%S') (SWEEP_BURST=$SWEEP_BURST) ════" | tee $LOG

run_axis () {  # $1=태그 $2=yaw_ratio $3=biases
  local TAG=$1 YR=$2 BIAS=$3 OUT=results_burst_$1
  echo "[$(date '+%H:%M:%S')] $TAG (yaw_ratio=$YR) START" | tee -a $LOG
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  sleep 5
  rm -rf $OUT; mkdir -p $OUT
  timeout 3300 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --torque-yaw-ratio $YR --capture-mode hijack \
    --capture-patterns circle,figure8,waypoint,aggressive \
    --capture-disturbances none:0,wind_turbulence:8,wind_turbulence:12 \
    --capture-biases $BIAS --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] $TAG DONE rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0)" | tee -a $LOG
}

# 단일축(roll): δ{0.6,0.7,0.8}×4.36
run_axis single 0.0 2.616,3.052,3.488
# 동시축(roll=pitch): δ{0.6,0.7,0.8}×3.083
run_axis balanced 1.0 1.8497,2.1580,2.4663

pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch BURST_COMPARE_DONE
