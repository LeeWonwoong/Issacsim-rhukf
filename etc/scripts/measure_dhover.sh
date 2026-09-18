#!/usr/bin/env bash
# dhover deadline (조건③ 재측정, 신 플랜트). 무풍·지속공격에서 pattern×bias×delay dhover 생존.
# δ0.7(3.052)·δ0.8(3.488) × delay{0,1,2,3,4,5,8}+track × 4패턴 × ep3. Q_gyro 2e-3.
# Q fine sweep 완료 대기 후 실행.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
while [ ! -f MEASURE_QR_FINE_DONE ]; do sleep 60; done
sleep 10
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=-1 SWEEP_ATK_START=180 SWEEP_ATK_END=380   # 무풍(WIND_START=-1), 지속공격
LOG=measure_dhover.log; OUT=results_dhover
echo "════ dhover deadline 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done
sleep 5; rm -rf $OUT; mkdir -p $OUT
timeout 9000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --torque-yaw-ratio 0.0 \
  --capture-mode deadline --deadline-patterns circle,figure8,waypoint,aggressive \
  --deadline-biases 3.052,3.488 --deadline-delays 0,1,2,3,4,5,8 \
  --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%H:%M:%S')] DONE rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)" | tee -a $LOG
pkill -9 -f "run_sim.py --headless" 2>/dev/null||true
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done
touch MEASURE_DHOVER_DONE
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
