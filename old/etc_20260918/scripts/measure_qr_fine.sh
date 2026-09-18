#!/usr/bin/env bash
# Q_gyro fine sweep 보강: 3e-3, 7e-3, 1e-2 (기존 배치의 1e-3/2e-3/5e-3와 합쳐 1e-3~1e-2 정밀).
# 첫 배치(measure_qr_weak.sh) 완료 대기 후 실행. aggressive δ{0.1,0.3,0.5} × ws{0,8}.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
# 첫 배치 완료 대기
while [ ! -f MEASURE_QR_DONE ]; do sleep 60; done
sleep 10
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
LOG=measure_qr_fine.log
echo "════ Q_gyro fine 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG
kill_isaac(){ pkill -9 -f "run_sim.py --headless" 2>/dev/null||true; for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done; sleep 5; }
run(){ local OUT=results_qr_$1
  echo "[$(date '+%H:%M:%S')] $1 Qg=$2 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  UKF_Q_GYRO=$2 UKF_Q_VEL=5e-3 timeout 2400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
    --torque-yaw-ratio 0.0 --capture-mode hijack --capture-patterns aggressive \
    --capture-disturbances none:0,wind_turbulence:8 --capture-biases 0.436,1.308,2.18 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)" | tee -a $LOG
}
run G4_qg3e-3 3e-3
run G5_qg7e-3 7e-3
run G6_qg1e-2 1e-2
kill_isaac
echo "════ fine 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch MEASURE_QR_FINE_DONE
