#!/usr/bin/env bash
# Adam OFAT sweep (5 config): baseline에서 한 축씩 토글. 각 300ep.
#   1. lr3e-4 ams+ obs- net24  (baseline)
#   2. lr1e-3 ams+ obs- net24  (lr 효과)
#   3. lr3e-4 ams- obs- net24  (amsgrad 효과)
#   4. lr3e-4 ams+ obs+ net24  (관측정규화 /3.5 효과)
#   5. lr3e-4 ams+ obs- net16  (네트워크 효과)
# 확정 학습env: Q_gyro2e-3·회전바람arm0.02·에피400·δ0.1-0.8·burst5-20/10-30·추락bootstrap. speed=20.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 EP_MAX_STEPS=400
SPEED=20; LOG=adam_ofat.log
echo "════ Adam OFAT sweep(5) 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG
kill_isaac(){ pkill -9 -f "run_sim.py --headless" 2>/dev/null||true; for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done; sleep 5; }
run(){  # $1=tag $2=lr $3=ams $4=obs $5=net
  local OUT=results_opt_$1
  echo "[$(date '+%H:%M:%S')] $1 lr=$2 ams=$3 obs=$4 net=$5 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  local NETENV=""; [ "$5" = "16" ] && NETENV="16"
  ADAM_LR=$2 ADAM_AMSGRAD=$3 OBS_NORM=$4 NET_HIDDEN=$NETENV \
    timeout 18000 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 300 --speed $SPEED --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -cE 'Ep [0-9]+: ' $OUT/train.log 2>/dev/null||echo '?')" | tee -a $LOG
}
run base_lr3e4_ams1_obs0_net24 3e-4 1 0 24
run var_lr1e3_ams1_obs0_net24  1e-3 1 0 24
run var_lr3e4_ams0_obs0_net24  3e-4 0 0 24
run var_lr3e4_ams1_obs1_net24  3e-4 1 1 24
run var_lr3e4_ams1_obs0_net16  3e-4 1 0 16
kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch ADAM_OFAT_DONE
