#!/usr/bin/env bash
# adam_mse — Adam huber OFF (순수 MSE) 추가 팔. SGD 완료 후 자동. seed42 200ep, 신 샘플러.
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f CALIBCAP_DONE ]; do sleep 60; done   # 캡처 후로 순서 변경
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_t9_adammse; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] Adam+MSE(huber off) START" | tee -a isaac_train0909.log
env NET_HIDDEN=16 ADAM_LR=3e-4 ADAM_LOSS=mse \
  timeout 12000 python3 -u online_rl_main.py --headless --agent adam \
  --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
echo "[$(date '+%m-%d %H:%M')] Adam+MSE END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)" | tee -a isaac_train0909.log
kill_all; touch ADAMMSE_DONE
