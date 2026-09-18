#!/usr/bin/env bash
# isaac_adamfront (2026-09-04) — Adam 쪽 Pareto 전선(frontier)을 채운다.
#   근거: SWIRL 은 P₀ 로 (학습셋 d′ ↔ held-out 일반화) 전선을 그린다 — P0.002 d′2.93/evalFA .583
#         … P0.005 d′2.66/evalFA .422. Adam 은 지금 2점(lr 3e-4, 1e-3)뿐이라 전선 비교가 불가.
#   추가: ① Adam+Huber lr3e-4 (유계영향 공정 베이스라인 — 리뷰어 1순위 반박)
#         ② Adam+MSE  lr1e-4 (전선 저-d′ 쪽 연장)
#   ★ calwinner 완료 후 자동. Isaac 순차.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_adamfront.log
echo "[$(date '+%m-%d %H:%M')] calwinner 완료 대기..." | tee -a $LOG
while [ ! -f results_o3_calwinner/RUN_DONE ]; do sleep 120; done
sleep 30
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local TAG=$1 ENVC=$2; local OUT=results_o3_$TAG
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $TAG"|tee -a $LOG; return; }
  echo "[$(date '+%m-%d %H:%M')] $TAG START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $ENVC timeout 10800 python3 -u online_rl_main.py --headless --agent adam \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $TAG END rc=$rc ep=$ep"|tee -a $LOG; touch $OUT/RUN_DONE; }
run adam_huber_3e4 "NET_HIDDEN=16 ADAM_LOSS=huber ADAM_AMSGRAD=0 ADAM_LR=3e-4"
run adam_mse_1e4   "NET_HIDDEN=16 ADAM_LOSS=mse   ADAM_AMSGRAD=0 ADAM_LR=1e-4"
kill_isaac; touch ADAMFRONT_DONE
echo "[$(date '+%m-%d %H:%M')] Adam 전선 완료"|tee -a $LOG
