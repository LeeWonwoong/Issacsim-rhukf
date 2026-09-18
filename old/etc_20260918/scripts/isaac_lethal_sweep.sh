#!/usr/bin/env bash
# isaac_lethal_sweep — 강공격 비중 30/40/50 × {SWIRL, Adam, SGD} huber 전부 OFF. seed42 200ep.
#   목적: 추락 빈도↑ = 가치학습 난이도↑ 에서 옵티마이저 격차 확대 확인.
#   SWIRL huber off = RHUKF_HUBER_C=1e9 · Adam/SGD off = ADAM_LOSS=mse
#   현 train0909(SGD+AdamMSE) 완료 후 자동 시작.
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f ADAMMSE_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07
LOG=isaac_lethal_sweep.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ # $1=agent_flag $2=extra_env $3=tag
  kill_all; OUT=results_lth_$3; rm -rf $OUT; mkdir -p $OUT
  log "$3 START"
  env NET_HIDDEN=16 $2 timeout 12000 python3 -u online_rl_main.py --headless $1 \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  log "$3 END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"
}
for L in 0.30 0.40 0.50; do
  Lt=$(echo $L | tr -d '.')
  # SWIRL huber off
  run "--agent rhukf" "PROB_LETHAL=$L RHUKF_N=6 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 RHUKF_HUBER_C=1e9" "swirl_L${Lt}"
  # Adam huber off (MSE)
  run "--agent adam" "PROB_LETHAL=$L ADAM_LR=3e-4 ADAM_LOSS=mse" "adam_L${Lt}"
  # SGD huber off (MSE)
  run "--agent adam" "PROB_LETHAL=$L ADAM_LR=3e-4 OPT=sgd ADAM_LOSS=mse" "sgd_L${Lt}"
done
kill_all; touch LETHAL_SWEEP_DONE; log "강공격 스윕 완료"
