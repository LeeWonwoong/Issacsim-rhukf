#!/usr/bin/env bash
# l5d4 — Isaac 재현 실험: surrogate 동결설정(decay4000·N6) 그대로 — SWIRL vs Adam+Huber
#   가설: surrogate 의 Δ+0.055(p=0.02) 격차가 decay4000 조건에서 실환경 전이된다
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f INTSWEEP_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 PROB_LETHAL=0.50 EPS_DECAY=4000
LOG=isaac_l5d4.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ kill_all; OUT=results_l5d4_$3; rm -rf $OUT; mkdir -p $OUT; log "$3 START"
  env NET_HIDDEN=16 $2 timeout 12000 python3 -u online_rl_main.py --headless $1 \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  log "$3 END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"; }
run "--agent rhukf" "RHUKF_N=6 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1" swirl
run "--agent adam" "ADAM_LR=3e-4" adam
kill_all; touch L5D4_DONE; log "decay4000 재현 2팔 완료"
