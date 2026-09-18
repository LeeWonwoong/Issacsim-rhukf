#!/usr/bin/env bash
# lethal05 — 강공격 비중 0.5 에서 3팔 재학습 (환경 난이도↑ 에서 격차 확인). 각 seed42 200ep.
#   SWIRL = 최고 config (P0.03·N5·α0.1·R1, huber_c 기본 3.0 = 구조 그대로)
#   ★표준형(2026-09-09 확정): SWIRL full(huber-R c3) vs Adam+Huber(표준 DQN) vs SGD+Huber. lr 3e-4
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 PROB_LETHAL=0.50
LOG=isaac_lethal05.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ kill_all; OUT=results_l5_$3; rm -rf $OUT; mkdir -p $OUT; log "$3 START"
  env NET_HIDDEN=16 $2 timeout 12000 python3 -u online_rl_main.py --headless $1 \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  log "$3 END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"; }
run "--agent rhukf" "RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1" swirl
run "--agent adam" "ADAM_LR=3e-4" adam
run "--agent adam" "ADAM_LR=3e-4 OPT=sgd" sgd
kill_all; touch LETHAL05_DONE; log "lethal0.5 3팔 완료"
