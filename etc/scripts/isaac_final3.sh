#!/usr/bin/env bash
# final3 — Isaac 본선 밤샘 (2026-09-09 사용자 확정 스펙):
#   공통: γ0.85 · batch128 · buffer 50000 · eps decay 5000 · 200ep · lethal 0.5 · 표준형
#   SWIRL: N5 · P0.03 · R1 · α0.1 · huber c3   /   Adam: 진짜 베이스라인 (lr 3e-4 · huber)
#   {SWIRL, Adam} × seed {42, 43, 44} = 6런 (~10h)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 PROB_LETHAL=0.50 EPS_DECAY=5000 BUFFER_SIZE=50000
LOG=isaac_final3.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ kill_all; OUT=results_fin_$3; rm -rf $OUT; mkdir -p $OUT; log "$3 START"
  env NET_HIDDEN=16 $2 timeout 12000 python3 -u online_rl_main.py --headless $1 \
    --max-ep 200 --speed 2.5 --seed $4 --outdir $OUT > $OUT/train.log 2>&1
  log "$3 END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"; }
for SD in 42 43 44; do
  run "--agent rhukf" "RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1" swirl_s$SD $SD
  run "--agent adam" "ADAM_LR=3e-4" adam_s$SD $SD
done
kill_all; touch FINAL3_DONE; log "본선 6런 완료"
