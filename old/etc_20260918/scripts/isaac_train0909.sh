#!/usr/bin/env bash
# train0909 — 밴드 보정 후(GO 신호) Isaac 4팔 비교: SWIRL top-2 + Adam + SGD (각 seed42 200ep)
#   신 샘플러(약버스트+급작치명, LOC done) · frozen_v3 실공간
#   환경: frozen_v3 + 실공간 궤적 + 강풍상한 6 (config 반영됨) + LOC done (PHYSICAL_TERMINALS)
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f TRAIN0909_GO ]; do sleep 60; done   # 밴드 보정 완료 후 수동 GO
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07
LOG=isaac_train0909.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }

kill_all; OUT=results_t9_swirl; rm -rf $OUT; mkdir -p $OUT
log "SWIRL 학습 START (신 샘플러 lethal30%, LOC done)"
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
    timeout 12000 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
log "SWIRL END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"

kill_all; OUT=results_t9_swirl2; rm -rf $OUT; mkdir -p $OUT
log "SWIRL#2 (P0.03) START"
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
    timeout 12000 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
log "SWIRL#2 END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"

kill_all; OUT=results_t9_adam; rm -rf $OUT; mkdir -p $OUT
log "Adam+Huber 학습 START"
env NET_HIDDEN=16 ADAM_LR=3e-4 \
    timeout 12000 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
log "Adam END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"

kill_all; OUT=results_t9_sgd; rm -rf $OUT; mkdir -p $OUT
log "SGD START"
env NET_HIDDEN=16 ADAM_LR=3e-4 OPT=sgd \
    timeout 12000 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
log "SGD END csv=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)"
kill_all; touch TRAIN0909_DONE; log "4팔 학습 완료"
