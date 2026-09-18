#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_inference; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] inference START"
env NET_HIDDEN=16 RHUKF_N=6 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
    LOAD_MODEL=results_t9_swirl2/final_model.pt PROB_LETHAL=0.6 \
    timeout 3000 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 12 --speed 2.5 --seed 777 --log-zu --outdir $OUT > $OUT/train.log 2>&1
echo "[$(date '+%m-%d %H:%M')] inference END"; kill_all; touch INFER_DONE
