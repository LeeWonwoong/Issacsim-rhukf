#!/usr/bin/env bash
# opt3_queue (2026-09-02): FROZEN-ENV v2 에서 3-옵티마이저 비교 밤샘 큐.
#   선발: 순수 Adam(MSE·no-AMSGrad) lr{1e-3,3e-4} + 순수 SGD(MSE·momentum0) lr{1e-2,1e-3}
#   후발: SWIRL 기준(err pd0.01) + 튜닝 3종(err pd0.1 / abs p0.01 / abs p0.1)
#   전부 headless · speed 2.5 · seed 42 · [16,16] · 200ep
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=opt3_queue.log
# ★ FROZEN-ENV v2 (동결 2026-09-02)
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local OUT=results_o3_$1 AG=$2 ENVC=$3
  if [ -f $OUT/RUN_DONE ]; then say "$1 SKIP"; return; fi
  say "$1 START agent=$AG"
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $ENVC timeout 10800 python3 -u online_rl_main.py --headless --agent $AG \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  say "$1 END rc=$rc ep=$ep"
  [ "$ep" -ge 150 ] || [ "$rc" = "0" ] && touch $OUT/RUN_DONE
}
CB="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5"
# 선발 (순수 베이스라인)
run adam_mse_1e3 adam  "NET_HIDDEN=16 ADAM_LOSS=mse ADAM_AMSGRAD=0 ADAM_LR=1e-3"
run adam_mse_3e4 adam  "NET_HIDDEN=16 ADAM_LOSS=mse ADAM_AMSGRAD=0 ADAM_LR=3e-4"
run sgd_1e2      adam  "NET_HIDDEN=16 OPT=sgd ADAM_LOSS=mse ADAM_LR=1e-2"
run sgd_1e3      adam  "NET_HIDDEN=16 OPT=sgd ADAM_LOSS=mse ADAM_LR=1e-3"
# 후발 (SWIRL 기준 + 튜닝)
run swirl_base    rhukf "$CB RHUKF_PD=0.01"
run swirl_pd01    rhukf "$CB RHUKF_PD=0.1"
run swirl_abs001  rhukf "$CB RHUKF_FORM=absolute RHUKF_PINIT=0.01"
run swirl_abs01   rhukf "$CB RHUKF_FORM=absolute RHUKF_PINIT=0.1"
kill_isaac; touch OPT3_DONE
say "8런 완료."
