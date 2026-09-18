#!/usr/bin/env bash
# stealth_test (2026-09-01): 약공격 우세(STEALTH_FRAC=0.8) × {SWIRL,Adam} seed42
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=stealth_test.log
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400 STEALTH_FRAC=0.8
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local OUT=results_st_$1
  if [ -f $OUT/RUN_DONE ]; then echo "[$(date '+%H:%M')] $1 SKIP"|tee -a $LOG; return; fi
  echo "[$(date '+%m-%d %H:%M')] $1 agent=$2 STEALTH=0.8 START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 --max-ep 200 --speed 1 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $1 END rc=$rc ep=$ep"|tee -a $LOG
  if [ "$ep" -ge 150 ] || [ "$rc" = "0" ]; then touch $OUT/RUN_DONE; fi
}
CB="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.01"
run swirl rhukf "$CB"
run adam  adam  "ADAM_LR=1e-3 NET_HIDDEN=16"
kill_isaac
echo "════ stealth_test 완료 $(date '+%m-%d %H:%M') ════"|tee -a $LOG
touch STEALTH_TEST_DONE
