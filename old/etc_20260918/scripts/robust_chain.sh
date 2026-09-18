#!/usr/bin/env bash
# robust_chain — 강건성/샘플효율 6런 (2026-08-29): 짧은ON(10,20)/긴OFF(25,40) sparse 공격.
#   RHUKF-B(B_r15_pd001) vs Adam × seed{42,43,44}. 표준 reward(옵티마이저만 변수).
#   측정: F1/reward seed 분산(강건성) + 학습곡선(샘플효율).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=robust_chain.log
# ★공격 setting: 짧은 ON + 긴 OFF (sparse)
export ATK_ON_LO=10 ATK_ON_HI=20 ATK_OFF_LO=25 ATK_OFF_HI=40
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
echo "[$(date '+%m-%d %H:%M')] robust_chain 시작 (ON10-20/OFF25-40 sparse)" | tee -a $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ # $1=이름 $2=agent $3=env $4=seed
  local OUT=results_rb_$1
  if [ -f $OUT/RUN_DONE ]; then echo "[$(date '+%H:%M')] $1 SKIP" | tee -a $LOG; return; fi
  echo "[$(date '+%m-%d %H:%M')] $1 agent=$2 seed=$4 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 1 --seed $4 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $1 END rc=$rc ep=$ep" | tee -a $LOG
  if [ "$ep" -ge 150 ] || [ "$rc" = "0" ]; then touch $OUT/RUN_DONE; fi
}
CB="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.01"
CA="ADAM_LR=1e-3 NET_HIDDEN=16"
# 인터리브: 페어별로 (부분완료시도 쌍 유지)
run rhukf_s42 rhukf "$CB" 42
run adam_s42  adam  "$CA" 42
run rhukf_s43 rhukf "$CB" 43
run adam_s43  adam  "$CA" 43
run rhukf_s44 rhukf "$CB" 44
run adam_s44  adam  "$CA" 44
kill_isaac
echo "════ robust_chain 6런 완료 $(date '+%m-%d %H:%M') ════" | tee -a $LOG
touch ROBUST_CHAIN_DONE
