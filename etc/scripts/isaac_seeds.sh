#!/usr/bin/env bash
# isaac_seeds (2026-09-04) — 핵심 대결의 시드 반복. n=1 → n=3.
#   대결: SWIRL CALwin (R2·Q1e-2·P0.01·abs)  vs  Adam+Huber lr3e-4
#         → 두 config 가 fpr 0.137 vs 0.138 로 **작동점이 거의 동일**해서 공정 비교가 성립.
#   seed42 는 이미 있음 → 43, 44 추가.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_seeds.log
kill_isaac(){ for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local TAG=$1 AG=$2 SD=$3 ENVC=$4; local OUT=results_o3_$TAG
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $TAG"|tee -a $LOG; return 0; }
  echo "[$(date '+%m-%d %H:%M')] $TAG START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $ENVC timeout 10800 python3 -u online_rl_main.py --headless --agent $AG \
      --max-ep 200 --speed 2.5 --seed $SD --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $TAG END rc=$rc ep=$ep"|tee -a $LOG
  touch $OUT/RUN_DONE; return 0; }
SW="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-2 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=2 RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10"
AH="NET_HIDDEN=16 ADAM_LOSS=huber ADAM_AMSGRAD=0 ADAM_LR=3e-4"
run calwin_s43   rhukf 43 "$SW"
run adamhub_s43  adam  43 "$AH"
run calwin_s44   rhukf 44 "$SW"
run adamhub_s44  adam  44 "$AH"
run adam_mse_1e4 adam  42 "NET_HIDDEN=16 ADAM_LOSS=mse ADAM_AMSGRAD=0 ADAM_LR=1e-4"
kill_isaac; touch SEEDS_DONE
echo "[$(date '+%m-%d %H:%M')] 시드 반복 완료"|tee -a $LOG
