#!/usr/bin/env bash
# isaac_final4 (2026-09-05) — 본실험: SWIRL 상위2 · Adam 상위1 · SGD 상위1 × seed{42,43,44,45}
#   SWIRL#1 CALwin  : R2·Q1e-2·P0.01·abs  (Isaac 검증완료, 실효P≈0.035)   — s42,43,44 보유 → s45 만
#   SWIRL#2 P0.03R1 : R1·Q1e-3·P0.03·abs  (surrogate 최적 h=0.39, Isaac 미검증) — 4런 전부
#   Adam    +Huber  : lr3e-4 (n=3 비교에서 최강 Adam)                       — s42,43,44 보유 → s45 만
#   SGD             : lr1e-2 (SGD 중 상위)                                  — s42 보유 → s43,44,45
#   총 9런 ≈ 12.6h.  기존 RUN_DONE 은 자동 SKIP.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_final4.log
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
  local er=$(grep -c 'sequence size exceeds' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $TAG END rc=$rc ep=$ep ddserr=$er"|tee -a $LOG
  touch $OUT/RUN_DONE; return 0; }
SW1="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-2 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=2   RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10"
SW2="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1   RHUKF_FORM=absolute RHUKF_PINIT=0.03 RHUKF_ALPHA=0.10"
AH="NET_HIDDEN=16 ADAM_LOSS=huber ADAM_AMSGRAD=0 ADAM_LR=3e-4"
SG="NET_HIDDEN=16 OPT=sgd ADAM_LOSS=mse ADAM_LR=1e-2"
# 시드별로 4-arm 을 묶어서 도는 순서 — 중간에 끊겨도 시드 단위로 완결
for SD in 45 42 43 44; do
  case $SD in 45) run calwin_s45 rhukf 45 "$SW1";; esac
  run sw2_s$SD rhukf $SD "$SW2"
  case $SD in 45) run adamhub_s45 adam 45 "$AH";; esac
  case $SD in 43|44|45) run sgd_s$SD adam $SD "$SG";; esac
done
kill_isaac; touch FINAL4_DONE
echo "[$(date '+%m-%d %H:%M')] 본실험 4시드 완료"|tee -a $LOG
