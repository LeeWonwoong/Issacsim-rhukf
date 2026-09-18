#!/usr/bin/env bash
# isaac_ext (09-16 새벽 자동 연장) — EXT_ISAAC_SET=A: 조건 지속 20 ep 스케줄(surrogate per20 시드 42 와 같은 체인) SW·Adam3e-4 150 ep + 급변 블록 Adam1e-3 시드 42 120 ep
#                                  EXT_ISAAC_SET=B: 급변 블록 시드 44 SW·Adam3e-4 120 ep (Isaac 3시드째)
#   무대 플래그는 isaac_tonight.sh 와 동일(aggressive 수정 코드·하한 0.10). Isaac 에피소드 번호가 1부터일 수 있어 스케줄이 1 ep 어긋날 수 있음(분석에서 CSV episode 로 정렬).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.10 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0 GAMMA=0.9 EPS_DECAY=4000 HOVER_DWELL=5 EPS_HOVER_P=0.1 REPLAY_MODE=uniform
export WIND_WIN="60,290" ATK_COND_DONE=50
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ -f results/claudecodefortest/isaac_tonight.running ] || ps aux | grep -q "[i]saac_tonight.sh"; do sleep 60; done   # 오늘 밤 Isaac 이 끝난 뒤 착수
SW="RHUKF_PINIT=${EXT_SW_PINIT:-0.03} RHUKF_R=1 RHUKF_N=${EXT_SW_N:-7} RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1 RHUKF_HUBER_C=3"
ADAM="ADAM_INIT=he ADAM_HUBER_BETA=3 ADAM_AMSGRAD=0"
BLOCK="0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4"
PER20=$(python3 -c "import sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q; print(Q.ext_stage('per20', 42)['SURR_TIER_SCHED'])")
run_one(){ NAME=$1; AGENT=$2; SEED=$3; EPS=$4; SCHED=$5; shift 5; OUT=results/claudecodefortest/isaac_x_${NAME}_s${SEED}
  [ -f $OUT/RUN_DONE ] && { log "$OUT 이미 완료 — 건너뜀"; return; }
  rm -rf $OUT; mkdir -p $OUT; echo "$SCHED" > $OUT/WIND_SCHED.txt; kill_all; log "$OUT START (agent=$AGENT seed=$SEED ep=$EPS)"
  env SCENARIO_SEED=$SEED WIND_SCHED="$SCHED" "$@" timeout 12600 python3 -u online_rl_main.py --headless --agent $AGENT --max-ep $EPS --speed 2.5 --seed $SEED --outdir $OUT > $OUT/train.log 2>&1
  log "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0) err=$(grep -c Traceback $OUT/train.log) hard=$(grep -c HARD_RESET $OUT/train.log)"; touch $OUT/RUN_DONE; }
case "${EXT_ISAAC_SET:-A}" in
  A) run_one per20_swirl rhukf 42 150 "$PER20" $SW
     run_one per20_adam  adam  42 150 "$PER20" $ADAM ADAM_LR=3e-4
     run_one blk_adam1e-3 adam 42 120 "$BLOCK" $ADAM ADAM_LR=1e-3 ;;
  B) run_one blk_swirl rhukf 44 120 "$BLOCK" $SW
     run_one blk_adam  adam  44 120 "$BLOCK" $ADAM ADAM_LR=3e-4 ;;
esac
kill_all; touch results/claudecodefortest/ISAAC_EXT_DONE; log "isaac_ext ${EXT_ISAAC_SET:-A} 전체 완료"
