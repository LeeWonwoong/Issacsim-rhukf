#!/usr/bin/env bash
# isaac_v30p (2026-09-15 09:40) — isaac_v30 의 짝비교판: SCENARIO_SEED=42 로 학습기 4종이 에피소드별 동일 시나리오(패턴·δ·온셋·티어)를 받는다. 나머지 무대 동일. 순서 adam → swirl → ukf → ekf.
#   무대: 바람 티어 스케줄 WIND_SCHED(ep0–59 ws0/7=.6/.4 → 60–109 ws10 100 % → 110– 복귀), 창 60–290, 가족 v5 하한 0.05, 4항 보상, D5, hover 탐험 0.1,
#   γ0.9·decay4000, 공격 조건부 지오펜스 ATK_COND_DONE=50, G2+·클립 4, failsafe 파라미터. 목적: 부분관측이 만드는 reward-타깃 반전이 Isaac 에서도 나타나는지.
#   실행: setsid nohup bash etc/scripts/isaac_v30.sh > results/claudecodefortest/isaac_v30.out 2>&1 < /dev/null &
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.05 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0 GAMMA=0.9 EPS_DECAY=4000 HOVER_DWELL=5 EPS_HOVER_P=0.1 REPLAY_MODE=uniform
export SCENARIO_SEED=42 WIND_SCHED="0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4" WIND_WIN="60,290" ATK_COND_DONE=50
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
nvidia-smi >/dev/null 2>&1 || { log "GPU 불가 — 중단"; exit 1; }
SW="RHUKF_PINIT=0.03 RHUKF_R=1 RHUKF_N=7 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1 RHUKF_HUBER_C=3"
KTD="RHUKF_PINIT=0.03 RHUKF_R=1 RHUKF_N=1 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=0 RHUKF_HUBER_C=3"
ADAM="ADAM_LR=3e-4 ADAM_INIT=he ADAM_HUBER_BETA=3 ADAM_AMSGRAD=0"
run_one(){ NAME=$1; AGENT=$2; shift 2; OUT=results/claudecodefortest/isaac_v30p_$NAME
  [ -f $OUT/RUN_DONE ] && { log "$OUT 이미 완료 — 건너뜀"; return; }
  rm -rf $OUT; mkdir -p $OUT; kill_all; log "$OUT START (agent=$AGENT $*)"
  env "$@" timeout 14400 python3 -u online_rl_main.py --headless --agent $AGENT --max-ep 150 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  log "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0) err=$(grep -c Traceback $OUT/train.log) hard=$(grep -c HARD_RESET $OUT/train.log) geo_nowind_skip=$(grep -c GEOFENCE $OUT/train.log)"; touch $OUT/RUN_DONE; }
run_one adam  adam  $ADAM
run_one swirl rhukf $SW
run_one ukf   rhukf $KTD RHUKF_MODE=ukf
run_one ekf   rhukf $KTD RHUKF_MODE=ekf
kill_all; touch results/claudecodefortest/ISAAC_V30P_DONE; log "isaac_v30p 전체 완료"
