#!/usr/bin/env bash
# isaac_rerun_sw (09-16 03:20): 오늘 밤 Isaac SWIRL 은 GPU 경합으로 learn-step 63–68 ms > 예산 40 ms(모든 에피) → 행동 지연 교란. GPU 단독으로 SWIRL s42·s43 재실행(120 ep), 끝나면 surrogate GPU 워커 재개.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.10 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0 GAMMA=0.9 EPS_DECAY=4000 HOVER_DWELL=5 EPS_HOVER_P=0.1 REPLAY_MODE=uniform
export WIND_SCHED="0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4" WIND_WIN="60,290" ATK_COND_DONE=50
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
SW="RHUKF_PINIT=0.03 RHUKF_R=1 RHUKF_N=7 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1 RHUKF_HUBER_C=3"
for SEED in 42 43; do OUT=results/claudecodefortest/isaac_t2_swirl_s${SEED}
  [ -f $OUT/RUN_DONE ] && continue
  rm -rf $OUT; mkdir -p $OUT; kill_all; log "$OUT START (SWIRL 재실행, GPU 단독, seed=$SEED, 120 ep)"
  env SCENARIO_SEED=$SEED $SW timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf --max-ep 120 --speed 2.5 --seed $SEED --outdir $OUT > $OUT/train.log 2>&1
  log "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log) err=$(grep -c Traceback $OUT/train.log) hard=$(grep -c HARD_RESET $OUT/train.log)"; touch $OUT/RUN_DONE
done
kill_all
P=$(cat results/claudecodefortest/night/tonight/PAUSED_GPU_PIDS 2>/dev/null); [ -n "$P" ] && kill -CONT $P 2>/dev/null
touch results/claudecodefortest/ISAAC_RERUN_DONE; log "isaac_rerun_sw 완료 → surrogate GPU 워커 재개($P)"
