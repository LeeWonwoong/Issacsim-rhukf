#!/usr/bin/env bash
# roll_v30p (2026-09-15) — 학습 정책 greedy 롤아웃: isaac_v30p_{learner}/final_model.pt 를 LOAD_MODEL 로 올려 스윕 셀에서 비행. δ{0,0.05,0.10,0.80}(0/0.218/0.436/3.488 N·m) × ws{0,10} × {circle,aggressive} × 2ep = 32 ep/학습기. detail CSV = 스텝별 위치·기준·행동·NIS.
#   무대·필터·약속 hover(D5)·ATK_COND_DONE 은 isaac_v30p 학습과 동일. 학습기 순서 swirl → adam → ukf → ekf.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.15 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0 GAMMA=0.9 EPS_DECAY=4000 HOVER_DWELL=5 EPS_HOVER_P=0.1 ATK_COND_DONE=50
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
nvidia-smi >/dev/null 2>&1 || { log "GPU 불가 — 중단"; exit 1; }
roll(){ NAME=$1; AGENT=$2; shift 2; OUT=results/claudecodefortest/roll_v30p_$NAME; MODEL=results/claudecodefortest/isaac_v30p_$NAME/final_model.pt
  [ -f $OUT/sweep_summary.csv ] && [ "$(wc -l < $OUT/sweep_summary.csv)" -ge 33 ] && { log "$OUT 이미 완료"; return; }
  rm -rf $OUT; mkdir -p $OUT; kill_all; log "$OUT START (LOAD_MODEL=$MODEL agent=$AGENT)"
  env "$@" LOAD_MODEL=$MODEL SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="model" \
  timeout 5400 python3 -u online_rl_main.py --headless --agent $AGENT --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns circle,aggressive --capture-disturbances none:0,wind_turbulence:10 \
    --capture-biases 0.0,0.218,0.436,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "$OUT END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log) loaded=$(grep -c 'LOAD_MODEL' $OUT/run.log)"; }
SW="RHUKF_PINIT=0.03 RHUKF_R=1 RHUKF_N=7 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1 RHUKF_HUBER_C=3"
KTD="RHUKF_PINIT=0.03 RHUKF_R=1 RHUKF_N=1 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=0 RHUKF_HUBER_C=3"
roll swirl rhukf $SW
roll adam adam ADAM_LR=3e-4 ADAM_INIT=he ADAM_HUBER_BETA=3 ADAM_AMSGRAD=0
roll ukf rhukf $KTD RHUKF_MODE=ukf
roll ekf rhukf $KTD RHUKF_MODE=ekf
kill_all; log "roll_v30p 전체 완료"
