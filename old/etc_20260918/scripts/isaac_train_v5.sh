#!/usr/bin/env bash
# train_v5 (2026-09-12 밤) — 확정 환경(v5)에서 Isaac 온라인 학습: SWIRL vs Adam × 추락벌점 {0, −5}. 200 ep · 300 스텝 · seed 42.
#   환경: 샘플러 v5 (δ 0.5·U(0.15,0.72)+0.5·U(0.72,0.84), plateau U(25,50), 온셋 U(60,200), 1버스트, P(공격)=0.5)
#         G2+ 필터 · 클립 4.0 · 변조 off · arm 0.05 · failsafe hover · 종료 = 지면 ∨ 10 m·1 s (대칭) · 각도 규칙 없음
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.15 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=50 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export EPS_DECAY=4000 BUFFER_SIZE=50000 GAMMA=0.85 NET_HIDDEN=16
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_train_v5.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
SW="RHUKF_PINIT=0.2 RHUKF_R=2 RHUKF_N=7 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1"
while [ ! -f results/claudecodefortest/DEADLINE_DONE ]; do sleep 60; done
for RUN in "swirl 0" "adam 0" "swirl -5" "adam -5"; do
  set -- $RUN; AG=$1; PEN=$2; OUT=results/claudecodefortest/train_v5_${AG}_pen${PEN#-}; rm -rf $OUT; mkdir -p $OUT
  kill_all; log "$OUT START (TERMINAL_PEN=$PEN)"
  if [ "$AG" = swirl ]; then
    env $SW TERMINAL_PEN=$PEN timeout 14400 python3 -u online_rl_main.py --headless --agent rhukf --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  else
    env ADAM_LR=3e-4 TERMINAL_PEN=$PEN timeout 14400 python3 -u online_rl_main.py --headless --agent adam --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  fi
  log "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0) err=$(grep -c Traceback $OUT/train.log)"; touch $OUT/RUN_DONE
done
kill_all; touch results/claudecodefortest/TRAIN_V5_DONE; log "train_v5 전체 완료"
