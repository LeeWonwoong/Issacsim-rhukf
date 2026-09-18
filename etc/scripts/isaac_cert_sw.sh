#!/usr/bin/env bash
# cert_SW (2026-09-14) — 바람 속 약공격·스텔스: δ{0.05,0.08,0.1,0.2,0.3,0.4}(0.218/0.349/0.436/0.872/1.308/1.744 N·m) × ws{7,10} × 5패턴 × {track,dhover3} × 2ep = 240 ep (~90 min)
#   풀 v5c 의 atk_s*_ws7/10·atk_b0–b3_ws7/10 키를 채운다(없으면 env8 이 무풍 분포로 폴백 = 비물리). ws10 hover 지오펜스율 n 도 보강.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.15 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
nvidia-smi >/dev/null 2>&1 || { log "GPU 불가 — 중단"; exit 1; }
OUT=results/claudecodefortest/cert_SW; rm -rf $OUT; mkdir -p $OUT; kill_all; log "cert_SW START (약공격·스텔스 × ws7/10 × track,dhover3)"
env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances wind_turbulence:7,wind_turbulence:10 \
  --capture-biases 0.218,0.349,0.436,0.872,1.308,1.744 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "cert_SW END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
kill_all
