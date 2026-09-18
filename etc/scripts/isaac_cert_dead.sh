#!/usr/bin/env bash
# cert_DEAD (2026-09-14, HANDOFF §9 ⑤) — 데드라인 티어별: δ{0.80,0.84}(3.488/3.662 N·m) × dhover{5,10,15,25} × ws{7,10} × 5패턴 × 2ep = 160 ep (~55 min)
#   무풍 데드라인 실측(δ0.80 d≤0.5s 100%·1–2.5s 65–85% / δ0.84 d0.5s 85%·≥1.0s 0–5%)이 바람에서 유지되는지. 환경은 isaac_commit.sh 의 cert 블록과 동일.
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
OUT=results/claudecodefortest/cert_DEAD; rm -rf $OUT; mkdir -p $OUT; kill_all; log "cert_DEAD START (δ0.80/0.84 × dhover5/10/15/25 × ws7/10)"
env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="dhover5,dhover10,dhover15,dhover25" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances wind_turbulence:7,wind_turbulence:10 \
  --capture-biases 3.488,3.662 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "cert_DEAD END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
kill_all
