#!/usr/bin/env bash
# isaac_commit (2026-09-13) — 재부팅 후 Isaac 체인: ① train_v5 나머지(adam pen0) ② 약속 hover D∈{5,10} × {swirl, adam} (4항 보상 R0, hover 탐험 0.05, γ0.95, decay 12000)
#   D=3/20 은 surrogate v15 결과 보고 추가. 한 런 ≈ 2–3 h (200 ep · 300 스텝 · speed 2.5).
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
SW="RHUKF_PINIT=0.05 RHUKF_R=3 RHUKF_N=7 RHUKF_Q=1e-3 RHUKF_ALPHA=0.1 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_FORM=absolute RHUKF_SPAS=1"
nvidia-smi >/dev/null 2>&1 || { log "GPU 드라이버 불일치 — 재부팅 필요. 중단"; exit 1; }
run_one(){ AG=$1; OUT=$2; shift 2; rm -rf $OUT; mkdir -p $OUT; kill_all; log "$OUT START ($*)"
  if [ "$AG" = swirl ]; then env $SW "$@" timeout 14400 python3 -u online_rl_main.py --headless --agent rhukf --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  else env ADAM_LR=3e-4 "$@" timeout 14400 python3 -u online_rl_main.py --headless --agent adam --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1; fi
  log "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0) err=$(grep -c Traceback $OUT/train.log) hard=$(grep -c HARD_RESET $OUT/train.log)"; touch $OUT/RUN_DONE; }
# ⓪ 스텔스 대역 캡처 cert_S: δ{0.03,0.05,0.08}(=0.131/0.218/0.349 N·m) × 5패턴 × ws{0,6} × {track,dhover3} × 2ep — 풀 v5c 용 (δ<0.1 실측)
if [ ! -f results/claudecodefortest/cert_S/sweep_summary.csv ]; then
  OUT=results/claudecodefortest/cert_S; rm -rf $OUT; mkdir -p $OUT; kill_all; log "cert_S START (stealth δ 0.03/0.05/0.08)"
  env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases 0.131,0.218,0.349 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "cert_S END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
fi
# ⓪′ 바람 티어 실측 cert_WIND (HANDOFF §9): δ{0,0.5,0.7,0.76,0.80,0.84}(N·m 0/2.18/3.052/3.314/3.488/3.662) × ws{0,7,10} × 5패턴 × {track,dhover3} × 2ep
if [ ! -f results/claudecodefortest/cert_WIND/sweep_summary.csv ]; then
  OUT=results/claudecodefortest/cert_WIND; rm -rf $OUT; mkdir -p $OUT; kill_all; log "cert_WIND START (wind tiers 0/7/10)"
  env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
  timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:7,wind_turbulence:10 \
    --capture-biases 0.0,2.18,3.052,3.314,3.488,3.662 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "cert_WIND END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
fi
# ① train_v5 나머지: adam pen0 (swirl pen0 짝)
[ -f results/claudecodefortest/train_v5_adam_pen0/RUN_DONE ] || run_one adam results/claudecodefortest/train_v5_adam_pen0 EPS_DECAY=4000 GAMMA=0.85 ATK_ON_HI=50
# ② 약속 hover D 스캔 (R0 4항 보상 그대로)
DECAY=$(cat results/claudecodefortest/COMMIT_DECAY 2>/dev/null || echo 8000); GAM=$(cat results/claudecodefortest/COMMIT_GAMMA 2>/dev/null || echo 0.95)
for D in 5 10; do for AG in swirl adam; do
  run_one $AG results/claudecodefortest/train_commit_D${D}_${AG} HOVER_DWELL=$D EPS_HOVER_P=0.1 GAMMA=$GAM EPS_DECAY=$DECAY
done; done
kill_all; touch results/claudecodefortest/ISAAC_COMMIT_DONE; log "isaac_commit 전체 완료"
