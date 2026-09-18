#!/usr/bin/env bash
# cert (2026-09-11 밤) — 확정 스윕. 전제: failsafe hover 채택(FAILSAFE_PARAMS), 속도변조 off, rise 없음(플랜트 τ 만), G2+/클립4.0 은 오프라인.
#   A 결과성 곡선  δ 0.70~0.82 (0.02) × plateau 30 × 5패턴 × ws{0,6} × {track,dhover3} × 3ep = 420
#   B plateau      δ{0.80,0.82} × plateau{15,20,25,40} × 5 × ws{0,6} × {track,dhover3} × 2ep = 160
#   W 약공격 성능  δ{0.15,0.25,0.4,0.5,0.6} × plateau 30 × 5 × ws{0,6} × track × 2ep = 100
#   D 평시 failsafe 파라미터 상시  δ0 × 5 × ws{0,6} × track × 2ep, FAILSAFE_ALWAYS=1 = 20 (+ 명목 δ0 은 A/W 의 δ0 없음 → 20 추가)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_cert.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ # name biases policies episodes atk_end extra_env
  kill_all; OUT=results/claudecodefortest/cert_$1; rm -rf $OUT; mkdir -p $OUT; log "$1 START biases=$2 pol=$3 ep=$4 atk_end=$5 $6"
  env $6 WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=$5 CAPTURE_POLICIES="$3" \
  timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases $2 --episodes $4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "$1 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"; }
while [ ! -f results/claudecodefortest/FAILSAFE_DONE ]; do sleep 30; done
run A     3.052,3.139,3.226,3.314,3.401,3.488,3.575 track,dhover3 3 210 "X=1"
run B15   3.488,3.575 track,dhover3 2 195 "X=1"
run B20   3.488,3.575 track,dhover3 2 200 "X=1"
run B25   3.488,3.575 track,dhover3 2 205 "X=1"
run B40   3.488,3.575 track,dhover3 2 220 "X=1"
run W     0.654,1.090,1.744,2.180,2.616 track 2 210 "X=1"
run D_on  0 track 2 210 "FAILSAFE_ALWAYS=1"
run D_off 0 track 2 210 "X=1"
kill_all; touch results/claudecodefortest/CERT_DONE; log "cert 전체 완료"
