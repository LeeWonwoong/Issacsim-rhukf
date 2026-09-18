#!/usr/bin/env bash
# bandmap v2 (2026-09-10) — ★ACCEL_FF=1 (추종 회귀 수정 후) 로 전체 지도 재측정. 정책 {track, dhover3}, 5패턴 × 3ep = 30에피/조합
#   A 무풍: δ{0.82,0.85,0.90} × ON{5,10,15,30}  +  δ{0.74,0.77,0.80} × ON10        (15조합, 450에피 ≈ 2.5h)
#   B ws6 : δ{0.80,0.82,0.85} × ON{15,30}                                            (6조합, 180에피 ≈ 1h)
#   C ws8 : 동일                                                                     (6조합, 180에피 ≈ 1h)
cd /home/acsl/projects/Issacsim-rhukf

set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export ACCEL_FF=1 SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_bandmap.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
PATS=waypoint,circle,figure8,aggressive,scurve; POL="track,dhover3"
# run <name> <ON스텝> <biases> <disturbances> [WIND env]
run(){ kill_all; OUT=results_bandmap_$1; rm -rf $OUT; mkdir -p $OUT; log "$1 START (ON=$2 b=$3 d=$4)"
  env WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=$((180+$2)) CAPTURE_POLICIES="$POL" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances $4 --capture-biases $3 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "$1 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"; }
# δ→bias: 0.74→3.226 0.77→3.357 0.80→3.488 0.82→3.575 0.85→3.706 0.90→3.924
run A_on5   5  3.575,3.706,3.924 none:0
run A_on10  10 3.226,3.357,3.488,3.575,3.706,3.924 none:0
run A_on15  15 3.226,3.357,3.488,3.575,3.706,3.924 none:0
run A_on30  30 3.226,3.357,3.488,3.575,3.706,3.924 none:0
run B_on15  15 3.488,3.575,3.706 wind_turbulence:6,wind_turbulence:8
run B_on30  30 3.488,3.575,3.706 wind_turbulence:6,wind_turbulence:8
kill_all; touch BANDMAP_DONE; log "bandmap v2 완료"
