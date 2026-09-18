#!/usr/bin/env bash
# A-1 (2026-09-11 사용자 지시) — 각도(flip) 종료 규칙 제거 후 밴드 재측정: 종료 = 지면(고도<0.5m) ∨ 비행공간 이탈(10m·1s)
#   δ{0.76,0.78,0.80,0.82,0.84} × plateau 30 × 5패턴 × ws{0,6} × {track,dhover3} × 2ep = 200 ep (~1.1 h). failsafe hover, 변조 off, zu 로깅
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0 FLIP_TERMINAL=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_cert.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ ! -f results/claudecodefortest/CERTFILL_DONE ]; do sleep 30; done
kill_all; OUT=results/claudecodefortest/cert_A1; rm -rf $OUT; mkdir -p $OUT; log "A1 START (FLIP_TERMINAL=0) biases=3.314,3.401,3.488,3.575,3.662"
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 3.314,3.401,3.488,3.575,3.662 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "A1 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
kill_all; touch results/claudecodefortest/A1_DONE
