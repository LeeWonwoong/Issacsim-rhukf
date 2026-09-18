#!/usr/bin/env bash
# failsafe (2026-09-11 사용자 승인) — hover 진입 시 PX4 rate-loop 파라미터 스위치 실측
#   δ{0.76,0.80} × ws{0,6} × 5패턴 × dhover3 × 2ep, ON 3s, SPEED_MOD_AMP=0 (nomod_dh 와 동일 조건 → A/B)
#   + track 도 같이 떠서 failsafe 가 track 에 영향 없음(파라미터는 hover 에서만) 확인
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0
export FAILSAFE_PARAMS="${FAILSAFE_PARAMS:-MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8}"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_failsafe.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ ! -f results/claudecodefortest/PULSE_DONE ]; do sleep 30; done
kill_all; OUT=results/claudecodefortest/failsafe_dh; rm -rf $OUT; mkdir -p $OUT
log "failsafe START ($FAILSAFE_PARAMS)"
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="dhover3,track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 3.314,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "failsafe END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)  FAILSAFE 로그 $(grep -c FAILSAFE $OUT/run.log)"
kill_all; touch results/claudecodefortest/FAILSAFE_DONE
