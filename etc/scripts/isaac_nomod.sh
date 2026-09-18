#!/usr/bin/env bash
# nomod (2026-09-11) — 속도변조 제거(SPEED_MOD_AMP=0) time-step NIS A/B: 5패턴 × ws{0,6} × δ{0,0.25,0.80} × track(+δ0.8 dhover3) × 2ep, ON 3s, zu 로깅
#   이어서 연기된 펄스 ON25 조합 2개 실행.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_nomod.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
windows(){ python3 -c "
on,off=$1,$2; t=180; w=[]
while t+on<=280: w.append(f'{t}-{t+on}'); t+=on+off
print(','.join(w))"; }
# ① nomod 캡처
kill_all; OUT=results/claudecodefortest/nomod; rm -rf $OUT; mkdir -p $OUT
log "nomod START (SPEED_MOD_AMP=0, ON 30, δ 0/0.25/0.80, ws 0/6)"
SPEED_MOD_AMP=0 WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 0,1.09,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "nomod END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
kill_all; OUT=results/claudecodefortest/nomod_dh; rm -rf $OUT; mkdir -p $OUT
SPEED_MOD_AMP=0 WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="dhover3" \
timeout 1800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "nomod_dh END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"; touch results/claudecodefortest/NOMOD_DONE
# ② 연기된 펄스 ON25
for ONOFF in "25 5" "25 15"; do
  set -- $ONOFF; ON=$1; OFF=$2; W=$(windows $ON $OFF); NP=$(echo $W | tr ',' '\n' | wc -l)
  kill_all; OUT=results/claudecodefortest/pulse_on${ON}_off${OFF}; rm -rf $OUT; mkdir -p $OUT
  log "pulse ON=$ON OFF=$OFF N=$NP START  windows=$W"
  WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=280 SWEEP_ATK_WINDOWS="$W" CAPTURE_POLICIES="track,dhover3" \
  timeout 5400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases 3.314,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "pulse ON=$ON OFF=$OFF END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done
kill_all; touch results/claudecodefortest/PULSE_DONE; log "nomod+pulse 전체 완료"
