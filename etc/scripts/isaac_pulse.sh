#!/usr/bin/env bash
# pulse (2026-09-11) — 펄스열 공격: δ{0.76,0.80} × (ON,OFF)∈{(5,5),(5,15),(15,5),(15,15),(25,5),(25,15)} × ws{0,6} × {track,dhover3}
#   펄스열은 180 스텝부터 시작해 280 까지 채움 (총 10 s 창). FF on · arm 0.05 · 5패턴 × 2ep → 480 에피 (~3h)
#   판정: (δ, ON, OFF) 별 track 사망률·사망 펄스 번호·lag, dhover3(첫 펄스 온셋+3 호버) 생존
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_pulse.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
windows(){ python3 -c "
on,off=$1,$2; t=180; w=[]
while t+on<=280: w.append(f'{t}-{t+on}'); t+=on+off
print(','.join(w))"; }
for ONOFF in "5 5" "5 15" "15 5" "15 15" "25 5" "25 15"; do
  set -- $ONOFF; ON=$1; OFF=$2; W=$(windows $ON $OFF); NP=$(echo $W | tr ',' '\n' | wc -l)
  kill_all; OUT=results/claudecodefortest/pulse_on${ON}_off${OFF}; rm -rf $OUT; mkdir -p $OUT
  log "pulse ON=$ON OFF=$OFF N=$NP START  windows=$W"
  WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=280 SWEEP_ATK_WINDOWS="$W" CAPTURE_POLICIES="track,dhover3" \
  timeout 5400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases 3.314,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "pulse ON=$ON OFF=$OFF END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done
kill_all; touch results/claudecodefortest/PULSE_DONE; log "pulse 전체 완료"
