#!/usr/bin/env bash
# bandcap_on (2026-09-10) — bandcap(ON30) 과 같은 격자를 ON5·ON15 로. weakfloor 뒤 자동.
#   5패턴 × ws{0,6} × δ{0.25,0.74,0.76,0.78,0.80,0.82} × {track,dhover3} × 2ep = 240에피/ON (~1.5h). δ0 은 ON 무관이라 생략.
#   출력: results/claudecodefortest/bandcap_on{5,15}/ (+ 자동 분석)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_bandcap_on.log
for ON in 15; do
  kill_all; OUT=results/claudecodefortest/bandcap_on$ON; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] bandcap ON=$ON START" | tee -a $LOG
  WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=$((180+ON)) CAPTURE_POLICIES="track,dhover3" \
  timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases 1.090,3.226,3.314,3.401,3.488,3.575 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] bandcap ON=$ON END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
  (setsid nohup python3 etc/scripts/analyze_bandcap.py $OUT $ON > $OUT/analysis.log 2>&1 < /dev/null &)
done
kill_all; touch results/claudecodefortest/BANDCAP_ON_DONE; echo "[$(date '+%m-%d %H:%M')] bandcap_on 완료" | tee -a $LOG
