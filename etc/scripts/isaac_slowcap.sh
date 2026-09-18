#!/usr/bin/env bash
# slowcap (2026-09-11) — 궤적 속도 −15% (TRAJ_SCALE=0.85, PX4 캡 불변) 에서 밴드·aliasing·추종 재측정. arm 0.05.
#   δ{0,0.25,0.74,0.76,0.78,0.80} × ws{0,6} × {track,dhover3} × 5패턴 × 2ep × ON30 = 240에피 (~1.6h), zu 로그 → E/G2 분석
#   비교 기준: results_bandcap(ws0) / bandcap_arm005(ws6) — 속도 1.0
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f results/claudecodefortest/DH1CHECK_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 TRAJ_SCALE=0.85 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results/claudecodefortest/slowcap; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] slowcap START (TRAJ_SCALE 0.85, arm 0.05)" | tee results/claudecodefortest/isaac_slowcap.log
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 0,1.090,3.226,3.314,3.401,3.488 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] slowcap END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a results/claudecodefortest/isaac_slowcap.log
(setsid nohup python3 etc/scripts/analyze_bandcap.py $OUT 30 > $OUT/analysis.log 2>&1 < /dev/null &)
kill_all; touch results/claudecodefortest/SLOWCAP_DONE
