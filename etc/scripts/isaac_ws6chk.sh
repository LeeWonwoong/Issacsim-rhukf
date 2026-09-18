#!/usr/bin/env bash
# δ_max 정밀: v3.2 환경(속도1.0·acc4.0·arm0.07), 최악 셀(ws8)에서
#   "track 확실 사망 ∧ 호버 확실 생존" 인 상한을 찾는다. δ{0.72,0.75,0.78,0.80} × 3정책 × 4ep
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 MPC_ACC_HOR=4.0
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all
OUT=results_ws6chk; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] ws6chk START" | tee isaac_ws6chk.log
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns aggressive,waypoint,figure8 \
  --capture-disturbances wind_turbulence:6 \
  --capture-biases 3.270,3.488 --episodes 4 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] ws6chk END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_ws6chk.log
kill_all; touch WS6CHK_DONE
