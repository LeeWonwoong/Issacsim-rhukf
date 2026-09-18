#!/usr/bin/env bash
# bandcap (2026-09-10 사용자 지정) — FF on(현 기본) 무대에서 밴드 확정 + E/G2 오프라인 비교용 zu 캡처
#   5패턴 × ws{0, 6} × δ{0, 0.25, 0.74, 0.76, 0.78, 0.80, 0.82} × {track, dhover3} × 2ep = 280에피 (~1.7h)
#   공격 ON 30스텝(3 s, 180~210) · 바람창 60~290 · EP 300 · --log-zu (E vs G2 재생) · ref_x/y/alt 기록(3D 플롯)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_bandcap; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] bandcap START (ACCEL_FF=$ACCEL_FF agg_phase 5.6)" | tee isaac_bandcap.log
# δ→bias(×4.36): 0→0  0.25→1.090  0.74→3.226  0.76→3.314  0.78→3.401  0.80→3.488  0.82→3.575
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 0,1.090,3.226,3.314,3.401,3.488,3.575 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] bandcap END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_bandcap.log
kill_all; touch BANDCAP_DONE
