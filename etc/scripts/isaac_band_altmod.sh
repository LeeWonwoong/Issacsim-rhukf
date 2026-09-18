#!/usr/bin/env bash
# band_altmod (측정1.5) — 측정1과 동일 (사다리꼴·무풍·δ0.78-0.84) + 고도변조 ±0.5m.
#   판정: ① vel 채널 NIS 상승(aliasing) ② 밴드(추락) 영향 (고도변조가 밴드 오염하나)
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f BAND_TRAP_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
export ALT_MOD_AMP=0.5 ALT_MOD_FREQ=0.3   # ★고도 ±0.5m sine
LOG=isaac_band_altmod.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_band_altmod; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] band_altmod START (고도변조 ±0.5m, 사다리꼴 무풍 δ0.78-0.84)" | tee $LOG
SWEEP_ATK_START=180 SWEEP_ATK_END=245 CAPTURE_POLICIES="track,dhover3" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances none:0 \
  --capture-biases 3.401,3.488,3.575,3.662 --ramp 2.5 --episodes 5 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] band_altmod END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
kill_all; touch BAND_ALTMOD_DONE
