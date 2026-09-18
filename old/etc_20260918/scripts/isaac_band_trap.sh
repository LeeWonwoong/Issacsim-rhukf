#!/usr/bin/env bash
# band_trap (측정1) — 학습용 사다리꼴 공격으로 무풍 밴드. rise2.5s(25)+hold4s(40)=창 180-245.
#   δ_end {0.78,0.80,0.82,0.84} × 무풍 × 전5패턴 × track/dhover3 × 5ep. headless s2.5
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_band_trap.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_band_trap; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] band_trap START (사다리꼴 rise2.5+hold4, 무풍, δ0.78-0.84)" | tee $LOG
# 사다리꼴: --ramp 2.5 (rise) + 공격창 180-245 (rise25+hold40). 무풍.
SWEEP_ATK_START=180 SWEEP_ATK_END=245 CAPTURE_POLICIES="track,dhover3" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances none:0 \
  --capture-biases 3.401,3.488,3.575,3.662 --ramp 2.5 --episodes 5 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] band_trap END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
kill_all; touch BAND_TRAP_DONE
