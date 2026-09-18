#!/usr/bin/env bash
# bprime — B′ 검증: INT_LIM 1.0 (트림 상한 해제) · ramp2.5 · δ{0.85,0.95,1.05} · hold 15s · 무풍
#   판정: ① dhover 가 setpoint 사수하나(절대끌림 <1m) ② track 이 경로이탈/기동실패하나 (결과성 잔존?)
#         ③ PX4 가 다 보상해버려 둘 다 무사하면 → B′ 기각
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
export MC_INT_LIM=1.0
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_bprime; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] bprime START (INT_LIM=1.0, ramp2.5, δ0.85/0.95/1.05, hold15s)" | tee isaac_bprime.log
SWEEP_ATK_START=180 SWEEP_ATK_END=330 CAPTURE_POLICIES="track,dhover3" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 \
  --capture-biases 3.706,4.142,4.578 --ramp 2.5 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] bprime END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_bprime.log
kill_all; touch BPRIME_DONE
