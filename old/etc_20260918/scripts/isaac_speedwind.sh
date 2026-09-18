#!/usr/bin/env bash
# speedwind — δ0.8 고정 × {현속도, 5%감속} × {무풍, ws8, ws10} × 전패턴 × track/dhover3 × 4ep
#   목적: 밴드(호버 전패턴 생존) 유지되는 최대 바람 확정 + 무풍 δ0.8 결과성 확인 + 5%감속 효과
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f CHAIN0908B_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_speedwind.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
run(){ # $1=SPEED_SCALE $2=dist $3=tag
  kill_all; OUT=results_sw_$3; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] $3 (scale$1 $2) START" | tee -a $LOG
  SPEED_SCALE=$1 WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
  timeout 9000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances $2 \
    --capture-biases 3.488 --episodes 4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $3 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
}
run 1.0  none:0            cur_none
run 1.0  wind_turbulence:8 cur_ws8
run 1.0  wind_turbulence:10 cur_ws10
run 0.95 none:0            slow_none
run 0.95 wind_turbulence:8 slow_ws8
run 0.95 wind_turbulence:10 slow_ws10
kill_all; touch SPEEDWIND_DONE
