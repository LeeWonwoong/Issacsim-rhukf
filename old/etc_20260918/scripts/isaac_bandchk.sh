#!/usr/bin/env bash
# bandchk — 3확인점 (δ0.79·ws8, δ0.8·ws7, δ0.8·ws9) × 전5패턴 × track/dhover3 × 4ep
#   기준: 전 패턴 호버 생존 ∧ track 사망. 바람창(안정화후). 이후 ②~⑤ 원 체인 이어짐.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_bandchk.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
run(){ # $1=bias $2=ws $3=tag
  kill_all; OUT=results_bandchk_$3; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] $3 (δ$(python3 -c "print(round($1/4.36,2))") ws$2) START" | tee -a $LOG
  WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances wind_turbulence:$2 \
    --capture-biases $1 --episodes 4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $3 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
}
run 3.444 8 d79ws8    # δ0.79 ws8
run 3.488 7 d80ws7    # δ0.80 ws7
run 3.488 9 d80ws9    # δ0.80 ws9
kill_all; touch BANDCHK_DONE
