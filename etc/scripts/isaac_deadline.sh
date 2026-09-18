#!/usr/bin/env bash
# deadline (2026-09-12) — 강공격 반응 데드라인 곡선: dhover d∈{5,10,15,20,25} × δ{0.80,0.84} × 5패턴 × ws{0,6} × 2ep = 200
#   FS2 판정 후 FS_PARAMS.env(있으면) 의 failsafe 파라미터 사용. 새 규칙(각도 종료 없음), hover drift 대칭.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_cert.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ ! -f results/claudecodefortest/FS2_DONE ]; do sleep 30; done
kill_all; OUT=results/claudecodefortest/cert_deadline; rm -rf $OUT; mkdir -p $OUT; log "DEADLINE START ($FAILSAFE_PARAMS)"
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="dhover5,dhover10,dhover15,dhover20,dhover25" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 3.488,3.662 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "DEADLINE END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback' $OUT/run.log)"; kill_all; touch results/claudecodefortest/DEADLINE_DONE
