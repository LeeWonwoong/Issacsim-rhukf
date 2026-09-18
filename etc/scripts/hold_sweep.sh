#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=hold_sweep.log; echo "════ hold_sweep 시작 $(date '+%H:%M:%S') ════" | tee $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5
export EP_MAX_STEPS=400 SWEEP_ATK_START=150 SWEEP_ATK_END=250 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES=whover2
for EM in 0 1.0 1.5 2.5 1000000; do
  echo "[$(date '+%H:%M:%S')] EMAX=$EM START" | tee -a $LOG
  kill_isaac; rm -rf results_hold_$EM; mkdir -p results_hold_$EM
  HOVER_EMAX=$EM timeout 2400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
    --capture-mode hijack --capture-patterns waypoint --capture-disturbances none:0 \
    --capture-biases 2.616,3.488 --episodes 1 --speed 1 --log-zu \
    --outdir results_hold_$EM > results_hold_$EM/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] EMAX=$EM DONE rc=$?" | tee -a $LOG
done
kill_isaac; echo "════ 완료 ════" | tee -a $LOG; touch HOLD_SWEEP_DONE
