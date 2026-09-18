#!/usr/bin/env bash
# cal-error 5% 캡처: UKF에 현실적 캘리브 오차(C_torque/C_thrust/mass 각 +5%) → 기동 gyro aliasing 유발.
# §6-2(clean)과 대조용. aggressive × ws{0,9}. 현재 config + 센서σ.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
export SENSOR_NOISE_SCALE=1.0
export SWEEP_ATTACK_TYPE=tilt
export SWEEP_ATK_START=100
export UKF_CALIB_ERR="ctq:0.05,cth:0.05,m:0.05"
rm -f results_calib5_DONE
for WS in 0 9; do
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  sleep 3
  OUT="results_calib5_ws${WS}"
  if [ -d "$OUT" ]; then rm -rf "$OUT"; fi
  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] cal5 ws=${WS} START (UKF_CALIB_ERR 5%)"
  python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --sweep-values 2.6 \
      --sweep-pattern aggressive --sweep-wind-type wind_turbulence --sweep-wind-speed ${WS} \
      --episodes 10 --speed 10 --outdir "$OUT" > "$OUT/run.log" 2>&1
  echo "[$(date '+%H:%M:%S')] cal5 ws=${WS} DONE rc=$?"
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  sleep 5
done
echo "CALIB5_ALL_DONE $(date '+%H:%M:%S')" > results_calib5_DONE
