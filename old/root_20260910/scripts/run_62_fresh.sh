#!/usr/bin/env bash
# §6-2 fresh 재측정: 현재 config(틸트 δ0.6 / log(1+√)clip3 / 실기 σ) × 바람 5레벨, 순차(Isaac 1개).
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
export SENSOR_NOISE_SCALE=1.0
export SWEEP_ATTACK_TYPE=tilt
export SWEEP_ATK_START=100
rm -f results_62_DONE
for WS in 0 5 9 12 15; do
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  sleep 3
  OUT="results_62_ws${WS}"
  if [ -d "$OUT" ]; then rm -rf "$OUT"; fi
  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] ws=${WS} START"
  python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --sweep-values 2.6 \
      --sweep-pattern aggressive --sweep-wind-type wind_turbulence --sweep-wind-speed ${WS} \
      --episodes 10 --speed 10 --outdir "$OUT" > "$OUT/run.log" 2>&1
  echo "[$(date '+%H:%M:%S')] ws=${WS} DONE rc=$?"
  pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
  sleep 5
done
echo "SEC62_FRESH_ALL_DONE $(date '+%H:%M:%S')" > results_62_DONE
