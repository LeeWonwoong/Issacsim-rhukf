#!/usr/bin/env bash
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
pkill -9 -f "run_sim.py --headless" 2>/dev/null; sleep 2
OUT="results_zu_noisy"; [ -d "$OUT" ] && mv "$OUT" "${OUT}_old_$(date +%H%M%S)"; mkdir -p "$OUT"
export SENSOR_NOISE_SCALE=1.0
echo "[$(date '+%H:%M:%S')] 노이즈 zu 캡처 START (adam, 40ep, log-zu, SENSOR_NOISE_SCALE=1)"
python3 -u online_rl_main.py --headless --agent adam --speed 10 --seed 0 --max-ep 40 --log-zu --outdir "$OUT" > "$OUT/run.log" 2>&1
echo "ZU_NOISY_DONE rc=$? $(date '+%H:%M:%S')" > "$OUT/DONE_MARKER"
pkill -9 -f "run_sim.py --headless" 2>/dev/null
