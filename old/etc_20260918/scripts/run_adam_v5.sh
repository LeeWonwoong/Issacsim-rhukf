#!/usr/bin/env bash
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
OUT="results_adam_v5"
[ -d "$OUT" ] && mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M%S)"; mkdir -p "$OUT"
echo "[$(date '+%m-%d %H:%M:%S')] ADAM v5 START → $OUT"
python3 -u online_rl_main.py --headless --agent adam --speed 10 --seed 0 --outdir "$OUT" > "$OUT/run.log" 2>&1
echo "ADAM_V5_DONE rc=$? $(date '+%m-%d %H:%M:%S')" > "$OUT/DONE_MARKER"
