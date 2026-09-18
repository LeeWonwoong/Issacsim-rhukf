#!/usr/bin/env bash
# run_rhukf_v5.sh — RHUKF, v5 config (log(1+√NIS)clip3 압축), 입력~3 안정성 실측
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
OUT="results_rhukf_v5"
[ -d "$OUT" ] && mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M%S)"; mkdir -p "$OUT"
echo "[$(date '+%m-%d %H:%M:%S')] RHUKF v5 START → $OUT (log(1+√NIS)clip3, 300ep)"
python3 -u online_rl_main.py --headless --agent rhukf --speed 10 --seed 0 --outdir "$OUT" > "$OUT/run.log" 2>&1
echo "RHUKF_V5_DONE rc=$? $(date '+%m-%d %H:%M:%S')" > "$OUT/DONE_MARKER"
