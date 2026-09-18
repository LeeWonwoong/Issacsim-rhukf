#!/usr/bin/env bash
# run_seq_v5.sh — RHUKF v5 → Adam v5 순차(동시실행 불가). log(1+√NIS)clip3 압축.
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
# 방어: 시작 전 stray 정리
pkill -9 -f "online_rl_main.py" 2>/dev/null; pkill -9 -f "run_sim.py --headless" 2>/dev/null; sleep 3
for AG in rhukf adam; do
  OUT="results_${AG}_v5"
  [ -d "$OUT" ] && mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M%S)"; mkdir -p "$OUT"
  echo "[$(date '+%m-%d %H:%M:%S')] ${AG} v5 START → $OUT"
  python3 -u online_rl_main.py --headless --agent ${AG} --speed 10 --seed 0 --outdir "$OUT" > "$OUT/run.log" 2>&1
  echo "${AG}_V5_DONE rc=$? $(date '+%m-%d %H:%M:%S')" > "$OUT/DONE_MARKER"
  echo "[$(date '+%m-%d %H:%M:%S')] ${AG} v5 DONE"
  pkill -9 -f "run_sim.py --headless" 2>/dev/null; sleep 5   # 방어: sim 잔여 정리
done
echo "SEQ_V5_ALL_DONE $(date '+%m-%d %H:%M:%S')" > results_seq_v5_DONE
