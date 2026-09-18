#!/usr/bin/env bash
# run_windcrash.sh — benign(bias0) aggressive 에서 풍속별 크래시율 측정 (강풍 상한 결정용)
#   nohup bash run_windcrash.sh > windcrash.log 2>&1 &
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
for WS in 8 10 12 14; do
  OUT="results_windcrash_ws${WS}"
  [ -d "${OUT}" ] && rm -rf "${OUT}"
  mkdir -p "${OUT}"
  echo "==== [$(date +%H:%M:%S)] WS=${WS} START → ${OUT} ===="
  python3 -u online_rl_main.py --headless --agent adam --speed 10 --sweep \
      --sweep-mode torque --sweep-values 0.0 --sweep-pattern aggressive \
      --hover-delays 3 --episodes 12 \
      --sweep-wind-type wind_turbulence --sweep-wind-speed ${WS} \
      --outdir "${OUT}" > "${OUT}/run.log" 2>&1
  echo "==== [$(date +%H:%M:%S)] WS=${WS} DONE ===="
done
echo "WINDCRASH_ALL_DONE $(date +%H:%M:%S)" > windcrash_DONE
