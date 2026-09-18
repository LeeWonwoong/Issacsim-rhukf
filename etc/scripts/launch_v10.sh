#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
while [ "$(ls results/claudecodefortest/night | grep -c 'V9_w.*_DONE')" -lt 8 ] || [ "$(ls results/claudecodefortest/night | grep -c 'V9b_w.*_DONE')" -lt 6 ]; do sleep 60; done
rm -f results/claudecodefortest/night/v10_w*.json results/claudecodefortest/night/V10_w*_DONE
for w in 0 1 2 3 4 5 6 7; do setsid nohup python3 etc/scripts/env_scan_v10.py $w 8 > results/claudecodefortest/night/v10_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] env_scan_v10 (A2 vs R0, 250ep) 8 workers START (20 runs)" >> results/claudecodefortest/night/v5_launch.log
