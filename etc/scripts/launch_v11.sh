#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
while [ "$(ls results/claudecodefortest/night | grep -c 'V10_w.*_DONE')" -lt 8 ]; do sleep 60; done
rm -f results/claudecodefortest/night/v11_w*.json results/claudecodefortest/night/V11_w*_DONE
for w in 0 1 2 3 4 5; do setsid nohup python3 etc/scripts/env_scan_v11.py $w 6 > results/claudecodefortest/night/v11_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] env_scan_v11 (A2 lump-sum coeff, 250ep) 6 workers START (12 runs)" >> results/claudecodefortest/night/v5_launch.log
