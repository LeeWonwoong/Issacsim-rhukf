#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V19_w.*_DONE')" -lt 12 ]; do sleep 60; done
for w in $(seq 0 7); do OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v21b.py $w 8 > $N/v21b_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v21b (SWIRL hover탐험0.15 × 3셀 × 5시드) START" >> $N/v5_launch.log
