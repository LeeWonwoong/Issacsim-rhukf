#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V22b_w.*_DONE')" -lt 10 ]; do sleep 60; done
rm -f $N/v23_w*.json $N/V23_w*_DONE
for w in $(seq 0 9); do OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v23.py $w 10 > $N/v23_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v23 (n-step1 / D0 ablation × SW/Adam × 5시드, 30런) START" >> $N/v5_launch.log
