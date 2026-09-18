#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V21b_w.*_DONE')" -lt 8 ]; do sleep 60; done
rm -f $N/v22b_w*.json $N/V22b_w*_DONE
for w in $(seq 0 9); do OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v22b.py $w 10 > $N/v22b_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v22b (체인 가족 × SW/Adam × 5시드, 30런) START" >> $N/v5_launch.log
