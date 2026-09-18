#!/usr/bin/env bash
# v13/v13b/v14/v14b 전부 완료 후 v15(약속 D 스캔 48런) 16 워커 시작
cd /home/acsl/projects/Issacsim-rhukf
N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V13_w.*_DONE')" -lt 12 ] || [ "$(ls $N | grep -c 'V13b_w.*_DONE')" -lt 12 ] || [ "$(ls $N | grep -c 'V14_w.*_DONE')" -lt 15 ] || [ "$(ls $N | grep -c 'V14b_w.*_DONE')" -lt 12 ]; do sleep 60; done
rm -f $N/v15_w*.json $N/V15_w*_DONE
for w in $(seq 0 23); do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v15.py $w 24 > $N/v15_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] env_scan_v15 (약속 D 3/5/10/20 × R0/A3 × 3opt × 2seed, hover탐험0.05, 200ep) 24 workers START (111 runs)" >> $N/v5_launch.log
