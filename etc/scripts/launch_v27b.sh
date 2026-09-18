#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V26gpu_w.*_DONE')" -lt 10 ]; do sleep 60; done
rm -f $N/v27b*_w*.json $N/V27b*_DONE
for w in $(seq 0 9); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v27b.py $w 10 > $N/v27bgpu_w$w.log 2>&1 < /dev/null & done
for w in $(seq 0 4); do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v27b.py $w 5 > $N/v27bcpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v27b 드문·긴 난기류 임펄스 (GPU 25 + CPU 15) START" >> $N/v5_launch.log
