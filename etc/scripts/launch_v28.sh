#!/usr/bin/env bash
# v28 (09-14 낮): 돌풍 gU + 대조 nU × {SWa, SWb, UKF-TD, EKF-TD}(GPU 24) + Adam 3e-4(CPU 6) × seed 42–44, 150ep. Isaac 캡처와 GPU 공유 → 워커 6.
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
rm -f $N/v28*_w*.json $N/V28*_DONE
for w in $(seq 0 5); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v28.py $w 6 > $N/v28gpu_w$w.log 2>&1 < /dev/null & done
for w in $(seq 0 2); do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v28.py $w 3 > $N/v28cpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v28 돌풍 무대 시드3 × {SWa,SWb,Adam3e-4,UKF,EKF} (GPU 24 + CPU 6) START" >> $N/v5_launch.log
