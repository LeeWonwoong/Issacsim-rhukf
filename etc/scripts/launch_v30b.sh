#!/usr/bin/env bash
# v30b (09-15 새벽): v31 완료 후 blk50k 시드 47–51 보강 (SW·UKF·EKF GPU 15 + Adam CPU 5) → v30 과 합쳐 시드 10.
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V31gpu_w.*_DONE')" -lt 8 ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] v31 완료 → v30b 착수"
rm -f $N/v30b*_w*.json $N/V30b*_DONE
for w in 0 1 2 3 4; do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v30b.py $w 5 > $N/v30bgpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v30b.py $w 2 > $N/v30bcpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v30b blk50k 시드 보강 (GPU 15 + CPU 5) START" >> $N/v5_launch.log
