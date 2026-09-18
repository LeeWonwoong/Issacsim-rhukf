#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V15_w.*_DONE')" -lt 12 ]; do sleep 60; done
rm -f $N/v16_w*.json $N/V16_w*_DONE
for w in $(seq 0 8); do OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v16.py $w 9 > $N/v16_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v16(프레임워크 200ep×9) START" >> $N/v5_launch.log
sleep 90; D=$(sed -n 's/.* D=\([0-9]*\).*/\1/p' results/claudecodefortest/COMMIT_CFG 2>/dev/null); D=${D:-5}
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" python3 etc/scripts/surr_cusum_eval.py $D 300 0.02 > $N/cusum_D$D.log 2>&1
echo "[$(date '+%m-%d %H:%M')] CUSUM 베이스라인 D=$D 완료" >> $N/v5_launch.log
