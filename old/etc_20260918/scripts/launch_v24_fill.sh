#!/usr/bin/env bash
# v24 완료 후 실패(예외) 셀만 재실행 (json 에 없는 이름만 돌아감; UKF/EKF P 복구 패치 적용)
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V24_w.*_DONE')" -lt 12 ]; do sleep 60; done
rm -f $N/V24_w*_DONE
for w in $(seq 0 11); do OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v24.py $w 12 > $N/v24f_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v24 실패 셀 재실행 (P 복구 패치)" >> $N/v5_launch.log
