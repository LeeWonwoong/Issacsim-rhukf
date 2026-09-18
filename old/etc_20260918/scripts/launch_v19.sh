#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V18_w.*_DONE')" -lt 8 ]; do sleep 60; done
rm -f $N/v19_w*.json $N/V19_w*_DONE
# SW 15런: GPU 워커 8 (jobs 는 SW/Adam 섞여 있으므로 워커 전부 GPU 가시, Adam 은 1스레드라도 GPU 로 감 — 느리지만 소수)
for w in $(seq 0 11); do OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup python3 etc/scripts/env_scan_v19.py $w 12 > $N/v19_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v19 (5시드 확인 60런: 6셀×SW/Adam) START" >> $N/v5_launch.log
