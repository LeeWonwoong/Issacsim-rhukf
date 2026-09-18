#!/usr/bin/env bash
# v33 (09-15 오후): 패널티 크기(blk50k k2·k4, mix50k k4) × 마르코프 날씨 체제(mkv50k) — GPU 60 런(SW·UKF·EKF) + Adam CPU 20 런, 시드 42–46. Isaac 유휴 확인 후.
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
[ "$(ps aux | grep -cE '[o]nline_rl_main.py')" -eq 0 ] || { echo "Isaac 실행 중 — GPU 동시 실행 금지, 중단"; exit 1; }
rm -f $N/v33*_w*.json $N/V33*_DONE
for w in $(seq 0 9); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v33.py $w 10 > $N/v33gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2 3; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v33.py $w 4 > $N/v33cpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v33 패널티 k2·k4 × 마르코프 체제 (GPU 10 워커 60 런 + CPU 4 워커 20 런) START" >> $N/v5_launch.log
