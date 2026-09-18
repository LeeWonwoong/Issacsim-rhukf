#!/usr/bin/env bash
# v34 (09-15 저녁 큐): v33 GPU 완료(DONE 10) ∧ Isaac 유휴 대기 → blk50k 튜닝 전이·Huber c/R·공정성 (GPU 100 런 10 워커 + CPU Adam 10 런).
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
while [ "$(ls $N | grep -c 'V33gpu_w.*_DONE')" -lt 10 ] || [ "$(ps aux | grep -cE '[o]nline_rl_main.py')" -gt 0 ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] v33 완료 → v34 착수"
rm -f $N/v34*_w*.json $N/V34*_DONE
for w in $(seq 0 9); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v34.py $w 10 > $N/v34gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v34.py $w 2 > $N/v34cpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v34 튜닝 전이·Huber c/R·공정성 (GPU 10 워커 100 런 + CPU 2 워커 10 런) START" >> $N/v5_launch.log
