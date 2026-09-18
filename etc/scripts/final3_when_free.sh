#!/usr/bin/env bash
# P₀ 스크린(tuneq) GPU 5런이 끝나면 final3/final3g GPU 6런 착수. 동시 실행으로 서로 느려지지 않게.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight
while :; do
  n=$(ls $O/q 2>/dev/null | grep -c '^tuneq_.*json$')
  [ "${n:-0}" -ge 5 ] && break
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] tuneq 완료 → final3/final3g GPU 6런 착수" >> $O/ext_launch.log
for w in 0 1 2 3 4 5; do QSET=final3 EXT_SW=SW JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 5 python3 etc/scripts/tonight_queue.py work >> $O/F3gpu_w$w.log 2>&1 & sleep 3; done
