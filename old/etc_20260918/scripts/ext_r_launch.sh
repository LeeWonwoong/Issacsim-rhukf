#!/usr/bin/env bash
# ext_D(확약·n-step) GPU 작업이 다 끝나면 ext_R(잡음·aliasing 스트레스 스윕) 착수. 보류(claim only) 작업은 남은 것으로 세지 않는다.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight
export EXT_SW=SW EXT_ADAM=Adam1e-3 QSET=ext_R
while :; do
  left=$(JOBSET=gpu python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
print(sum(1 for j in Q.ext_jobs('gpu','ext_D') if not os.path.exists(f'{Q.Q}/{j[0]}.json') and not os.path.exists(f'{Q.Q}/claims/{j[0]}')))" 2>/dev/null)
  run=$(pgrep -f "tonight_queue.py work" | wc -l)
  [ "${left:-1}" -eq 0 ] && [ "${run:-1}" -le 1 ] && break
  sleep 120
done
echo "[$(date '+%m-%d %H:%M')] ext_D 완료 → ext_R(잡음·aliasing 스윕) 착수" >> $O/ext_launch.log
for w in 0 1 2; do JOBSET=cpu OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= setsid nohup nice -n 12 python3 etc/scripts/tonight_queue.py work >> $O/Rcpu_w$w.log 2>&1 & sleep 1; done
for w in 0 1 2 3 4 5 6 7 8 9; do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 9 python3 etc/scripts/tonight_queue.py work >> $O/Rgpu_w$w.log 2>&1 & sleep 3; done
echo "[$(date '+%m-%d %H:%M')] ext_R 워커 GPU 10 + CPU 3" >> $O/ext_launch.log
