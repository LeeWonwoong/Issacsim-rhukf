#!/usr/bin/env bash
# ext_h_launch (09-16 08:05): ext_G GPU 작업이 다 선점되면 dual200(200 ep, 시드 44–45 × SW·UKF·EKF) 를 띄운다. 08:55 배치 64 와 GPU 를 나눠 쓰므로 워커는 6 으로 제한.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight; LOG=$O/ext_launch.log
export EXT_SW=SW EXT_ADAM=Adam1e-3 QSET=ext_H
while :; do
  left=$(JOBSET=gpu python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
print(sum(1 for j in Q.ext_jobs('gpu','ext_G') if not os.path.exists(f'{Q.Q}/{j[0]}.json') and not os.path.exists(f'{Q.Q}/claims/{j[0]}')))" 2>/dev/null)
  [ "${left:-1}" -eq 0 ] && break; sleep 60
done
echo "[$(date '+%m-%d %H:%M')] ext_G 선점 완료 → ext_H(dual200) 착수" >> $LOG
for w in 0 1; do JOBSET=cpu OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= setsid nohup nice -n 10 python3 etc/scripts/tonight_queue.py work >> $O/Hcpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2 3 4 5; do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/tonight_queue.py work >> $O/Hgpu_w$w.log 2>&1 < /dev/null & sleep 3; done
echo "[$(date '+%m-%d %H:%M')] ext_H 워커 GPU 6 + CPU 2" >> $LOG
