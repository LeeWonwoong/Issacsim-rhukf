#!/usr/bin/env bash
# ext_d_launch (09-16 10:30): 진행 중인 ext_K·ext_P GPU 작업이 다 끝나면 ext_D(n-step 끄기 / 호버 확약 D10 / 둘 다 × SW·UKF·EKF × 시드 42–44) 착수. 11시 보고 런과 경합하지 않게 대기.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight; LOG=$O/ext_launch.log
export EXT_SW=SW EXT_ADAM=Adam1e-3 QSET=ext_D
while :; do
  left=$(JOBSET=gpu python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
n=0
for qs in ('ext_K','ext_P','ext_E'):
    n += sum(1 for j in Q.ext_jobs('gpu',qs) if not os.path.exists(f'{Q.Q}/{j[0]}.json'))
print(n)" 2>/dev/null)
  [ "${left:-1}" -eq 0 ] && break; sleep 60
done
echo "[$(date '+%m-%d %H:%M')] ext_K·ext_P·ext_E 완료 → ext_D(가치함수 몫 키우기) 착수" >> $LOG
for w in 0 1 2; do JOBSET=cpu OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= setsid nohup nice -n 12 python3 etc/scripts/tonight_queue.py work >> $O/Dcpu_w$w.log 2>&1 < /dev/null & sleep 1; done
for w in 0 1 2 3 4 5 6 7; do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 9 python3 etc/scripts/tonight_queue.py work >> $O/Dgpu_w$w.log 2>&1 < /dev/null & sleep 3; done
echo "[$(date '+%m-%d %H:%M')] ext_D 워커 GPU 8 + CPU 3" >> $LOG
