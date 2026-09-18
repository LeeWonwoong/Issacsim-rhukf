#!/usr/bin/env bash
# ext_launch (09-16 04:40 자동 연장 결정, night/EXT_DECISION_0916.md): Isaac SWIRL 재실행 끝(ISAAC_RERUN_DONE)을 기다린 뒤
#   CPU 워커 4(ext_G Adam) + GPU 워커를 총 10(대기열 워커 전부 합산, 정지 상태 제외)으로 채운다. STOP_CLAIMS 또는 할 일 0 이면 종료.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight; LOG=$O/ext_launch.log
export EXT_SW=SW EXT_ADAM=Adam1e-3 QSET=ext_G
echo "[$(date '+%m-%d %H:%M')] 대기 시작 (ISAAC_RERUN_DONE)" >> $LOG
while [ ! -f results/claudecodefortest/ISAAC_RERUN_DONE ]; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] Isaac 재실행 끝 → CPU 워커 4 착수" >> $LOG
for w in 0 1 2 3; do JOBSET=cpu OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= setsid nohup nice -n 10 python3 etc/scripts/tonight_queue.py work >> $O/Xcpu_w$w.log 2>&1 < /dev/null & done
while :; do
  [ -f $O/q/STOP_CLAIMS ] && { echo "[$(date '+%m-%d %H:%M')] STOP_CLAIMS → 종료" >> $LOG; break; }
  left=$(JOBSET=gpu python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
print(sum(1 for j in Q.ext_jobs('gpu','ext_G') if not os.path.exists(f'{Q.Q}/{j[0]}.json') and not os.path.exists(f'{Q.Q}/claims/{j[0]}')))" 2>/dev/null)
  [ "${left:-0}" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M')] ext_G GPU 할 일 0 → 종료" >> $LOG; break; }
  n=0
  for p in $(pgrep -f "tonight_queue.py work"); do
    st=$(ps -o stat= -p $p 2>/dev/null); case "$st" in T*|Z*|"") continue;; esac
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx 'JOBSET=gpu' && n=$((n+1))
  done
  if [ $n -lt 10 ]; then
    JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/tonight_queue.py work >> $O/Xgpu_$(date +%H%M%S).log 2>&1 < /dev/null &
    echo "[$(date '+%m-%d %H:%M')] ext_G GPU 워커 추가 (실행 중 $n → +1, 남은 작업 $left)" >> $LOG; sleep 8; continue
  fi
  sleep 60
done
