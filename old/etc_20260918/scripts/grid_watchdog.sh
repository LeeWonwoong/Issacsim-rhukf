#!/usr/bin/env bash
# 워치독: batch3b 가 죽으면 자동 재시작 (batch3b 는 완료런 skip). 9런 다 되면 종료.
cd /home/acsl/projects/Issacsim-rhukf
LOG=grid_watchdog.log
while true; do
  done=0
  for r in A_r10_pd001 A_r10_pd005 A_r15_pd001 A_r15_pd005 B_r10_pd001 B_r10_pd005 B_r15_pd001 B_r15_pd005 adam_n16; do
    [ -f results_g1_$r/RUN_DONE ] && done=$((done+1))
  done
  if [ "$done" -ge 9 ]; then echo "[$(date '+%m-%d %H:%M')] 9/9 완료 — 워치독 종료" >> $LOG; break; fi
  if ! pgrep -f "tune_batch3b.sh" >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] batch3b 죽음(완료 $done/9) → 재시작" >> $LOG
    for p in $(pgrep -f "python.sh run_sim"); do kill -9 $p 2>/dev/null; done
    for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null; done
    sleep 5
    setsid nohup bash etc/scripts/tune_batch3b.sh > /tmp/batch3b.log 2>&1 < /dev/null &
  fi
  sleep 180
done
