#!/usr/bin/env bash
# 1시간마다 자가질문: 진행·프로세스·에러 3중 점검 — 이상 시 즉시 종료(=Claude 호출)
cd /home/acsl/projects/Issacsim-rhukf
prev=0
while true; do
  sleep 3600
  [ -f FINAL3_DONE ] && { echo "[watchdog] final3 완료 — 정상 종료"; exit 0; }
  cur=$(cat results_fin_*/metrics_*.csv 2>/dev/null | wc -l)
  np=$(pgrep -cf online_rl_main)
  err=$(tail -50 results_fin_*/train.log 2>/dev/null | grep -cE "Traceback|HARD_RESET|Timeout 150s")
  echo "[watchdog $(date '+%H:%M')] Q1 진행? 행=$cur(직전 $prev) | Q2 프로세스? $np | Q3 에러? $err"
  if [ "$cur" -le "$prev" ] || [ "$np" -eq 0 ] || [ "$err" -ge 3 ]; then
    echo "[watchdog] ★이상 감지 — 로그 꼬리:"
    tail -3 isaac_final3.log 2>/dev/null
    exit 1
  fi
  prev=$cur
done
