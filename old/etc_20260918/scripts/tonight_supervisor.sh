#!/usr/bin/env bash
# tonight 감독: ① tonight_hm GPU 워커가 칼만-TD(hm UKF/EKF) 작업으로 넘어가면 종료(사용자 23:20: KTD 대신 지속시간 스윕)
#   ② GPU 워커 총수(tonight_hm + tonight_queue)를 10 으로 유지하도록 대기열 워커를 띄운다. 대기열이 비면 종료.
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight; LOG=$O/supervisor.log
echo "[$(date '+%m-%d %H:%M')] 감독 시작" >> $LOG
while :; do
  hm=0; qn=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in
      "python3 etc/scripts/tonight_hm.py work "*" 10 ")
        e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E '^(JOBSET|TONIGHT_SET)=' | paste -sd,)
        case "$e" in *TONIGHT_SET=*|*JOBSET=cpu*) continue;; esac
        W=$(echo "$c" | awk '{print $4}')
        nxt=$(python3 -c "
import json,sys; sys.path.insert(0,'etc/scripts'); import tonight_hm as T
jb=T.build('gpu')[$W::10]
try: d=json.load(open('$O/Tgpu_w$W.json'))
except Exception: d={}
k=sum(1 for j in jb if j[0] in d); print(jb[k][6] if k < len(jb) else 'END')" 2>/dev/null)
        if [ "$nxt" = "UKFp03" ] || [ "$nxt" = "EKFp03" ]; then kill $p; echo "[$(date '+%m-%d %H:%M')] tonight w$W 칼만-TD 작업 진입 → 종료" >> $LOG; else hm=$((hm+1)); fi ;;
      "python3 etc/scripts/tonight_queue.py work"*)
        e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^JOBSET=' ); [ "$e" = "JOBSET=gpu" ] && qn=$((qn+1)) ;;
    esac
  done 2>/dev/null
  left=$(JOBSET=gpu python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
print(sum(1 for j in Q.jobs('gpu') if not os.path.exists(f'{Q.Q}/{j[0]}.json') and not os.path.exists(f'{Q.Q}/claims/{j[0]}')))" 2>/dev/null)
  if [ "${left:-0}" -gt 0 ] && [ $((hm+qn)) -lt 10 ]; then
    JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/tonight_queue.py work >> $O/Qgpu_$(date +%H%M%S).log 2>&1 < /dev/null &
    echo "[$(date '+%m-%d %H:%M')] 대기열 GPU 워커 추가 (hm $hm + queue $qn → +1, 남은 작업 $left)" >> $LOG
    sleep 5; continue
  fi
  if [ "${left:-0}" -eq 0 ] && [ $((hm+qn)) -eq 0 ]; then echo "[$(date '+%m-%d %H:%M')] 모든 GPU 작업 종료 — 감독 종료" >> $LOG; break; fi
  sleep 30
done
