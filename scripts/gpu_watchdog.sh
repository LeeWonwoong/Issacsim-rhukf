#!/usr/bin/env bash
# gpu_watchdog.sh <grid이름조각> — Isaac SWIRL 런(learn-step 예산 40 ms)이 도는 동안은 격자의 GPU 워커·cuda 학습 프로세스를 SIGSTOP(경합 시 16→40 ms 실측),
#   Isaac SWIRL 이 없을 때(Adam 창·체인 종료)만 SIGCONT. CPU 워커는 건드리지 않는다.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
G=$1; LOG=results/claudecodefortest/watchdog_$G.log; STOPPED=0
gpids(){ ps -eo pid,args | grep -E "grid.py worker configs/grids/$G.yaml [a-z]+ gpu|train.py --config .*$G.*run.device=cuda|train.py --config .*run.device=cuda.*$G" | grep -v grep | awk '{print $1}'; }
while true; do
  ISAAC=$(ps -eo args | grep "train.py --config configs/isaac_v5_swirl" | grep -v grep | wc -l)
  PIDS=$(gpids)
  if [ "$ISAAC" -gt 0 ]; then
    if [ $STOPPED = 0 ] && [ -n "$PIDS" ]; then kill -STOP $PIDS 2>/dev/null; STOPPED=1; echo "$(date '+%H:%M:%S') STOP (Isaac SWIRL 실행 중) pids: $(echo $PIDS | tr '\n' ' ')" >> $LOG; fi
  else
    if [ $STOPPED = 1 ]; then kill -CONT $PIDS 2>/dev/null; STOPPED=0; echo "$(date '+%H:%M:%S') CONT (Isaac SWIRL 없음)" >> $LOG; fi
  fi
  [ -f results/claudecodefortest/$G/GRID_DONE ] && { kill -CONT $PIDS 2>/dev/null; echo "$(date '+%H:%M:%S') 격자 완료 → 종료" >> $LOG; exit 0; }
  sleep 20
done
