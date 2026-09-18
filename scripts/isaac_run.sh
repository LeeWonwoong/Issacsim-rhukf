#!/usr/bin/env bash
# scripts/isaac_run.sh — Isaac 실행 래퍼: 남은 Isaac/PX4 정리 → ROS 환경 → train.py (설정은 전부 YAML)
#   setsid nohup bash scripts/isaac_run.sh <outdir> --config A.yaml --config configs/overlays/isaac.yaml [--set k=v ...] > /dev/null 2>&1 < /dev/null &
#   로그: <outdir>/train_stdout.log · 종료 표식 <outdir>/RUN_DONE (rc 기록)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
OUT=$1; shift
mkdir -p "$OUT"
set +u; source /opt/ros/humble/setup.bash 2>/dev/null || true; source ~/colcon_ws/install/setup.bash 2>/dev/null || true
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null || true; pkill -x px4 2>/dev/null || true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null || true; done
  for p in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null || true; done; sleep 6; }
kill_all
echo "[$(date '+%m-%d %H:%M')] START $*" >> "$OUT/run.log"
timeout ${ISAAC_TIMEOUT:-21600} python3 -u train.py "$@" --set "run.outdir=$OUT" > "$OUT/train_stdout.log" 2>&1
rc=$?
kill_all
echo "[$(date '+%m-%d %H:%M')] END rc=$rc eps=$(ls $OUT/capture 2>/dev/null | wc -l) err=$(grep -c Traceback $OUT/train_stdout.log)" >> "$OUT/run.log"
echo $rc > "$OUT/RUN_DONE"
