#!/usr/bin/env bash
# 기동 A/B 캡처 (2026-08-27): 같은 큰 박스(6.4m)·speed1·seed42, waypoint 프로파일만 다름.
#   A = 등속(구형, 사인 SPEED_MOD)   B = 기하 연동 가감속(코너 0.5↔직선 1.7)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=maneuver_ab.log
echo "════ maneuver A/B 시작 $(date '+%H:%M:%S') ════" | tee $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5
run_cap(){  # $1=tag $2=WP_CORNER_DECEL
  echo "[$(date '+%H:%M:%S')] zu_$1 START (WP_CORNER_DECEL=$2)" | tee -a $LOG
  kill_isaac; rm -rf results_zu_$1; mkdir -p results_zu_$1
  WP_CORNER_DECEL=$2 timeout 5400 python3 -u online_rl_main.py --headless --agent adam --max-ep 16 \
    --speed 1 --log-zu --seed 42 --outdir results_zu_$1 > results_zu_$1/cap.log 2>&1
  echo "[$(date '+%H:%M:%S')] zu_$1 DONE rc=$?" | tee -a $LOG
}
run_cap constA 0
run_cap decelB 1
kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch MANEUVER_AB_DONE
