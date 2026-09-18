#!/usr/bin/env bash
# sweetspot_sweep (2026-09-02): 바람 티어 스윗스팟 탐색.
#   arm{0.05,0.10} × ws{6,9} × 4패턴 × δ{0.6,0.7}(=Nm 2.616,3.052) × {track,hover}, 4ep/셀, speed3.
#   목적: hover 생존(조건③) 회복하면서 gyro aliasing 유지되는 (arm,ws) 찾기.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=sweetspot.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }

for ARM in 0.05 0.10; do
  OUT=results_sweetspot_a${ARM}
  say "$OUT START (arm=$ARM, ws{6,9}, δ{0.6,0.7}, 4패턴, track+hover, 4ep, speed3, WIND_START=40)"
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env SWEEP_ATTACK_TYPE=tilt WIND_MOMENT_ARM=$ARM SWEEP_ATK_START=80 WIND_START=40 SENSOR_NOISE_SCALE=1.0 \
    timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
      --capture-mode hijack --capture-patterns waypoint,circle,figure8,aggressive \
      --capture-disturbances wind_turbulence:6,wind_turbulence:9 \
      --capture-biases 2.616,3.052 --episodes 4 --speed 3 --outdir $OUT > $OUT/run.log 2>&1
  say "$OUT END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done
kill_isaac
touch SWEETSPOT_DONE
say "완료."
