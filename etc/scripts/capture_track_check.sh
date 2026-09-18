#!/usr/bin/env bash
# capture_track_check — SPEED_MOD_FREQ 0.5 버그수정 후 추종 품질 확인 캡처 (평시·무공격·무풍).
#   4패턴 + scurve, 각 1에피 400스텝, 위치로그(zu). 학습 아님 — 추종 검증 전용.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 "$p" 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 "$p" 2>/dev/null||true; done
sleep 5
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5
export EP_MAX_STEPS=400 CAPTURE_POLICIES=track
OUT=results_track_check
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] track_check 캡처 시작 (freq0.5, 4패턴+scurve, 평시)"
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode normal --capture-patterns circle,figure8,waypoint,aggressive,scurve \
  --capture-disturbances none:0 \
  --episodes 1 --speed 10 --log-zu \
  --outdir "$OUT" > "$OUT/run.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE rc=$?"
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 "$p" 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 "$p" 2>/dev/null||true; done
