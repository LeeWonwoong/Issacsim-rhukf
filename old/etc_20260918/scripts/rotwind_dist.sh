#!/usr/bin/env bash
# rotwind_dist (2026-09-01): 회전 바람(WIND_MOMENT_ARM) 키우면 gyro 스파이크 생기나?
#   arm {0.02(현재),0.1,0.2} × zu-log × speed2(검증된 가속) → 리플레이로 4클래스 분포.
#   ⚠ obs-scale 실험(GPU) 완료 후 실행.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=rotwind_dist.log
export SENSOR_NOISE_SCALE=1.0 EP_MAX_STEPS=400 SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
EP=30
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

say "대기: obs-scale(OBS_SCALE_DONE) 완료 + online_rl 소멸..."
while true; do
  running=$(ps -eo args | grep -E "online_rl_main|surrogate_run|obs_scale" | grep -v grep | wc -l)
  [ -f OBS_SCALE_DONE ] && [ "$running" -eq 0 ] && break
  sleep 30
done
say "obs-scale 완료 확인 → rotwind 캡처 시작."

kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }

for ARM in 0.05 0.10 0.15; do
  OUT=results_rotwind_a${ARM}
  say "$OUT START (WIND_MOMENT_ARM=$ARM speed3 ep=$EP zu-log)"
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env WIND_MOMENT_ARM=$ARM timeout 5400 python3 -u online_rl_main.py --headless --agent adam \
      --max-ep $EP --speed 3 --seed 42 --log-zu --outdir $OUT > $OUT/train.log 2>&1
  say "$OUT END rc=$? zu=$([ -f $OUT/zu_log.npz ]&&echo Y||echo N)"
done
kill_isaac
touch ROTWIND_DIST_DONE
say "완료. 리플레이+플롯: python3 etc/scripts/rotwind_plot.py"
