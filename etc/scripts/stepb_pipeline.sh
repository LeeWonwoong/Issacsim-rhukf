#!/usr/bin/env bash
# stepb_pipeline (2026-09-02): rotwind 완료 → gyro aliasing gate → Step B 크래시밴드 → obs-scale 200ep.
#   전부 speed3(검증된 가속). Step B: δ{0.6,0.7,0.8}=Nm{2.616,3.052,3.488} × 4기동패턴 × {no-wind, ws9} × {track,hover}.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=stepb_pipeline.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }

# 1) rotwind 완료 + Isaac/GPU free 대기
say "대기: ROTWIND_DIST_DONE + 프로세스 소멸..."
while true; do
  r=$(ps -eo args|grep -E "online_rl_main|surrogate_run|obs_scale|rotwind_dist"|grep -v grep|wc -l)
  [ -f ROTWIND_DIST_DONE ] && [ "$r" -eq 0 ] && break
  sleep 30
done
say "rotwind 완료 확인."

# 2) gate: gyro aliasing 확인 + arm 선택
ARM=$(python3 etc/scripts/rotwind_gate.py 2> rotwind_gate.err)
say "gate: ARM=$ARM"; sed 's/^/    /' rotwind_gate.err | tee -a $LOG

# 3) Step B 크래시밴드
if [ "$ARM" != "SKIP" ] && [ -n "$ARM" ]; then
  OUT=results_stepb_crashband
  say "Step B START (arm=$ARM, δ0.6/0.7/0.8, 4패턴, none+ws9, track+hover, speed3)"
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env SWEEP_ATTACK_TYPE=tilt WIND_MOMENT_ARM=$ARM SWEEP_ATK_START=80 WIND_START=40 SENSOR_NOISE_SCALE=1.0 \
    timeout 21600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
      --capture-mode hijack --capture-patterns waypoint,circle,figure8,aggressive \
      --capture-disturbances none:0,wind_turbulence:9 \
      --capture-biases 2.616,3.052,3.488 --episodes 6 --speed 3 --outdir $OUT > $OUT/run.log 2>&1
  say "Step B END rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)"
  kill_isaac
else
  say "gate SKIP — gyro aliasing 미확인(d'<0.4). Step B 생략, 아침에 검토."
fi
touch STEPB_DONE

# 4) obs-scale 200ep 재실행 (surrogate)
say "obs-scale 200ep START"
rm -f OBS_SCALE_DONE
python3 etc/scripts/obs_scale_swirl.py > /tmp/obs_scale200.log 2>&1
say "obs-scale 200ep END: $(grep -E 'seed42:' /tmp/obs_scale200.log|tail -4|tr '\n' ' ')"
touch PIPELINE_ALL_DONE
say "전체 파이프라인 완료."
