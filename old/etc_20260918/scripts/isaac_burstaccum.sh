#!/usr/bin/env bash
# burstaccum (D실험) — 반복 burst 누적: ON15(1.5s,단발LOC미달) OFF20 × 6회. 기본(약)호버.
#   track(무대응) 누적으로 K번째 LOC 하나 ∧ whover(공격중 호버) 리셋으로 생존하나?
#   δ{0.7,0.8,0.9} · 무풍 · 전5패턴 × 3ep
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=420 SWEEP_ATTACK_TYPE=tilt
export SWEEP_ATK_WINDOWS="180-195,215-230,250-265,285-300,320-335,355-370"   # 6버스트 ON15 OFF20
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_burstaccum; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] burstaccum START (6버스트 ON15 OFF20, 약호버, δ0.7/0.8/0.9)" | tee isaac_burstaccum.log
# track 은 SWEEP_ATK_WINDOWS 로 6버스트. whover3 는 각 온셋 근방 호버(창 전체 커버). SWEEP_ATK_START/END 는 전체 span.
SWEEP_ATK_START=180 SWEEP_ATK_END=370 CAPTURE_POLICIES="track,whover3" \
timeout 12000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 \
  --capture-biases 3.052,3.488,3.924 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] burstaccum END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_burstaccum.log
kill_all; touch BURSTACCUM_DONE
