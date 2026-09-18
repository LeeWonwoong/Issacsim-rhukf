#!/usr/bin/env bash
# driftdemo — 방향성 편류 3D 데모: 직선/궤적 비행 중 월드 한방향(NE 45°) 편류.
#   track(무대응→저멀리 흘러감) vs whover3(호버로 잡고 안정화→복귀). ramp 없음(급편류), δ0.65(밴드아래=안죽음).
#   위치 로깅(pos_x,pos_y) 필요 → 신 코드로 실행 (night0908 이후 대기).
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f NIGHT0908_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
export WORLD_ATK_DIR=45   # 월드 NE (오른쪽+전방) 편류
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_driftdemo; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] driftdemo START (WORLD_ATK_DIR=45, δ0.65, 편류)" | tee isaac_driftdemo.log
# 공격창 120-240(12s 편류), track 은 안잡고 흘러감 / whover3 는 온셋+3 잡고 종료+20 복귀
SWEEP_ATK_START=120 SWEEP_ATK_END=240 CAPTURE_POLICIES="track,whover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns line,waypoint,circle,figure8,aggressive,scurve \
  --capture-disturbances none:0 --capture-biases 2.834 --episodes 1 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] driftdemo END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_driftdemo.log
kill_all; touch DRIFTDEMO_DONE
