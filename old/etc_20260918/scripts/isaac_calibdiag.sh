#!/usr/bin/env bash
# calibdiag — C/I 오차가 급기동 gyro 잔차를 공격과 겹치게 하나 (진단, 학습 아님).
#   조건: baseline / ctq{5,10}% / I{5,10}% / ctq10+I10 = 6  × 패턴{waypoint,scurve,aggressive}
#   × δ0.8 공격 track × 3ep. res_g 로깅 → gyro 애매화 판정.
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f LETHAL_SWEEP_DONE ]; do sleep 60; done   # 강공격스윕 완료 후 (GPU 순차)
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_calibdiag.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
run(){ # $1=UKF_CALIB_ERR $2=tag
  kill_all; OUT=results_cd_$2; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] $2 ($1) START" | tee -a $LOG
  UKF_CALIB_ERR="$1" SWEEP_ATK_START=180 SWEEP_ATK_END=245 CAPTURE_POLICIES="track" \
  timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,scurve,aggressive --capture-disturbances none:0 \
    --capture-biases 3.488 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $2 END rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)" | tee -a $LOG
}
# G = C_torque/I. G↑ 를 세 경로로: ctq↑ / i↓ / 둘 다 (G 크게)
run "" base
run "ctq:0.05" G5_via_ctq          # G +5% (C_torque 경로)
run "ctq:0.10" G10_via_ctq         # G +10%
run "i:-0.05" G5_via_i             # G +5.3% (I 경로) — 순수 G 효과 대조
run "i:-0.10" G11_via_i            # G +11%
run "ctq:0.10,i:-0.10" G22         # G +22% (양경로 = 급기동 강애매화)
run "ctq:0.20,i:-0.15" G41         # G +41% (극단 — 상한 탐색)
kill_all; touch CALIBDIAG_DONE
