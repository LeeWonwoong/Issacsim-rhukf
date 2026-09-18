#!/usr/bin/env bash
# capture_fig4_nis (09-16, 논문 그림 4용): 탐지 비활성(track 고정) 대본 시간창에서 압축 NIS(gyro·vel) 관측을 캡처한다.
#   창: 기저 0–60 · 바람만 60–150 · 겹침 150–180 · 공격만 180–260 · 기저 260–300 (10 Hz, 300 스텝 = 30 s)
#   패턴 circle·aggressive × δ {0, 0.1, 0.4, 0.8} (bias = δ×4.36 N·m) × 강풍 티어 ws10, 에피 2.
#   현재 동결 환경(frozen_v3.env) 그대로 — config E 시절 capture_timewin_E.sh 와 다르다.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export EP_MAX_STEPS=300 WIND_START=60 WIND_END=180 SWEEP_ATK_START=150 SWEEP_ATK_END=260 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES=track
kill_isaac(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
OUT=results/claudecodefortest/fig4_nis; rm -rf $OUT; mkdir -p $OUT; kill_isaac
echo "[$(date '+%m-%d %H:%M')] fig4_nis START (기저→바람60→겹침150→공격만180→기저260)" | tee -a results/claudecodefortest/isaac_commit.log
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns circle,aggressive \
  --capture-disturbances wind_turbulence:10 \
  --capture-biases 0.0,0.436,1.744,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] fig4_nis END rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0)" | tee -a results/claudecodefortest/isaac_commit.log
kill_isaac; touch $OUT/DONE
