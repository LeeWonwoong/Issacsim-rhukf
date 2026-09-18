#!/usr/bin/env bash
# v3_band_sweep (2026-09-06) — [B1] FROZEN-v3 에서 결과성 밴드 재측정.
#   목적: track 추락 ∧ (d)hover 생존인 δ 구간 확정 (조건①②) + whover3 로 복귀 생존 미리보기.
#   try3: δ {0.6,0.7,0.8,0.9} × 정책 3종 × 4패턴 × 바람 2 × 3ep = 288 (천장 δ1.0+ 는 try2 로 확정)
#   × 바람 {none, turb ws8} × 3ep = 144 셀에피.  공격 = 지속(창 180~380 스텝 = 20s).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3,whover3"
LOG=v3_band_sweep.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
kill_all
OUT=results_v3_band; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] v3_band START (try3 정밀판: δ{0.6,0.7,0.8,0.9} × 4패턴, dhover 온셋앵커 수정판)" | tee $LOG
env | grep -E "^(ATK_|SPEED_|COM_|EP_MAX|WIND_|MPC_|SENSOR_|SWEEP_|CAPTURE_)" | sort | tee -a $LOG
timeout 21600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive \
  --capture-disturbances none:0,wind_turbulence:8 \
  --capture-biases 2.616,3.052,3.488,3.924 \
  --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] v3_band END rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)" | tee -a $LOG
kill_all; touch V3BAND_DONE
