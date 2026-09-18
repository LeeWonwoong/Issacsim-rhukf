#!/usr/bin/env bash
# sparse (2026-09-10 사용자 지정) — 급작 단일버스트 δ{0.74,0.77,0.80} × ON{5,15,30스텝} × {track,dhover1,dhover3,dhover5}
#   × 5패턴 × 3ep = 540에피. 무풍 · EP300 · 온셋 180. 판정: (δ,ON) 별 track 사망률 vs dhover{d} 생존률 = sparse 데드라인 표.
#   + 4단계: 약공격 δ{0.2,0.35,0.5} × ON{15,30} × track × 5패턴 × 2ep = 60에피 (현 config 약공격 NIS 시계열 캡처)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_sparse.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
PATS=waypoint,circle,figure8,aggressive,scurve
for ON in 15 30; do
  kill_all; OUT=results_sparse_weak_on$ON; rm -rf $OUT; mkdir -p $OUT; log "WEAK ON=$ON START"
  SWEEP_ATK_START=180 SWEEP_ATK_END=$((180+ON)) CAPTURE_POLICIES="track" \
  timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances none:0 \
    --capture-biases 0.872,1.526,2.180 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "WEAK ON=$ON END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done
for ON in 5 15 30; do
  kill_all; OUT=results_sparse_on$ON; rm -rf $OUT; mkdir -p $OUT; log "ON=$ON START"
  SWEEP_ATK_START=180 SWEEP_ATK_END=$((180+ON)) CAPTURE_POLICIES="track,dhover1,dhover3,dhover5" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances none:0 \
    --capture-biases 3.226,3.357,3.488 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "ON=$ON END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done
kill_all; touch SPARSE_DONE; log "sparse 전체 완료"
