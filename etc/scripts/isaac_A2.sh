#!/usr/bin/env bash
# A-2 (2026-09-11) — A-1 뒤 보강. 새 규칙(각도 종료 없음).
#   ① 확률 구간 하단 재확인: δ{0.72,0.74} × plateau 30 × {track,dhover3} × 2ep = 80
#   ② A-1 에서 hover 가 δ0.84 ws6 에서 살면(사망 ≤1/20): 상한 탐색 δ{0.86,0.88} (80) + B@δ0.84 plateau{15,20,25} track/dhover3 (60)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0 FLIP_TERMINAL=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_cert.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
run(){ kill_all; OUT=results/claudecodefortest/cert_$1; rm -rf $OUT; mkdir -p $OUT; log "$1 START biases=$2 pol=$3 ep=$4 atk_end=$5"
  WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=$5 CAPTURE_POLICIES="$3" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
    --capture-biases $2 --episodes $4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "$1 END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"; }
while [ ! -f results/claudecodefortest/A1_DONE ]; do sleep 30; done
run A2lo 3.139,3.226 track,dhover3 2 210
HDEAD=$(python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('results/claudecodefortest/cert_A1/sweep_summary.csv')))
d=[r for r in rows if r['policy']=='dhover3' and abs(float(r['bias'])-3.662)<1e-3 and int(round(float(r['wind_speed'])))==6]
print(sum(1-int(float(r['survived'])) for r in d), len(d))
PY
)
log "A1 hover δ0.84 ws6 사망/n = $HDEAD"
set -- $HDEAD
if [ "${2:-0}" -gt 0 ] && [ "$1" -le 1 ]; then
  run A2hi 3.750,3.837 track,dhover3 2 210
  run B84_15 3.662 track,dhover3 2 195; run B84_20 3.662 track,dhover3 2 200; run B84_25 3.662 track,dhover3 2 205
else
  log "A2hi 생략 (hover 가 δ0.84 에서 이미 죽음 → 상한 ≤0.82)"
fi
kill_all; touch results/claudecodefortest/A2_DONE; log "A-2 완료"
