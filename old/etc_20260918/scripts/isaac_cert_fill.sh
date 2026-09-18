#!/usr/bin/env bash
# cert_fill — CERT_DONE 뒤 cert_A 에서 (bias, policy) 조합이 15행(5패턴×3ep) 미만이면 그 bias 를 재실행 (timeout 10800 에 잘린 꼬리 보충)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt SPEED_MOD_AMP=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_cert.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ ! -f results/claudecodefortest/CERT_DONE ]; do sleep 60; done
MISS=$(python3 - <<'PY'
import csv,collections
rows=list(csv.DictReader(open('results/claudecodefortest/cert_A/sweep_summary.csv')))
c=collections.Counter((r['bias'],r['policy']) for r in rows)
need=sorted({b for b in ('3.052','3.139','3.226','3.314','3.401','3.488','3.575') for pol in ('track','dhover3') if c.get((f'{float(b):.3f}',pol),0)<15 and c.get((b,pol),0)<15})
print(','.join(need))
PY
)
if [ -z "$MISS" ]; then log "cert_fill: 보충 불필요"; touch results/claudecodefortest/CERTFILL_DONE; exit 0; fi
kill_all; OUT=results/claudecodefortest/cert_A2; rm -rf $OUT; mkdir -p $OUT; log "cert_fill START biases=$MISS"
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases $MISS --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "cert_fill END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"; kill_all; touch results/claudecodefortest/CERTFILL_DONE
