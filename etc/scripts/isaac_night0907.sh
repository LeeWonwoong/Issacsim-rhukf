#!/usr/bin/env bash
# isaac_night0907 — ①δ{0.75,0.85} 밴드 정밀 ②[B2] ramp 파일럿(지연곡선) ③Adam lr 2런 (frozen_v3)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
LOG=isaac_night0907.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
while [ ! -f V3BAND_DONE ]; do sleep 60; done
log "체인 시작 (try3 완료 확인)"

# ── [A] 밴드 정밀: δ 0.75(3.270) / 0.85(3.706) × 3정책 × 4패턴 × 2바람 × 3ep = 144
kill_all; OUT=results_v3_band_fine; rm -rf $OUT; mkdir -p $OUT
log "[A] band fine START"
EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt \
CAPTURE_POLICIES="track,dhover3,whover3" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive \
  --capture-disturbances none:0,wind_turbulence:8 \
  --capture-biases 3.270,3.706 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
log "[A] END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ── [B2] ramp 파일럿: ramp 2.5s → δ0.8(3.488), hold 30 (창 180~235), whover 지연곡선
#     판정①: 곡선 단조("늦으면 죽는다")  판정②: track P(추락) @ hold30
kill_all; OUT=results_v3_rampb2; rm -rf $OUT; mkdir -p $OUT
log "[B2] ramp pilot START"
EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=235 SWEEP_ATTACK_TYPE=tilt \
CAPTURE_POLICIES="track,whover5,whover10,whover15,whover20,whover25,whover30" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,aggressive \
  --capture-disturbances none:0,wind_turbulence:8 \
  --capture-biases 3.488 --ramp 2.5 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
log "[B2] END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ── [C/D] Adam baseline lr {1e-3, 3e-4}, seed42, frozen_v3 (버스트만 — ramp 샘플러는 [C]동결 후)
for LR in 1e-3 3e-4; do
  kill_all; OUT=results_v3_adam_$LR; rm -rf $OUT; mkdir -p $OUT
  log "[adam $LR] START"
  env NET_HIDDEN=16 ADAM_LR=$LR timeout 10800 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  n=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)
  log "[adam $LR] END csv=$n ddserr=$(grep -c 'sequence size exceeds' $OUT/train.log 2>/dev/null||echo 0)"
  [ "$n" -ge 201 ] && touch $OUT/RUN_DONE
done
kill_all; touch NIGHT0907_DONE; log "체인 완료"
