#!/usr/bin/env bash
# night0908 v2 — headless 고속(s30) 4단. ④ 지연곡선은 ①정밀밴드 실측 종착δ 사용.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_night0908.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }

kill_all; OUT=results_fineband; rm -rf $OUT; mkdir -p $OUT
log "① fineband START (headless s30, δ0.75-0.82 × ws6-10)"
WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
timeout 43200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,aggressive,scurve \
  --capture-disturbances wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_turbulence:9,wind_turbulence:10 \
  --capture-biases 3.27,3.314,3.357,3.401,3.444,3.488,3.532,3.575 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "① END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

kill_all; OUT=results_rampscn; rm -rf $OUT; mkdir -p $OUT
log "② ramp scenario START"
WIND_START=40 WIND_END=260 SWEEP_ATK_START=90 SWEEP_ATK_END=290 SWEEP_ATK_WINDOWS="90-140,160-210,235-290" CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:8 --capture-biases 3.488 --ramp 2.5 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "② END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

kill_all; OUT=results_weakalias; rm -rf $OUT; mkdir -p $OUT
log "③ weak-alias START (δ0.2-0.4)"
WIND_START=40 WIND_END=260 SWEEP_ATK_START=90 SWEEP_ATK_END=290 SWEEP_ATK_WINDOWS="90-140,160-210,235-290" CAPTURE_POLICIES="track" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:8 --capture-biases 0.872,1.308,1.744 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "③ END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ④ 지연곡선 — ① 정밀밴드에서 종착δ 산출. 밴드하단·+0.05(더 센 절벽) 2개 종착으로.
DEND=$(python3 etc/scripts/band_center.py 2>>$LOG); DEND=${DEND:-0.80}
DEND2=$(python3 -c "print(round(min(0.85,$DEND+0.05),2))")
BQ1=$(python3 -c "print(round($DEND*4.36,3))"); BQ2=$(python3 -c "print(round($DEND2*4.36,3))")
log "④ delay-curve START (종착 δ=$DEND,$DEND2 → bias $BQ1,$BQ2)"
kill_all; OUT=results_delaycurve; rm -rf $OUT; mkdir -p $OUT
SWEEP_ATK_START=180 SWEEP_ATK_END=235 CAPTURE_POLICIES="track,whover5,whover10,whover15,whover20,whover25,whover30" \
timeout 18000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases $BQ1,$BQ2 --ramp 2.5 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "④ END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
kill_all; touch NIGHT0908_DONE; log "체인 완료 — 종합/플롯/아티팩트는 세션에서 처리"
