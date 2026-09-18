#!/usr/bin/env bash
# chain0908b — bandchk(3점) + ②ramp시나리오 + ③약공격 + ④지연곡선 + ⑤편류데모. headless speed2.5
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_chain0908b.log; PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }

# ① 밴드 3확인점 × 전패턴
for cfg in "3.444:8:d79ws8" "3.488:7:d80ws7" "3.488:9:d80ws9"; do
  B=${cfg%%:*}; rest=${cfg#*:}; WS=${rest%%:*}; TAG=${rest##*:}
  kill_all; OUT=results_bandchk_$TAG; rm -rf $OUT; mkdir -p $OUT
  log "① bandchk $TAG START"
  WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
  timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns $PATS --capture-disturbances wind_turbulence:$WS \
    --capture-biases $B --episodes 4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  log "① bandchk $TAG END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
done

# ② ramp 시나리오 타임라인
kill_all; OUT=results_rampscn; rm -rf $OUT; mkdir -p $OUT; log "② ramp scenario START"
WIND_START=40 WIND_END=260 SWEEP_ATK_START=90 SWEEP_ATK_END=290 SWEEP_ATK_WINDOWS="90-140,160-210,235-290" CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:8 --capture-biases 3.488 --ramp 2.5 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "② END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ③ 약공격 aliasing (δ0.2~0.4)
kill_all; OUT=results_weakalias; rm -rf $OUT; mkdir -p $OUT; log "③ weak-alias START"
WIND_START=40 WIND_END=260 SWEEP_ATK_START=90 SWEEP_ATK_END=290 SWEEP_ATK_WINDOWS="90-140,160-210,235-290" CAPTURE_POLICIES="track" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:8 --capture-biases 0.872,1.308,1.744 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "③ END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ④ ramp 지연곡선 (종착 δ0.8)
kill_all; OUT=results_delaycurve; rm -rf $OUT; mkdir -p $OUT; log "④ delay-curve START"
SWEEP_ATK_START=180 SWEEP_ATK_END=235 CAPTURE_POLICIES="track,whover5,whover10,whover15,whover20,whover25,whover30" \
timeout 18000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances none:0,wind_turbulence:6 --capture-biases 3.488 --ramp 2.5 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "④ END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"

# ⑤ 편류 3D 데모
kill_all; OUT=results_driftdemo; rm -rf $OUT; mkdir -p $OUT; log "⑤ drift demo START"
WORLD_ATK_DIR=45 SWEEP_ATK_START=120 SWEEP_ATK_END=240 CAPTURE_POLICIES="track,whover3" \
timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns line,waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 --capture-biases 2.834 --episodes 1 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
log "⑤ END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)"
kill_all; touch CHAIN0908B_DONE; log "체인 완료"
