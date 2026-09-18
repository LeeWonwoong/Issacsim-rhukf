#!/usr/bin/env bash
# remeasure v2 (2026-09-07) — 실기정합(acc3.0·jerk주입) 실공간 5패턴 GUI 체인
#  [A] 밴드+바람: δ{0.7,0.75,0.8,0.85} × ws{6,8,10} × {track,dhover3} × 5패턴 × 3ep
#  [B] 시나리오 타임라인: 안정화→바람→바람+공격×2→(바람빠짐)공격→정상, δ0.6 × 5패턴 × 무풍기저=ws8
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 DISPLAY=:1   # ★MPC_ACC_HOR 미설정 = run_sim 기본 3.0(실기)
unset MPC_ACC_HOR
export EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
LOG=isaac_remeasure.log
PATS="waypoint,circle,figure8,aggressive,scurve"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }

# [A] 밴드+바람 (지속공격, 온셋180)
kill_all; OUT=results_rm_band; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] [A] band+wind START (acc3.0, 5패턴, ws6/8/10, 바람창 60-430)" | tee $LOG
WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
timeout 28800 python3 -u online_rl_main.py --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:6,wind_turbulence:8,wind_turbulence:10 \
  --capture-biases 3.052,3.270,3.488,3.706 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] [A] END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG

# [B] 시나리오 타임라인 (그림용): 바람40-260, 공격 90-120·150-180·240-290(전이), δ0.6
kill_all; OUT=results_rm_scenario; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] [B] scenario timeline START" | tee -a $LOG
WIND_START=40 WIND_END=260 SWEEP_ATK_START=90 SWEEP_ATK_END=290 \
SWEEP_ATK_WINDOWS="90-120,150-180,240-290" CAPTURE_POLICIES="track,dhover3" \
timeout 10800 python3 -u online_rl_main.py --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns $PATS --capture-disturbances wind_turbulence:8 \
  --capture-biases 2.616 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] [B] END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
kill_all; touch REMEASURE_DONE
