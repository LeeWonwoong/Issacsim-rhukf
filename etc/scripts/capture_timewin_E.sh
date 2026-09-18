#!/usr/bin/env bash
# ============================================================
# capture_timewin_E.sh — config E 온라인 필터로 시간창 시나리오 캡처.
#   clean(0-100) → 강풍 ON(100) → 공격 ON(180, 바람과 겹침) → 바람 OFF(280, 공격만)
#   → 공격 OFF(380) → 복귀(-450).   4패턴 × ws8 × δ{0.4, 0.7} × ep1, track+hover 셀.
#   신기동(08-26 ①②③ + agg_phase_s 3.35) + config E(UKF env 노브) 상태의 관측을 그대로 기록.
# ============================================================
cd /home/acsl/projects/Issacsim-rhukf
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done
for p in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null || true; done
sleep 5

# 확정 POMDP env + 신기동
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=1.0
# ── config E (온라인 shadow UKF 에 직접 적용) ──
export UKF_Q_GYRO=2e-2 UKF_R_GYRO=0.02 UKF_R_VEL=0.01 UKF_Q_EULER=2e-3
# ── 시간창 ──
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=280 SWEEP_ATK_START=180 SWEEP_ATK_END=380
export SWEEP_ATTACK_TYPE=tilt

OUT=results_timewin_E
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] timewin_E 시작 (clean→바람100→공격180겹침→바람off280→공격off380)"
timeout 14000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns circle,figure8,waypoint,aggressive \
  --capture-disturbances wind_turbulence:8 \
  --capture-biases 1.744,3.052 --episodes 1 --speed 10 --outdir "$OUT" > "$OUT/run.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0)"
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done
for p in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null || true; done
