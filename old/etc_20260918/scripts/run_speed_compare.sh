#!/usr/bin/env bash
# ============================================================
# run_speed_compare.sh — 속도 1× vs 1.5× : 모델오차 innovation 이 속도에 늘어나나 (2026-08-17)
#   nohup ./run_speed_compare.sh > speed_compare.log 2>&1 &
#
# 가설: 플랜트=2차추력곡선+모터지연14ms(비선형), UKF=선형근사 → 기동 빠르면(토크명령 빨리바뀜)
#   UKF 미모델링 오차가 커져 innovation↑. 속도 올려 확인.
# 전 패턴 + 바람(none/turb8/gust5) + 공격(δ0.7 tilt) 동시, 시간창 전과 동일(바람80~220 공격140~260).
# raw innovation(res_v/res_g) 로깅. calib 오차는 OFF(자연 모델오차만 봄).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=speed_compare_report.txt
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260
DIST="none:0,wind_turbulence:8,wind_gust:5"
BIAS="0.0,3.05"                    # 평시 + δ0.7 tilt
PATS="hover,waypoint,circle,figure8,aggressive"
EP=4

echo "########## SPEED COMPARE START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"
while pgrep -f "run_sim.py --headless" >/dev/null 2>&1; do sleep 30; done

run_spd () {  # name scale out
  local name="$1" scale="$2" out="$3"
  echo | tee -a "$REPORT"
  echo "==== [$name] $(date +%H:%M:%S) SPEED_SCALE=$scale ====" | tee -a "$REPORT"
  mkdir -p "$out"
  SPEED_SCALE="$scale" SWEEP_ATTACK_TYPE=loe_combined setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
      --capture-disturbances "$DIST" --capture-biases "$BIAS" --capture-patterns "$PATS" \
      --episodes "$EP" --ramp 0.0 --speed 10 --outdir "$out" \
      > "$out/run.log" 2>&1 &
  local pid=$!; wait $pid
  local rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo "     [done] $out rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
}

run_spd "A 속도1.0×" 1.0 results_speed_1p0
run_spd "B 속도1.5×" 1.5 results_speed_1p5

echo | tee -a "$REPORT"
echo "==== [비교] 속도 1.0× vs 1.5× $(date +%H:%M:%S) ====" | tee -a "$REPORT"
~/isaacsim/python.sh speed_compare.py results_speed_1p0 results_speed_1p5 2>/dev/null \
    | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo | tee -a "$REPORT"
echo "########## SPEED COMPARE DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
