#!/usr/bin/env bash
# ============================================================
# run_wind_pomdp.sh — 바람 POMDP화 + 슬라이딩 윈도우 검증 (2026-08-16)
#   nohup ./run_wind_pomdp.sh > wind_pomdp.log 2>&1 &
#
# 목적: turbulence 바람(0~8)이 gyro NIS 를 튀겨 "바람 vs 공격" aliasing 생성 = POMDP.
#   ① 상수 공격 vs 바람  ② 펄스(FDI) 공격 vs 바람  둘 다 측정 → const/pulse 결정 근거.
#   ★ 슬라이딩 윈도우(4)가 단일샘플보다 공격을 바람에서 잘 분리하는지(관측구조 정당화).
# capture-hijack: track(무방어)로 각 (바람세기 × bias{0=평시, 0.6=공격}) 순회. --log-zu.
#   Q·R 은 방금 확정값 적용됨(R_gyro0.2/Q5e-3, R_vel0.1).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=wind_pomdp_report.txt
# turbulence 1~8 전세기 + gust:5 한개(흡수검증) + none:0(무풍기준)
DIST="none:0,wind_turbulence:1,wind_turbulence:2,wind_turbulence:3,wind_turbulence:4,wind_turbulence:5,wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_gust:5"
BIAS="0.0,3.05"                 # 0=평시(바람만), 3.05=δ0.7 tilt 공격(바람서 효과 좋은 강도)
PATS="hover,aggressive"         # 깨끗(바람vs공격) + 최악(기동+바람+공격 aliasing)
EP=5
# ★ 시간창(30s=300스텝): 바람 80~220(8~22s), 공격 140~260(14~26s)
#   → clean 0~8s | 바람만 8~14s | 겹침 14~22s | 공격만 22~26s | 복귀 26~30s
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260

echo "########## WIND POMDP START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"
while pgrep -f "run_sim.py --headless" >/dev/null 2>&1; do sleep 30; done

run_wind () {  # name burst out
  local name="$1" burst="$2" out="$3"
  echo | tee -a "$REPORT"
  echo "==== [$name] $(date +%H:%M:%S) burst='${burst:-상수}' turbulence 0~8 ====" | tee -a "$REPORT"
  mkdir -p "$out"
  local ENV="SWEEP_ATTACK_TYPE=loe_combined"
  [ -n "$burst" ] && ENV="$ENV SWEEP_BURST=$burst"
  env $ENV setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --torque-yaw-ratio 0.0 \
      --capture-mode hijack \
      --capture-disturbances "$DIST" \
      --capture-biases "$BIAS" \
      --capture-patterns "$PATS" \
      --episodes "$EP" --ramp 0.0 --speed 10 --log-zu --outdir "$out" \
      > "$out/run.log" 2>&1 &
  local pid=$!; wait $pid
  local rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo "     [done] $out rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
}

# A) 상수 공격 + 바람
run_wind "A 상수+바람" "" results_wind_const
# B) 펄스(FDI) 공격 + 바람
run_wind "B 펄스+바람" "rand:2,6,6,14" results_wind_pulse

# 분석: 바람세기별 공격vs바람 분리도 (단일 vs 윈도우4), const vs pulse
echo | tee -a "$REPORT"
echo "==== [분석] $(date +%H:%M:%S) ====" | tee -a "$REPORT"
~/isaacsim/python.sh wind_pomdp_analysis.py results_wind_const results_wind_pulse 2>/dev/null \
    | grep -v "Warn\|Deprecat" | tee -a "$REPORT"

echo | tee -a "$REPORT"
echo "########## WIND POMDP DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
