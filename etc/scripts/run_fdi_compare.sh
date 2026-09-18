#!/usr/bin/env bash
# ============================================================
# run_fdi_compare.sh — 상수 bias vs FDI 펄스 공격 비교 (2026-08-14)
#   nohup ./run_fdi_compare.sh > fdi_compare.log 2>&1 &
#
# run_E_full.sh (상수 밴드+Q·R) 끝난 뒤 자동 이어짐.
#   같은 밴드 δ0.4~0.8 · 같은 5궤적에서 **펄스(SWEEP_BURST) 주입**만 다르게.
#   상수 대조군 = results_E_tilt / results_E_roll (E_full 산출물).
#   펄스 = ON 3스텝 / OFF 7스텝(0.3s on / 0.7s off @10Hz, duty 30%) — peak ‖a‖ 동일.
#   비교: 결과성(생존·표류) + NIS 시그니처(스파이크 vs 지속).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=fdi_compare_report.txt
# 무작위 지시함수 γ_k (표준 FDI): ON~U(2,6) / OFF~U(6,14) 스텝 @10Hz
#   = 짧은 무작위 펄스(예측불가) → CUSUM 누적 불리·RL 윈도우 유리. peak ‖a‖는 상수와 동일.
BURST="rand:2,6,6,14"
BIAS_E="1.74,2.18,2.62,3.05,3.49"       # δ 0.4,0.5,0.6,0.7,0.8 (상수 E와 동일)
PATS5="hover,waypoint,figure8,circle,aggressive"

echo "########## FDI COMPARE START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"
# 선행 스윕이 실제로 도는지만 확인(run_sim.py=Isaac). E_full 은 이미 완료됨.
#   ※ 이전 wait 루프의 pgrep "online_rl_main.py --sweep" 는 감시명령 텍스트에 자기매칭돼 데드락 → 제거.
while pgrep -f "run_sim.py --headless" >/dev/null 2>&1; do sleep 30; done
echo "  [ok] Isaac 유휴 확인 — FDI 무작위 지시함수 비교 시작 (γ_k=$BURST)" | tee -a "$REPORT"

run_pulse () {  # name atype out
  local name="$1" atype="$2" out="$3"
  echo | tee -a "$REPORT"
  echo "==== [$name] $(date +%H:%M:%S) type=$atype burst=$BURST ep=6 ====" | tee -a "$REPORT"
  mkdir -p "$out"
  SWEEP_ATTACK_TYPE="$atype" SWEEP_BURST="$BURST" setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --torque-yaw-ratio 0.0 \
      --capture-mode deadline \
      --deadline-patterns "$PATS5" \
      --deadline-biases "$BIAS_E" \
      --deadline-delays 0,3,5 \
      --episodes 6 --ramp 0.0 --speed 10 --log-zu --outdir "$out" \
      > "$out/run.log" 2>&1 &
  local pid=$!; wait $pid
  local rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo "     [done] $out rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
}

# 펄스 tilt / roll
run_pulse "펄스 tilt" "loe_combined" results_F_tilt_pulse
run_pulse "펄스 roll" "loe_roll"     results_F_roll_pulse

# 비교표 (상수 vs 펄스)
echo | tee -a "$REPORT"
echo "==== [비교] 상수 vs 펄스 $(date +%H:%M:%S) ====" | tee -a "$REPORT"
echo "--- TILT ---" | tee -a "$REPORT"
~/isaacsim/python.sh compare_const_pulse.py results_E_tilt results_F_tilt_pulse 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo "--- ROLL ---" | tee -a "$REPORT"
~/isaacsim/python.sh compare_const_pulse.py results_E_roll results_F_roll_pulse 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"

echo | tee -a "$REPORT"
echo "########## FDI COMPARE DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
