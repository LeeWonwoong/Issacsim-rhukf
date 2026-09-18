#!/usr/bin/env bash
# ============================================================
# run_overnight_band.sh — 하이재킹 공격 밴드 확정 밤샘 스윕 (2026-08-13)
#   nohup ./run_overnight_band.sh > overnight_band.log 2>&1 &
#
# 순차: (선행 fill 대기) → B roll정밀 → C yaw단독 → D roll+pitch틸트
#   전부 순수 torque(thrust=0), ATK_EN=1, deadline 격자(hover+waypoint), ramp0.
#   각 스윕 후 analyze_roll_band.py 로 밴드표 자동 출력.
#   δ = bias_Nm / 4.36 (권한 정규화).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=overnight_band_report.txt
echo "########## OVERNIGHT BAND START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"

# ── 선행 스윕이 아직 돌면 끝날 때까지 대기 (Isaac 은 한 번에 하나만) ──
while pgrep -f "online_rl_main.py --sweep" >/dev/null 2>&1; do
  echo "  [wait] 선행 스윕 진행 중 … 30s" ; sleep 30
done
echo "  [ok] 선행 스윕 없음 — 밤샘 시작" | tee -a "$REPORT"

run_sweep () {
  local name="$1" atype="$2" yawr="$3" biases="$4" ep="$5" out="$6"
  echo | tee -a "$REPORT"
  echo "==== [$name] $(date +%H:%M:%S)  type=$atype yaw_ratio=$yawr ep=$ep ====" | tee -a "$REPORT"
  echo "     δ(권한)= $(echo $biases | tr ',' ' ' | awk '{for(i=1;i<=NF;i++)printf "%.2f ",$i/4.36; print ""}')" | tee -a "$REPORT"
  mkdir -p "$out"
  SWEEP_ATTACK_TYPE="$atype" setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --torque-yaw-ratio "$yawr" \
      --capture-mode deadline \
      --deadline-patterns hover,waypoint \
      --deadline-biases "$biases" \
      --deadline-delays 0,3,5 \
      --episodes "$ep" --ramp 0.0 --speed 10 --outdir "$out" \
      > "$out/run.log" 2>&1 &
  local pid=$!
  wait $pid
  local rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo "     [done] $out exit rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
  ~/isaacsim/python.sh analyze_roll_band.py "$out" 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
}

# ── B: roll 상단 정밀 (δ0.70~1.00 고해상도, ep8) — flip 시작점 + hover 구제 한계 확정 ──
#   δ0.4~0.7 은 이미 매끈(첫 런). 관심은 상단 절벽: track flip 어디서 / hover 어디까지 구제.
#   δ = 0.70,0.75,0.80,0.85,0.90,0.93,0.96,1.00 (Nm ×4.36)
run_sweep "B roll상단정밀" "loe_roll" 0.0 \
  "3.05,3.27,3.49,3.71,3.92,4.05,4.19,4.36" 8 results_band_B_rolltop

# ── C: yaw 단독 (δ0.1~1.0, yaw_ratio=1 로 gz=value). 위치표류≈0 이면 yaw≠위치하이재킹 실증 ──
run_sweep "C yaw단독" "loe_yaw" 1.0 \
  "0.44,0.87,1.31,1.74,2.18,3.05,3.92,4.36" 8 results_band_C_yaw

# ── D: roll+pitch 틸트(대각). loe_combined+yaw0+thrust0 → tqx=+δ tqy=-δ (틸트총=√2·δ_축) ──
run_sweep "D 틸트" "loe_combined" 0.0 \
  "0.87,1.31,1.74,2.18,2.62,3.05" 8 results_band_D_tilt

echo | tee -a "$REPORT"
echo "########## OVERNIGHT BAND DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
echo "  리포트: $REPORT" | tee -a "$REPORT"
