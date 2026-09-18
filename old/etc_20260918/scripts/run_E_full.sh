#!/usr/bin/env bash
# ============================================================
# run_E_full.sh — E 재실행: 밴드×축형태×전궤적 + Q·R 데이터 (2026-08-14)
#   nohup ./run_E_full.sh > E_full.log 2>&1 &
#
# 순서:
#   0) pitch 대칭 스팟체크 (δ0.70~0.90, hover+waypoint) — flip점이 roll과 같나
#   1) E-roll   : loe_roll        δ0.4~0.8, 5궤적, --log-zu
#   2) E-pitch  : loe_pitch       δ0.4~0.8, 5궤적, --log-zu
#   3) E-tilt   : loe_combined     δ0.4~0.8, 5궤적, --log-zu  (yaw0=순수 roll+pitch 대각)
#   4) Q·R 격자 : qr_tune_offline (E-tilt zu_log 재구동; 필터 수정은 검토로 남김)
# 전부 순수 torque(thrust=0), ATK_EN=1, ramp0. δ=Nm/4.36.
# aggressive 포함(데이터 보고 최종 포함여부 결정).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=E_full_report.txt
echo "########## E_FULL START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"

# 선행 스윕 있으면 대기
while pgrep -f "online_rl_main.py --sweep" >/dev/null 2>&1; do sleep 20; done

run_sweep () {  # name atype yawr biases pats ep delays out extra
  local name="$1" atype="$2" yawr="$3" biases="$4" pats="$5" ep="$6" delays="$7" out="$8" extra="$9"
  echo | tee -a "$REPORT"
  echo "==== [$name] $(date +%H:%M:%S) type=$atype yaw=$yawr ep=$ep pats=$pats ====" | tee -a "$REPORT"
  echo "     δ= $(echo $biases | tr ',' ' ' | awk '{for(i=1;i<=NF;i++)printf "%.2f ",$i/4.36}')" | tee -a "$REPORT"
  mkdir -p "$out"
  SWEEP_ATTACK_TYPE="$atype" setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --torque-yaw-ratio "$yawr" \
      --capture-mode deadline \
      --deadline-patterns "$pats" \
      --deadline-biases "$biases" \
      --deadline-delays "$delays" \
      --episodes "$ep" --ramp 0.0 --speed 10 $extra --outdir "$out" \
      > "$out/run.log" 2>&1 &
  local pid=$!; wait $pid
  local rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo "     [done] $out rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
  ~/isaacsim/python.sh analyze_roll_band.py "$out" 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
}

BIAS_E="1.74,2.18,2.62,3.05,3.49"              # δ 0.4,0.5,0.6,0.7,0.8 (단일축; B/스팟으로 상단 커버됨)
BIAS_TILT="1.74,2.18,2.62,3.05,3.49,3.71,3.92" # δ 0.4~0.9 (결합은 crash-drift/flip 상단까지 봐야)
PATS5="hover,waypoint,figure8,circle,aggressive"

# 0) pitch 대칭 스팟체크 — ✅ 이미 완료(results_pitch_top_check, 2026-08-14 16:09). 재실행 안 함.
# run_sweep "0 pitch대칭" "loe_pitch" 0.0 "3.05,3.27,3.49,3.71,3.92" "hover,waypoint" 4 "0,3,5" results_pitch_top_check ""

# 1) E-roll  (δ0.4~0.8)
run_sweep "1 E-roll" "loe_roll" 0.0 "$BIAS_E" "$PATS5" 6 "0,3,5" results_E_roll "--log-zu"
# 2) E-pitch (δ0.4~0.8)
run_sweep "2 E-pitch" "loe_pitch" 0.0 "$BIAS_E" "$PATS5" 6 "0,3,5" results_E_pitch "--log-zu"
# 3) E-tilt (결합 대각, yaw0) — ★δ0.4~0.9 로 확장: 결합 추락(호버구제)밴드 + flip점 확정
run_sweep "3 E-tilt" "loe_combined" 0.0 "$BIAS_TILT" "$PATS5" 6 "0,3,5" results_E_tilt "--log-zu"

# 4) Q·R 격자 (E-tilt zu_log = roll+pitch 둘 다 실림 → gyro 대칭튜닝에 최적)
echo | tee -a "$REPORT"
echo "==== [4 Q·R 격자] $(date +%H:%M:%S) 입력=results_E_tilt/zu_log.npz ====" | tee -a "$REPORT"
if [ -f results_E_tilt/zu_log.npz ]; then
  echo "     zu_log $(du -h results_E_tilt/zu_log.npz | cut -f1)" | tee -a "$REPORT"
  ~/isaacsim/python.sh qr_tune_offline.py results_E_tilt/zu_log.npz -o results_qr_grid_Efull \
      2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
else
  echo "     ✗ zu_log 없음" | tee -a "$REPORT"
fi

echo | tee -a "$REPORT"
echo "########## E_FULL DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
