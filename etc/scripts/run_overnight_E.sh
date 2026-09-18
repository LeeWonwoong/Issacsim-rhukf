#!/usr/bin/env bash
# ============================================================
# run_overnight_E.sh — 밴드 전-궤적 테스트(E) + Q·R 튜닝 격자 (2026-08-13)
#   nohup ./run_overnight_E.sh > overnight_E.log 2>&1 &
#
# B/C/D(run_overnight_band.sh) 끝난 뒤 자동 이어짐.
#  E : 측정된 하이재킹 밴드(δ0.4~0.7 순수 roll)를 hover/waypoint/figure8/circle 전 궤적에.
#      급기동(aggressive)은 데이터 보고 결정 → 지금은 제외.
#      --log-zu 로 zu_log.npz 동시 생성 → Q·R 오프라인 튜닝 입력.
#  Q·R: qr_tune_offline.py 로 gyro R·Q 격자 산출(기록만; 필터 수정은 아침 검토).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
REPORT=overnight_band_report.txt

# ── B/C/D 밤샘이 끝날 때까지 대기 (Isaac 한 번에 하나만) ──
while pgrep -f "run_overnight_band.sh" >/dev/null 2>&1 || pgrep -f "online_rl_main.py --sweep" >/dev/null 2>&1; do
  sleep 30
done
echo | tee -a "$REPORT"
echo "########## E + Q·R START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"

# ── 밴드 이식: 첫 런 + B(상단정밀) 결과에서 하이재킹 밴드 자동 산출 ──
BAND_BIASES=$(~/isaacsim/python.sh pick_band.py results_roll_band_0813 results_band_B_rolltop 2>>"$REPORT")
echo "  [band] E 이식 bias(Nm) = ${BAND_BIASES}" | tee -a "$REPORT"
if [ -z "${BAND_BIASES}" ]; then
  BAND_BIASES="1.74,2.18,2.62,3.05"
  echo "  [band] ⚠ 산출 실패 — fallback ${BAND_BIASES}" | tee -a "$REPORT"
fi

# ── E: 밴드 × 전 궤적 (hover,waypoint,figure8,circle), 순수 roll, --log-zu ──
OUT_E=results_band_E_patterns
mkdir -p "$OUT_E"
echo "==== [E 전궤적] $(date +%H:%M:%S)  bias(Nm)=${BAND_BIASES} (B밴드 이식) ====" | tee -a "$REPORT"
SWEEP_ATTACK_TYPE=loe_roll setsid ${PY} online_rl_main.py --sweep --headless \
    --sweep-mode torque --torque-yaw-ratio 0.0 \
    --capture-mode deadline \
    --deadline-patterns hover,waypoint,figure8,circle \
    --deadline-biases "${BAND_BIASES}" \
    --deadline-delays 0,3,5 \
    --episodes 8 --ramp 0.0 --speed 10 --log-zu --outdir "$OUT_E" \
    > "$OUT_E/run.log" 2>&1 &
PIDE=$!
wait $PIDE
rowsE=$(( $(wc -l < "$OUT_E/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
echo "     [done] $OUT_E rows=$rowsE $(date +%H:%M:%S)" | tee -a "$REPORT"
~/isaacsim/python.sh analyze_roll_band.py "$OUT_E" 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"

# ── Q·R 오프라인 튜닝 격자 (zu_log 재구동; 필터는 안 건드림) ──
echo | tee -a "$REPORT"
echo "==== [Q·R 격자] $(date +%H:%M:%S)  입력=$OUT_E/zu_log.npz ====" | tee -a "$REPORT"
if [ -f "$OUT_E/zu_log.npz" ]; then
  echo "     zu_log 크기 $(du -h $OUT_E/zu_log.npz | cut -f1)" | tee -a "$REPORT"
  ~/isaacsim/python.sh qr_tune_offline.py "$OUT_E/zu_log.npz" -o results_qr_grid_E \
      2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
else
  echo "     ✗ zu_log.npz 없음 — Q·R 격자 스킵" | tee -a "$REPORT"
fi

echo | tee -a "$REPORT"
echo "########## E + Q·R DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
