#!/usr/bin/env bash
# ============================================================
# run_final.sh — 공격 세팅 동결용 최종 환경 스윕(논문 수치 겸용)
#   nohup ./run_final.sh > sweep_final.log 2>&1 &
#
# 공격채널=torque 확정. ramp=0.0(채택), bias 1.26~1.34, 20ep(논문수치).
# 시작 로그 게이트: ramp=0.0s, ep=20, cells=42 검증. 불일치 시 즉시 종료.
# 완료 후 sweep_aggregate 자동.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.26,1.28,1.30,1.32,1.34"
RAMP="0.0"
EP="20"
OUT="results_torque_final"

mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] START final ramp=${RAMP}s ep=${EP} grid=${GRID} → ${OUT}"
echo "=========================================================="
setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
    --ramp "${RAMP}" --episodes "${EP}" \
    --sweep-mode torque --sweep-values "${GRID}" \
    --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!

# ── 게이트: 최대 120초 [SWEEP] 배너 대기 → ramp/ep/cells 검증 ──
banner=""; sweepline=""
for i in $(seq 1 120); do
  banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${OUT}/run.log" 2>/dev/null)
  sweepline=$(grep -m1 -E "\[SWEEP\] [0-9]+ cells × [0-9]+ ep" "${OUT}/run.log" 2>/dev/null)
  [ -n "${banner}" ] && [ -n "${sweepline}" ] && break
  kill -0 ${pid} 2>/dev/null || break
  sleep 1
done
cells=$(echo "${sweepline}" | grep -oE "\[SWEEP\] [0-9]+ cells" | grep -oE "[0-9]+")
eps=$(echo "${sweepline}"  | grep -oE "× [0-9]+ ep" | grep -oE "[0-9]+")
echo " [gate] banner=${banner:-<none>} cells=${cells:-?} ep=${eps:-?}  (expect ramp=0.0s cells=42 ep=20)"
if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "20" ] || [ "${cells}" != "42" ]; then
  echo " [ABORT] *** GATE MISMATCH *** killing run & stopping driver."
  kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
  exit 2
fi
echo " [gate] OK (ramp=0.0s, ep=20, cells=42) — running to completion..."

wait ${pid}; rc=$?
echo " [$(date +%H:%M:%S)] DONE final exit=${rc}  →  ${OUT}"
rows=$(( $(wc -l < "${OUT}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
echo " [rows] ${OUT}/sweep_summary.csv = ${rows} data rows (expect 840)"
if [ -s "${OUT}/sweep_summary.csv" ]; then
  ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
  echo " [agg] ${OUT}/aggregate.txt written"
else
  echo " [!] ${OUT}/sweep_summary.csv 비어있음 — run.log 확인"
fi
echo "########## FINAL SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
