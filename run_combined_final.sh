#!/usr/bin/env bash
# ============================================================
# run_combined_final.sh — combined ft1.5 최종 확정 스윕 (공격 세팅 동결 후보)
#   nohup ./run_combined_final.sh > sweep_combined_final.log 2>&1 &
#
# combined, ft_ratio=1.5, ramp=0.0. s∈{1.34,1.36,1.38,1.40,1.42}(밴드[1.36,1.40]+양끝마진).
# 정책 track/hover/dhover{1,2,3}. 20ep. 관측압축: vel=log0.5 / gyro=log1p(코드 반영됨).
# 시작 로그 게이트: ramp=0.0s, ep=20, cells=27, ft_ratio=1.5 검증. 불일치 시 즉시 종료.
# 완료 후 sweep_aggregate 자동(vel d′(log0.5)/gyr d′(log1p) 리포트 포함).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.34,1.36,1.38,1.40,1.42"
RAMP="0.0"
EP="20"
FT="1.5"
DELAYS="1,2,3"
OUT="results_combined_final"

mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] START combined ft=${FT} ramp=${RAMP}s ep=${EP} grid=${GRID} → ${OUT}"
echo "=========================================================="
setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
    --ramp "${RAMP}" --episodes "${EP}" \
    --sweep-mode combined --ft-ratio "${FT}" --sweep-values "${GRID}" \
    --hover-delays "${DELAYS}" \
    --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!

# ── 게이트: 최대 120초 [SWEEP] 배너 대기 → ramp/ep/cells/ft_ratio 검증 ──
banner=""; sweepline=""; ftline=""
for i in $(seq 1 120); do
  banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${OUT}/run.log" 2>/dev/null)
  sweepline=$(grep -m1 -E "\[SWEEP\] [0-9]+ cells × [0-9]+ ep" "${OUT}/run.log" 2>/dev/null)
  [ -n "${banner}" ] && [ -n "${sweepline}" ] && break
  kill -0 ${pid} 2>/dev/null || break
  sleep 1
done
cells=$(echo "${sweepline}" | grep -oE "\[SWEEP\] [0-9]+ cells" | grep -oE "[0-9]+")
eps=$(echo "${sweepline}"  | grep -oE "× [0-9]+ ep" | grep -oE "[0-9]+")
ftline=$(grep -m1 -oE "ft_ratio=[0-9.]+" "${OUT}/run.log" 2>/dev/null)
echo " [gate] banner=${banner:-<none>} cells=${cells:-?} ep=${eps:-?} ${ftline:-ft_ratio=?}  (expect ramp=0.0s cells=27 ep=20 ft_ratio=1.5)"
if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "20" ] || [ "${cells}" != "27" ] || [ "${ftline}" != "ft_ratio=1.5" ]; then
  echo " [ABORT] *** GATE MISMATCH *** killing run & stopping driver."
  kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
  exit 2
fi
echo " [gate] OK (ramp=0.0s, ep=20, cells=27, ft_ratio=1.5) — running to completion..."

wait ${pid}; rc=$?
echo " [$(date +%H:%M:%S)] DONE combined-final exit=${rc}  →  ${OUT}"
rows=$(( $(wc -l < "${OUT}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
echo " [rows] ${OUT}/sweep_summary.csv = ${rows} data rows (expect 540)"
if [ -s "${OUT}/sweep_summary.csv" ]; then
  ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
  echo " [agg] ${OUT}/aggregate.txt written"
else
  echo " [!] ${OUT}/sweep_summary.csv 비어있음 — run.log 확인"
fi
echo "########## COMBINED FINAL SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
