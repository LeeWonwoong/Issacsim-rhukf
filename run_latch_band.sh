#!/usr/bin/env bash
# ============================================================
# run_latch_band.sh — hover 실제 구제능력으로 밴드/데드라인 재측정 (min_alt 기록)
#   nohup ./run_latch_band.sh > sweep_latch_band.log 2>&1 &
#
# 목적: 스크립트 dhover(단일 latch = RL 고친 latch와 동일)가 combined 추력손실 밴드에서
#       실제로 얼마나 침하(min_alt)하나. 스크립트도 침하→추락 = hover 물리한계=밴드 재정의 필요.
#       스크립트는 살고 RL만 침하 = RL 정책 문제.
# combined, ft_ratio=1.5, ramp=0.0. s∈{1.34,1.37,1.40,1.42,1.44}(밴드+s≥1.41 일부).
# 정책 track/hover/dhover{1,3,5,7,10,14}(dhover10≈RL 실측 delay 9.5의 스크립트 아날로그).
# 10ep. min_alt = summary CSV 신규 컬럼(공격중 최저고도).
# 게이트: ramp=0.0s, ep=10, cells=42, ft_ratio=1.5 검증. 불일치 시 즉시 종료.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.34,1.37,1.40,1.42,1.44"
RAMP="0.0"
EP="10"
FT="1.5"
DELAYS="1,3,5,7,10,14"
OUT="results_latch_band"

mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] START latch-band combined ft=${FT} ramp=${RAMP}s ep=${EP} grid=${GRID} delays=${DELAYS} → ${OUT}"
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
echo " [gate] banner=${banner:-<none>} cells=${cells:-?} ep=${eps:-?} ${ftline:-ft_ratio=?}  (expect ramp=0.0s cells=42 ep=10 ft_ratio=1.5)"
if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "10" ] || [ "${cells}" != "42" ] || [ "${ftline}" != "ft_ratio=1.5" ]; then
  echo " [ABORT] *** GATE MISMATCH *** killing run & stopping driver."
  kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
  exit 2
fi
echo " [gate] OK (ramp=0.0s, ep=10, cells=42, ft_ratio=1.5) — running to completion..."

wait ${pid}; rc=$?
echo " [$(date +%H:%M:%S)] DONE latch-band exit=${rc}  →  ${OUT}"
rows=$(( $(wc -l < "${OUT}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
echo " [rows] ${OUT}/sweep_summary.csv = ${rows} data rows (expect 420)"
echo "########## LATCH-BAND SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
