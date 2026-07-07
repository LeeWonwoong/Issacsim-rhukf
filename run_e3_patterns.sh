#!/usr/bin/env bash
# ============================================================
# run_e3_patterns.sh — E3 기동 패턴 스팟체크 (밴드가 기동에 따라 이동하는지)
#   nohup ./run_e3_patterns.sh > sweep_e3.log 2>&1 &
#
# combined ft1.5, ramp0.0, s∈{1.34,1.37,1.40}. 정책 track/hover/dhover3. 10ep.
# 패턴 aggressive → circle 순차. 관측압축 vel=log0.5/gyro=log1p.
# 게이트: ramp=0.0s, ep=10, cells=11, ft_ratio=1.5, pat=<pattern>. 불일치 시 즉시 종료.
# 셀 = baseline(track,hover) + 3s×(track,hover,dhover3) = 11.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.34,1.37,1.40"
RAMP="0.0"; EP="10"; FT="1.5"; DELAYS="3"
EXP_CELLS="11"

run_one() {
  local PAT="$1" OUT="$2"
  mkdir -p "${OUT}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] START E3 pat=${PAT} ft=${FT} ramp=${RAMP}s ep=${EP} grid=${GRID} → ${OUT}"
  echo "=========================================================="
  setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
      --ramp "${RAMP}" --episodes "${EP}" \
      --sweep-mode combined --ft-ratio "${FT}" --sweep-values "${GRID}" \
      --hover-delays "${DELAYS}" --sweep-pattern "${PAT}" \
      --outdir "${OUT}" \
      > "${OUT}/run.log" 2>&1 &
  local pid=$!
  local banner="" sweepline="" ftline="" patline=""
  for i in $(seq 1 120); do
    banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${OUT}/run.log" 2>/dev/null)
    sweepline=$(grep -m1 -E "\[SWEEP\] [0-9]+ cells × [0-9]+ ep" "${OUT}/run.log" 2>/dev/null)
    patline=$(grep -m1 -oE "pat=[a-z0-9]+" "${OUT}/run.log" 2>/dev/null)
    [ -n "${banner}" ] && [ -n "${sweepline}" ] && [ -n "${patline}" ] && break
    kill -0 ${pid} 2>/dev/null || break
    sleep 1
  done
  local cells=$(echo "${sweepline}" | grep -oE "\[SWEEP\] [0-9]+ cells" | grep -oE "[0-9]+")
  local eps=$(echo "${sweepline}"  | grep -oE "× [0-9]+ ep" | grep -oE "[0-9]+")
  ftline=$(grep -m1 -oE "ft_ratio=[0-9.]+" "${OUT}/run.log" 2>/dev/null)
  echo " [gate] banner=${banner:-<none>} cells=${cells:-?} ep=${eps:-?} ${ftline:-ft_ratio=?} ${patline:-pat=?}  (expect ramp=0.0s cells=${EXP_CELLS} ep=${EP} ft_ratio=${FT} pat=${PAT})"
  if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "${EP}" ] || [ "${cells}" != "${EXP_CELLS}" ] || [ "${ftline}" != "ft_ratio=${FT}" ] || [ "${patline}" != "pat=${PAT}" ]; then
    echo " [ABORT] *** GATE MISMATCH (${PAT}) *** killing run."
    kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
    return 2
  fi
  echo " [gate] OK (${PAT}) — running to completion..."
  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE E3 pat=${PAT} exit=${rc} → ${OUT}"
  local rows=$(( $(wc -l < "${OUT}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo " [rows] ${OUT}/sweep_summary.csv = ${rows} data rows (expect 110)"
  if [ -s "${OUT}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
    echo " [agg] ${OUT}/aggregate.txt written"
  else
    echo " [!] ${OUT}/sweep_summary.csv 비어있음 — run.log 확인"
  fi
}

run_one aggressive results_pattern_agg
run_one circle     results_pattern_circle
echo "########## E3 PATTERN SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
