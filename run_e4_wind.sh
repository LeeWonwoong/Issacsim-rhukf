#!/usr/bin/env bash
# ============================================================
# run_e4_wind.sh — E4 바람 외란 스팟체크 (교란원 구성 / aliasing 근거)
#   nohup ./run_e4_wind.sh > sweep_e4.log 2>&1 &
#
# bias=0 정상비행 × wind {0, 5, 10 m/s, turbulence} × track. 10ep.
# 목적: (a) 정상 NIS 분포(vel log0.5/gyro log1p) 상승폭, (b) CUSUM FAR 상승.
# E3(sweep_e3.log)가 끝난 뒤 자동 시작(Isaac 경합 회피). 셀=5(baseline+1.34 track/hover/dhover3).
# 게이트: ramp=0.0s cells=5 ep=10 ft_ratio=1.5 pat=aggressive + (wind런) sim로그 wind_turbulence.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.34"; RAMP="0.0"; EP="10"; FT="1.5"; DELAYS="3"; PAT="aggressive"
EXP_CELLS="5"

# ── E3 완료 대기 (최대 6시간) ──
echo " [$(date +%H:%M:%S)] E4 대기: E3 완료 마커(sweep_e3.log)를 기다림..."
for i in $(seq 1 4320); do
  grep -q "E3 PATTERN SWEEP DONE" sweep_e3.log 2>/dev/null && { echo " [$(date +%H:%M:%S)] E3 완료 감지 — E4 시작"; break; }
  pgrep -f "run_e3_patterns.sh" >/dev/null 2>&1 || pgrep -f "online_rl_main" >/dev/null 2>&1 || { sleep 5; }
  sleep 5
done

run_one() {  # WTYPE WSPEED OUT
  local WTYPE="$1" WSPEED="$2" OUT="$3"
  local WARGS=""
  [ "${WTYPE}" != "none" ] && WARGS="--sweep-wind-type ${WTYPE} --sweep-wind-speed ${WSPEED}"
  mkdir -p "${OUT}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] START E4 wind=${WTYPE}/${WSPEED}m/s ep=${EP} → ${OUT}"
  echo "=========================================================="
  setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
      --ramp "${RAMP}" --episodes "${EP}" \
      --sweep-mode combined --ft-ratio "${FT}" --sweep-values "${GRID}" \
      --hover-delays "${DELAYS}" --sweep-pattern "${PAT}" ${WARGS} \
      --outdir "${OUT}" \
      > "${OUT}/run.log" 2>&1 &
  local pid=$!
  local banner="" sweepline="" ftline="" patline="" windok=0
  for i in $(seq 1 150); do
    banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${OUT}/run.log" 2>/dev/null)
    sweepline=$(grep -m1 -E "\[SWEEP\] [0-9]+ cells × [0-9]+ ep" "${OUT}/run.log" 2>/dev/null)
    patline=$(grep -m1 -oE "pat=[a-z0-9]+" "${OUT}/run.log" 2>/dev/null)
    if [ "${WTYPE}" = "none" ]; then windok=1; else
      grep -q "${WTYPE}" "${OUT}/sim_process.log" 2>/dev/null && windok=1
    fi
    [ -n "${banner}" ] && [ -n "${sweepline}" ] && [ -n "${patline}" ] && [ "${windok}" = "1" ] && break
    kill -0 ${pid} 2>/dev/null || break
    sleep 1
  done
  local cells=$(echo "${sweepline}" | grep -oE "\[SWEEP\] [0-9]+ cells" | grep -oE "[0-9]+")
  local eps=$(echo "${sweepline}"  | grep -oE "× [0-9]+ ep" | grep -oE "[0-9]+")
  ftline=$(grep -m1 -oE "ft_ratio=[0-9.]+" "${OUT}/run.log" 2>/dev/null)
  echo " [gate] banner=${banner:-?} cells=${cells:-?} ep=${eps:-?} ${ftline:-ft=?} ${patline:-pat=?} wind=${WTYPE}(ok=${windok})  (expect ramp=0.0s cells=${EXP_CELLS} ep=${EP} ft_ratio=${FT} pat=${PAT})"
  if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "${EP}" ] || [ "${cells}" != "${EXP_CELLS}" ] || [ "${ftline}" != "ft_ratio=${FT}" ] || [ "${patline}" != "pat=${PAT}" ] || [ "${windok}" != "1" ]; then
    echo " [ABORT] *** GATE MISMATCH (wind=${WTYPE}) *** killing run."
    kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
    return 2
  fi
  echo " [gate] OK (wind=${WTYPE}/${WSPEED}) — running..."
  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE E4 wind=${WTYPE}/${WSPEED} exit=${rc} → ${OUT}"
  if [ -s "${OUT}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
    echo " [agg] ${OUT}/aggregate.txt written"
  else
    echo " [!] ${OUT}/sweep_summary.csv 비어있음 — run.log 확인"
  fi
}

run_one none            0  results_wind0
run_one wind_turbulence 5  results_wind5
run_one wind_turbulence 10 results_wind10
echo "########## E4 WIND SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
