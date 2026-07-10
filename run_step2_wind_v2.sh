#!/usr/bin/env bash
# ============================================================
# run_step2_wind_v2.sh — STEP 2: 바람 스윕 (turbulence 진폭버그 수정 후)
#   nohup ./run_step2_wind_v2.sh > sweep_wind_v2.log 2>&1 &
#
# bias=0(무공격) × wind {0,3,5,7} × 정상비행. sweep-values 0 → 모든 셀 bias=0(공격 없음).
# 측정: 각 wind에서 정상 NIS 분포(vel log0.5/gyro log1p, mean·99pct) + CUSUM FAR.
# 전제: run_sim.py WindModel turbulence 진폭 OU 이산화 수정(sqrt(1-a^2)) 적용됨.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
RAMP="0.0"; EP="5"; FT="1.5"; PAT="aggressive"

run_one() {  # WTYPE WSPEED OUT
  local WTYPE="$1" WSPEED="$2" OUT="$3"
  local WARGS=""
  [ "${WTYPE}" != "none" ] && WARGS="--sweep-wind-type ${WTYPE} --sweep-wind-speed ${WSPEED}"
  mkdir -p "${OUT}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] START STEP2 wind=${WTYPE}/${WSPEED}m/s ep=${EP} → ${OUT}"
  setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
      --ramp "${RAMP}" --episodes "${EP}" \
      --sweep-mode combined --ft-ratio "${FT}" --sweep-values 0 \
      --hover-delays 3 --sweep-pattern "${PAT}" ${WARGS} \
      --outdir "${OUT}" \
      > "${OUT}/run.log" 2>&1 &
  local pid=$!
  # 게이트: 30초 내 wind 확인
  local windok=0
  for i in $(seq 1 60); do
    if [ "${WTYPE}" = "none" ]; then windok=1; break; fi
    grep -q "${WTYPE}" "${OUT}/sim_process.log" 2>/dev/null && { windok=1; break; }
    kill -0 ${pid} 2>/dev/null || break
    sleep 1
  done
  echo " [gate] wind=${WTYPE}(ok=${windok})"
  if [ "${windok}" != "1" ]; then
    echo " [ABORT] wind 미확인 — kill"; kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null; return 2
  fi
  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE wind=${WTYPE}/${WSPEED} exit=${rc} → ${OUT}"
  if [ -s "${OUT}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
    echo " [agg] ${OUT}/aggregate.txt"
  else
    echo " [!] ${OUT}/sweep_summary.csv 비어있음"
  fi
}

run_one none            0 results_wind_v2_w0
run_one wind_turbulence 3 results_wind_v2_w3
run_one wind_turbulence 5 results_wind_v2_w5
run_one wind_turbulence 7 results_wind_v2_w7
echo "########## STEP2 WIND_V2 DONE: $(date +%Y%m%d_%H%M%S) ##########"
