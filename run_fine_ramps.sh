#!/usr/bin/env bash
# ============================================================
# run_fine_ramps.sh — torque fine 그리드 × ramp {0.0, 0.1, 0.3}
#   nohup ./run_fine_ramps.sh > sweep_fine.log 2>&1 &
#
# - 3판 순차(Isaac/포트8888은 동시 1개만 가능).
# - 각 판 시작 시 [SWEEP] 배너의 실효 ramp를 게이트로 검증.
#   기대값과 불일치하면 해당 python을 죽이고 드라이버 즉시 종료(exit 2).
# - 판당 50 cells × 8 ep = 400행. 완료 후 sweep_aggregate 자동 실행.
# ============================================================
# ROS setup.bash는 set -u에서 unbound var로 치명적 종료 → set +u로 감싼다.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.20,1.25,1.30,1.35,1.40,1.45"

run_one() {
  local ramp="$1" out="$2"
  mkdir -p "${out}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] START ramp=${ramp}s  →  ${out}"
  echo "=========================================================="
  setsid ${PY} online_rl_main.py --sweep --headless \
      --sweep-mode torque --sweep-values "${GRID}" \
      --ramp "${ramp}" --speed 10 --outdir "${out}" \
      > "${out}/run.log" 2>&1 &
  local pid=$!

  # ── ramp 게이트: 최대 120초 배너 대기 → 실효 ramp 검증 ──
  local banner="" i
  for i in $(seq 1 120); do
    banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${out}/run.log" 2>/dev/null)
    [ -n "${banner}" ] && break
    # python이 죽었으면 대기 중단
    kill -0 ${pid} 2>/dev/null || break
    sleep 1
  done
  echo " [gate] ${out} banner=${banner:-<none>}  (expected ramp=${ramp}s)"
  if [ "${banner}" != "ramp=${ramp}s" ]; then
    echo " [ABORT] *** RAMP MISMATCH *** killing run & stopping driver."
    kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
    exit 2
  fi
  echo " [gate] RAMP OK — running to completion..."

  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE ramp=${ramp}s exit=${rc}  →  ${out}"
  local rows=$(( $(wc -l < "${out}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo " [rows] ${out}/sweep_summary.csv = ${rows} data rows (expect 400)"
  if [ -s "${out}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${out}" > "${out}/aggregate.txt" 2>&1
    echo " [agg] ${out}/aggregate.txt written"
  else
    echo " [!] ${out}/sweep_summary.csv 비어있음 — run.log 확인"
  fi
}

echo "########## FINE RAMP SWEEP 시작: $(date +%Y%m%d_%H%M%S) ##########"
run_one 0.0 results_torque_r00_fine
run_one 0.1 results_torque_r01_fine
run_one 0.3 results_torque_r03_fine
echo "########## ALL 3 FINE SWEEPS DONE: $(date +%Y%m%d_%H%M%S) ##########"
