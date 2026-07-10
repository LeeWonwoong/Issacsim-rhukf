#!/usr/bin/env bash
# ============================================================
# run_band.sh — torque 밴드 정밀화(단일 ramp) 세분 그리드
#   nohup ./run_band.sh > sweep_band.log 2>&1 &
#
# - E2 판독 결과 공격채널=torque 확정 후, 밴드 폭 확정용.
# - grid: 1.30 주변 세분. ramp=0.0(abrupt, ramp무관 실측). 8ep(지형 파악용).
# - 시작 시 [SWEEP] 배너의 실효 ramp 게이트 검증. 완료 후 sweep_aggregate 자동.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.28,1.30,1.32,1.34"
RAMP="0.0"
OUT="results_torque_band"

mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] START band ramp=${RAMP}s grid=${GRID} → ${OUT}"
echo "=========================================================="
setsid ${PY} online_rl_main.py --sweep --headless \
    --sweep-mode torque --sweep-values "${GRID}" \
    --ramp "${RAMP}" --speed 10 --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!

# ── ramp 게이트: 최대 120초 배너 대기 → 실효 ramp 검증 ──
banner=""
for i in $(seq 1 120); do
  banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${OUT}/run.log" 2>/dev/null)
  [ -n "${banner}" ] && break
  kill -0 ${pid} 2>/dev/null || break
  sleep 1
done
echo " [gate] ${OUT} banner=${banner:-<none>}  (expected ramp=${RAMP}s)"
if [ "${banner}" != "ramp=${RAMP}s" ]; then
  echo " [ABORT] *** RAMP MISMATCH *** killing run & stopping driver."
  kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
  exit 2
fi
echo " [gate] RAMP OK — running to completion..."

wait ${pid}; rc=$?
echo " [$(date +%H:%M:%S)] DONE band exit=${rc}  →  ${OUT}"
rows=$(( $(wc -l < "${OUT}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
echo " [rows] ${OUT}/sweep_summary.csv = ${rows} data rows"
if [ -s "${OUT}/sweep_summary.csv" ]; then
  ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
  echo " [agg] ${OUT}/aggregate.txt written"
else
  echo " [!] ${OUT}/sweep_summary.csv 비어있음 — run.log 확인"
fi
echo "########## BAND SWEEP DONE: $(date +%Y%m%d_%H%M%S) ##########"
