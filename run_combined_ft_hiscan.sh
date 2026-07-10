#!/usr/bin/env bash
# ============================================================
# run_combined_ft_hiscan.sh — combined vel-활성화 경계 스캔 (고 ft_ratio)
#   nohup ./run_combined_ft_hiscan.sh > sweep_ft_hiscan.log 2>&1 &
#
# 목적: T(추력바이어스)를 PX4 고도루프 흡수한계(~2.85N) 위아래로 밀어
#       vel NIS가 "대응 가능 영역"에서 유의하게 켜지는 창이 있는지 확인.
#       (있으면 combined 재검토, 없으면 torque-only 최종 확정.)
# ※ CLAUDE.md의 torque-only 동결값(FROZEN)·swrl_config 는 건드리지 않음(탐색 전용).
#
# 설계: ft_ratio(=T/τ) ∈ {2.0,2.5,3.5}  (T=ft·s 를 흡수한계 2.85N 위아래로)
#       s(=τ) ∈ {1.32,1.36,1.40}
#       정책 track/hover/dhover3, 10ep/cell, ramp0.0, combined.
#       cells = 2(baseline) + 3값×3정책 = 11.  summary rows = 110.
#       th_n = ft_ratio·s 는 summary th_n 컬럼에 기록(흡수한계 대비 확인용).
# 3판 순차(Isaac 동시 1개). 각 판 게이트: mode=combined, ft_ratio, cells=11, ep=10, ramp=0.0.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.32,1.36,1.40"
DELAYS="3"
RAMP="0.0"
EP="10"

run_one() {
  local ftr="$1" out="$2"
  mkdir -p "${out}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] START ft_ratio=${ftr} → ${out}"
  echo "=========================================================="
  setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
      --sweep-mode combined --sweep-values "${GRID}" \
      --ft-ratio "${ftr}" --hover-delays "${DELAYS}" \
      --ramp "${RAMP}" --episodes "${EP}" --outdir "${out}" \
      > "${out}/run.log" 2>&1 &
  local pid=$!

  # ── 게이트: 최대 300초 [SWEEP] 배너 대기 → mode/ft_ratio/cells/ep/ramp 검증 ──
  local banner="" sweepline="" i
  for i in $(seq 1 300); do
    banner=$(grep -m1 -oE "ramp=[0-9.]+s" "${out}/run.log" 2>/dev/null)
    sweepline=$(grep -m1 -E "\[SWEEP\] [0-9]+ cells × [0-9]+ ep" "${out}/run.log" 2>/dev/null)
    [ -n "${banner}" ] && [ -n "${sweepline}" ] && break
    kill -0 ${pid} 2>/dev/null || break
    sleep 1
  done
  local cells eps hasmode hasftr
  cells=$(echo "${sweepline}" | grep -oE "\[SWEEP\] [0-9]+ cells" | grep -oE "[0-9]+")
  eps=$(echo "${sweepline}"   | grep -oE "× [0-9]+ ep" | grep -oE "[0-9]+")
  hasmode=$(echo "${sweepline}" | grep -c "mode=combined")
  hasftr=$(echo "${sweepline}"  | grep -c "ft_ratio=${ftr}")
  echo " [gate] ${out}: banner=${banner:-<none>} cells=${cells:-?} ep=${eps:-?} mode=combined?${hasmode} ft_ratio=${ftr}?${hasftr}"
  echo " [gate] sweepline: ${sweepline}"
  if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "10" ] || [ "${cells}" != "11" ] \
     || [ "${hasmode}" != "1" ] || [ "${hasftr}" != "1" ]; then
    echo " [ABORT] *** GATE MISMATCH *** killing run & stopping driver."
    kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
    exit 2
  fi
  echo " [gate] OK (combined, ft_ratio=${ftr}, cells=11, ep=10, ramp=0.0) — running..."

  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE ft_ratio=${ftr} exit=${rc} → ${out}"
  local rows=$(( $(wc -l < "${out}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo " [rows] ${out}/sweep_summary.csv = ${rows} data rows (expect 110)"
  if [ -s "${out}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${out}" > "${out}/aggregate.txt" 2>&1
    echo " [agg] ${out}/aggregate.txt written"
  else
    echo " [!] ${out}/sweep_summary.csv 비어있음 — run.log 확인"
  fi
}

echo "########## COMBINED HI-ft_ratio SCAN 시작: $(date +%Y%m%d_%H%M%S) ##########"
run_one 2.0 results_combined_ft20
run_one 2.5 results_combined_ft25
run_one 3.5 results_combined_ft35
echo "########## ALL 3 HI-ft_ratio SWEEPS DONE: $(date +%Y%m%d_%H%M%S) ##########"
