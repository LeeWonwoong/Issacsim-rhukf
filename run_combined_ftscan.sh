#!/usr/bin/env bash
# ============================================================
# run_combined_ftscan.sh — combined 공격 밴드 탐색: ft_ratio 3개 비교(탐색용 10ep)
#   nohup ./run_combined_ftscan.sh > sweep_ftscan.log 2>&1 &
#
# 목적: (1) vel NIS를 gyro에 필적하게 켜는 thrust 성분 탐색
#       (2) torque-only(τ<1.33) 대비 밴드 폭이 넓어지는지 확인.
# ※ CLAUDE.md의 torque-only 동결값(FROZEN)은 건드리지 않음. 이건 combined 후보 탐색.
# ※ swrl_config 학습 샘플러(bias_ft_ratio) 정렬은 탐색 결과 나온 뒤 결정(여기서 안 함).
#
# 설계: ft_ratio(=T/τ) ∈ {0.5,1.0,1.5},  s(=τ) ∈ {1.24,1.28,1.32,1.36,1.40}
#       정책 track/hover/dhover{1,2,3}, 10ep/cell, ramp0.0, combined.
#       th_n = ft_ratio·s 는 summary th_n 컬럼에 기록됨(흡수한계 2.85N 대비 확인용).
# 3판 순차(Isaac 동시 1개). 각 판 게이트: mode=combined, ft_ratio, cells=27, ep=10, ramp=0.0.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="1.24,1.28,1.32,1.36,1.40"
DELAYS="1,2,3"
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

  # ── 게이트: 최대 120초 [SWEEP] 배너 대기 → mode/ft_ratio/cells/ep/ramp 검증 ──
  local banner="" sweepline="" i
  for i in $(seq 1 120); do
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
  if [ "${banner}" != "ramp=0.0s" ] || [ "${eps}" != "10" ] || [ "${cells}" != "27" ] \
     || [ "${hasmode}" != "1" ] || [ "${hasftr}" != "1" ]; then
    echo " [ABORT] *** GATE MISMATCH *** killing run & stopping driver."
    kill -TERM -${pid} 2>/dev/null; sleep 3; kill -KILL -${pid} 2>/dev/null
    exit 2
  fi
  echo " [gate] OK (combined, ft_ratio=${ftr}, cells=27, ep=10, ramp=0.0) — running..."

  wait ${pid}; local rc=$?
  echo " [$(date +%H:%M:%S)] DONE ft_ratio=${ftr} exit=${rc} → ${out}"
  local rows=$(( $(wc -l < "${out}/sweep_summary.csv" 2>/dev/null || echo 1) - 1 ))
  echo " [rows] ${out}/sweep_summary.csv = ${rows} data rows (expect 270)"
  if [ -s "${out}/sweep_summary.csv" ]; then
    ${PY} sweep_aggregate.py "${out}" > "${out}/aggregate.txt" 2>&1
    echo " [agg] ${out}/aggregate.txt written"
  else
    echo " [!] ${out}/sweep_summary.csv 비어있음 — run.log 확인"
  fi
}

echo "########## COMBINED ft_ratio SCAN 시작: $(date +%Y%m%d_%H%M%S) ##########"
run_one 0.5 results_combined_ft05
run_one 1.0 results_combined_ft10
run_one 1.5 results_combined_ft15
echo "########## ALL 3 ft_ratio SWEEPS DONE: $(date +%Y%m%d_%H%M%S) ##########"
