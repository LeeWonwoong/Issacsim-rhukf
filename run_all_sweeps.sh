#!/usr/bin/env bash
# ============================================================
# run_all_sweeps.sh — combined / torque / thrust 순차 sweep + 집계 (퇴근용)
#   nohup ./run_all_sweeps.sh > sweep_all.log 2>&1 &
#
# - 각 python을 setsid로 '독립 세션'에서 실행 → 한 모드가 자기 그룹을
#   정리해도 이 스크립트는 안 죽음(안전).
# - 이미 끝난 모드(results_<mode>/sweep_summary.csv 존재)는 건너뜀 → 재실행 안전.
# 결과: results_{combined,torque,thrust}/ 에 sweep_*.csv + aggregate.txt + run.log
# ============================================================
set -u
PY=python3
echo "########## SWEEP ALL 시작: $(date +%Y%m%d_%H%M%S) ##########"

# setsid --wait 지원 여부 (있으면 세션 격리하며 대기)
RUNNER=""
if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait'; then
    RUNNER="setsid --wait"
fi

run_one() {
    local mode="$1" out="$2"
    mkdir -p "${out}"
    if [ -s "${out}/sweep_summary.csv" ]; then
        echo " [$(date +%H:%M:%S)] ${mode} 이미 완료됨(${out}/sweep_summary.csv) → 건너뜀"
        return 0
    fi
    echo ""
    echo "==================================================="
    echo " [$(date +%H:%M:%S)] MODE=${mode} → ${out}"
    echo "==================================================="
    ${RUNNER} ${PY} online_rl_main.py --sweep --headless \
        --sweep-mode "${mode}" --outdir "${out}" \
        > "${out}/run.log" 2>&1
    echo " [$(date +%H:%M:%S)] ${mode} sweep 종료 (exit=$?)"
    if [ -s "${out}/sweep_summary.csv" ]; then
        ${PY} sweep_aggregate.py "${out}" > "${out}/aggregate.txt" 2>&1
        echo " [$(date +%H:%M:%S)] ${mode} 집계 완료 → ${out}/aggregate.txt"
    else
        echo " [!] ${out}/sweep_summary.csv 없음 — ${out}/run.log 확인"
    fi
}

run_one combined results_combined
run_one torque   results_torque
run_one thrust   results_thrust

echo ""
echo "########## 전부 완료. 요약: ##########"
for m in combined torque thrust; do
    echo "----- results_${m}/aggregate.txt -----"
    sed -n '1,45p' "results_${m}/aggregate.txt" 2>/dev/null || echo "  (없음)"
done
