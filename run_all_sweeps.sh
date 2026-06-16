#!/usr/bin/env bash
# ============================================================
# run_all_sweeps.sh — combined / torque / thrust 3개 모드를
#   순차로 sweep하고 각각 집계까지 자동 실행 (퇴근용).
#
# 사용:
#   chmod +x run_all_sweeps.sh
#   ./run_all_sweeps.sh            # 백그라운드로 돌리려면:
#   nohup ./run_all_sweeps.sh > sweep_all.log 2>&1 &
#
# 결과:
#   results_combined/ , results_torque/ , results_thrust/
#   각 폴더에 sweep_summary.csv, sweep_detail.csv, aggregate.txt
#   + 최상위 sweep_all.log 에 전체 진행 로그
# ============================================================
set -u
PY=python3
STAMP=$(date +%Y%m%d_%H%M%S)
echo "########## SWEEP ALL 시작: $STAMP ##########"

for MODE in combined torque thrust; do
    OUT="results_${MODE}"
    echo ""
    echo "==================================================="
    echo " [$(date +%H:%M:%S)] MODE=${MODE} → ${OUT}"
    echo "==================================================="
    mkdir -p "${OUT}"

    # sweep 실행 (모드별 권장 grid 자동 적용; 값 바꾸려면 --sweep-values 0,0.5,... 추가)
    ${PY} online_rl_main.py --sweep --headless \
        --sweep-mode "${MODE}" --outdir "${OUT}" \
        > "${OUT}/run.log" 2>&1
    RC=$?
    echo " [$(date +%H:%M:%S)] ${MODE} sweep 종료 (exit=${RC})"

    # 집계 (실패해도 다음 모드 진행)
    if [ -f "${OUT}/sweep_summary.csv" ]; then
        ${PY} sweep_aggregate.py "${OUT}" > "${OUT}/aggregate.txt" 2>&1
        echo " [$(date +%H:%M:%S)] ${MODE} 집계 완료 → ${OUT}/aggregate.txt"
    else
        echo " [!] ${OUT}/sweep_summary.csv 없음 — sweep 실패? run.log 확인"
    fi
done

echo ""
echo "########## 전부 완료. 아침에 확인: ##########"
for MODE in combined torque thrust; do
    echo "----- results_${MODE}/aggregate.txt -----"
    sed -n '1,40p' "results_${MODE}/aggregate.txt" 2>/dev/null || echo "  (없음)"
done
