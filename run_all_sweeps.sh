#!/usr/bin/env bash
# ============================================================
# run_all_sweeps.sh — combined 밴드 하한 탐색 (+ torque/thrust 재확인)
#   nohup ./run_all_sweeps.sh > sweep_all.log 2>&1 &
#
# - 범위를 이 스크립트에 직접 박음(--sweep-values) → online_rl_main.py 안 바꿔도 됨.
#   (online_rl_main.py 는 --sweep-values 인자 + killpg 수정 포함 버전이어야 함)
# - 시작 시 results_{combined,torque,thrust} 비우고 처음부터 실행.
# - 각 python을 setsid 독립 세션에서 실행 → 한 모드 끝나도 스크립트 안 죽음.
#
# ※ combined만 다시 돌리고 싶으면 아래 run_one torque / run_one thrust 두 줄을 주석(#) 처리.
# 결과: results_{combined,torque,thrust}/ 에 sweep_*.csv + aggregate.txt + run.log
# ============================================================
set -u
PY=python3

# ── 모드별 sweep 값 ──
declare -A VALUES=(
  [combined]="0.8,1.0,1.1,1.2,1.25,1.3"     # Nm 토크 (추력=5b); track 추락 멈추는 b_track 하한 탐색
  [torque]="1.2,1.3,1.33,1.35,1.37,1.4,1.45,1.5"   # Nm; 밴드 [1.3,1.4) 정밀 (b_track·b_hover 박기)
  [thrust]="8,12,16,20,25,30"               # N; 붕괴(~25N) 재확인
)

echo "########## SWEEP ALL (combined 하한탐색) 시작: $(date +%Y%m%d_%H%M%S) ##########"

# 처음부터: 기존 결과 폴더 삭제
for m in combined torque thrust; do
    [ -d "results_${m}" ] && { echo " [clean] results_${m} 삭제"; rm -rf "results_${m}"; }
done

# setsid --wait 지원 시 세션 격리하며 대기
RUNNER=""
if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait'; then
    RUNNER="setsid --wait"
fi

run_one() {
    local mode="$1" out="results_$1" vals="${VALUES[$1]}"
    mkdir -p "${out}"
    echo ""
    echo "==================================================="
    echo " [$(date +%H:%M:%S)] MODE=${mode} | values=${vals} → ${out}"
    echo "==================================================="
    ${RUNNER} ${PY} online_rl_main.py --sweep --headless \
        --sweep-mode "${mode}" --sweep-values "${vals}" --outdir "${out}" \
        > "${out}/run.log" 2>&1
    echo " [$(date +%H:%M:%S)] ${mode} sweep 종료 (exit=$?)"
    if [ -s "${out}/sweep_summary.csv" ]; then
        ${PY} sweep_aggregate.py "${out}" > "${out}/aggregate.txt" 2>&1
        echo " [$(date +%H:%M:%S)] ${mode} 집계 완료 → ${out}/aggregate.txt"
    else
        echo " [!] ${out}/sweep_summary.csv 없음 — ${out}/run.log 확인"
    fi
}

run_one combined
run_one torque
run_one thrust

echo ""
echo "########## 전부 완료. 요약: ##########"
for m in combined torque thrust; do
    echo "----- results_${m}/aggregate.txt -----"
    sed -n '1,45p' "results_${m}/aggregate.txt" 2>/dev/null || echo "  (없음)"
done
