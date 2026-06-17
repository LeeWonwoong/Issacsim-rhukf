#!/usr/bin/env bash
# run_compare.sh — RHUKF / Adam 학습 + 결과 plot
# 사용:
#   ./run_compare.sh rhukf     # RHUKF만 학습 + 단일 plot
#   ./run_compare.sh adam      # Adam만 학습 + 단일 plot
#   ./run_compare.sh compare   # 둘 다 순차 학습 + 통합 비교 plot (기본)
# 학습은 max_episodes(=config, 기본 200)에서 자동 종료. 각 결과는 results_<agent>/.
set -u
MODE="${1:-compare}"
SPEED="${SPEED:-1}"   # 예: SPEED=3 ./run_compare.sh compare
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

run_agent () {
  local AGENT="$1"
  local OUT="results_${AGENT}"
  echo "=================================================================="
  echo " [${AGENT}] 학습 시작 → ${OUT}/  (max_episodes에서 자동 종료)"
  echo "=================================================================="
  rm -rf "$OUT"; mkdir -p "$OUT"
  # setsid --wait: sim 서브프로세스 그룹을 자식으로 묶어 깔끔히 종료
  setsid --wait python3 online_rl_main.py \
      --agent "$AGENT" --headless --outdir "$OUT" --speed "$SPEED" \
      > "${OUT}/run.log" 2>&1
  echo " [${AGENT}] 완료. 메트릭: ${OUT}/metrics_${AGENT}.csv"
}

case "$MODE" in
  rhukf|adam)
    run_agent "$MODE"
    python3 plot_results.py "results_${MODE}/metrics_${MODE}.csv" --outdir "results_${MODE}"
    echo "→ plot: results_${MODE}/metrics_${MODE}.png"
    ;;
  compare)
    run_agent rhukf
    run_agent adam
    mkdir -p results_compare
    python3 plot_results.py \
        results_rhukf/metrics_rhukf.csv \
        results_adam/metrics_adam.csv \
        --outdir results_compare
    echo "→ 통합 비교 plot: results_compare/compare_rhukf_adam.png"
    ;;
  *)
    echo "알 수 없는 모드: $MODE (rhukf|adam|compare)"; exit 1;;
esac
echo "[done] mode=$MODE"
