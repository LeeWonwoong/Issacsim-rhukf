#!/usr/bin/env bash
# ============================================================
# run_adam_seeds.sh — Adam baseline seed 반복 (LOCKED config, seed 0~4, 각 200ep)
#   nohup bash run_adam_seeds.sh > train_adam_seeds.log 2>&1 &
# 목적: 단일 seed(delay 14)의 신뢰 부족 → Adam 진짜 분포(평균±std) 확보. RHUKF 공식 비교 기준점.
# 순차 실행(단일 Isaac 인스턴스). 각 seed → results_adam_seed{N}/metrics_adam.csv
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
SEEDS="0 1 2 3 4"
for s in ${SEEDS}; do
  OUT="results_adam_seed${s}"
  [ -d "${OUT}" ] && mv "${OUT}" "${OUT}_old_$(date +%H%M%S)"
  mkdir -p "${OUT}"
  echo "=========================================================="
  echo " [$(date +%H:%M:%S)] SEED ${s} START → ${OUT} (LOCKED config, Adam, 200ep)"
  echo "=========================================================="
  python3 -u online_rl_main.py --headless --agent adam --speed 10 \
      --seed "${s}" --outdir "${OUT}" > "${OUT}/run.log" 2>&1
  echo " [$(date +%H:%M:%S)] SEED ${s} DONE"
  grep -A3 "최종 구간" "${OUT}/run.log" 2>/dev/null | head -4
done
echo "=========================================================="
echo " [$(date +%H:%M:%S)] ALL SEEDS COMPLETE"
echo "=========================================================="
