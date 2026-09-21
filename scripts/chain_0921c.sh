#!/usr/bin/env bash
# 09-21 15:30 사용자 지시: n33(보상 1,2,3,10 확인) → n29(칼만-TD 튜닝) → n17([32,32])
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
for g in n33_reward1231 n29_ktd_tune n17_net32; do
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_all 완료" >> $L
