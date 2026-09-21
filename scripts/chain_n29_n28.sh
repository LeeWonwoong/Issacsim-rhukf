#!/usr/bin/env bash
# n27 완료 → n30(alive 0 정수보상) → n29(칼만-TD q·p_init) → n28(축 B 최종; 보상·설정은 n30·n27·n29 판정 후 yaml 갱신) → Isaac 수동
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until grep -q "n27_B_tune 완료" $L; do sleep 120; done
for g in n30_alive0 n32_deadline n29_ktd_tune n28_axisB_final; do
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_all 완료 (Isaac v3 수동 시작 대기)" >> $L
