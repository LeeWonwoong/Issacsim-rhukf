#!/usr/bin/env bash
# 09-21 16:30 사용자 지시: n34(SWIRL pΔ0.2·α0.03, 새 보상) → n35(축 B 바람 전환, SWIRL pΔ0.2 vs Adam 3e-4). 그 뒤는 사용자 허락 후.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
for g in n34_pd02_reward1231 n35_axisB_reward1231; do
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "- $(date '+%H:%M') (09-21) chain_0921d: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "- $(date '+%H:%M') (09-21) chain_0921d: $g 완료" >> $L
done
echo "- $(date '+%H:%M') (09-21) chain_0921d 완료" >> $L
