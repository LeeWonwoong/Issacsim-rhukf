#!/usr/bin/env bash
# 09-21 17:40 사용자 지시: n35(바람 3단) 완료 후 → n16b(관측 /4 vs 1, v4·새 보상) → n29b(칼만-TD q {1e-3,1e-4,1e-5}, R1·He). 그 뒤는 허락 후.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n35_axisB_reward1231/GRID_DONE ]; do sleep 120; done
for g in n16b_obsdiv_v4 n29b_ktd_q_v4; do
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "- $(date '+%H:%M') (09-21) chain_0921e: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "- $(date '+%H:%M') (09-21) chain_0921e: $g 완료" >> $L
done
echo "- $(date '+%H:%M') chain_0921e 완료" >> $L
