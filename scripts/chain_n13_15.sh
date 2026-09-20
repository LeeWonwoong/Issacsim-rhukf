#!/usr/bin/env bash
# n12(γ×n-step) 종료 → n13(pΔ 표준 UT) → n14(c_d) → n15(q 사다리) 순차. 각 격자는 GRID_DONE 으로 종료 판정.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n12_gamma_nstep/GRID_DONE ]; do sleep 120; done
for g in n13_pdelta_ut n14_cd n15_ladder_q; do
  echo "[$(date '+%m-%d %H:%M')] chain_n13_15: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_n13_15: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_n13_15 완료" >> $L
