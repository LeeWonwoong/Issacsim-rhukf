#!/usr/bin/env bash
# 09-21 21:50 사용자 지시: Isaac 잠시 미루고 리플레이 버퍼 1e4 확인(n37a 축A → n37p 펄스) → 끝나면 Isaac v4 체인 재개
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
for g in n37a_buffer1e4 n37p_buffer1e4; do
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "- $(date '+%H:%M') (09-21) chain_0921g: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 3 --cpu 3 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 60; done
  echo "- $(date '+%H:%M') (09-21) chain_0921g: $g 완료" >> $L
done
echo "- $(date '+%H:%M') chain_0921g: Isaac v4 체인 재개" >> $L
exec bash scripts/chain_isaac_v4.sh
