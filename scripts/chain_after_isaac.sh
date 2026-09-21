#!/usr/bin/env bash
# Isaac v3 완료 후 남은 surrogate 부록 격자: n16(obs /4) → n18(eval_net 3시드) → n22(ε 6000) → n17([32,32])
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until grep -q "chain_isaac_v3 완료" $L; do sleep 300; done
for g in n16_obsdiv n18_evalnet n22_epsdecay n17_net32; do
  [ -f configs/grids/$g.yaml ] || continue
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_all 완료" >> $L
