#!/usr/bin/env bash
# chain_n13_15 완료 → n16(obs div) → n17(net32) → n18(act=active+eval_net target, yaml 있을 때만)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until grep -q "chain_n13_15 완료" $L; do sleep 180; done
for g in n16_obsdiv n17_net32 n18_evalnet; do
  [ -f configs/grids/$g.yaml ] || { echo "[$(date '+%m-%d %H:%M')] chain_n16_18: $g yaml 없음 → 건너뜀" >> $L; continue; }
  echo "[$(date '+%m-%d %H:%M')] chain_n16_18: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_n16_18: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_n16_18 완료" >> $L
