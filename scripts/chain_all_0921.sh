#!/usr/bin/env bash
# 09-21 체인(사용자 재정렬 v2): n13 완료 → n21(고-K×α @ui1) → n14 → n19 → n13b → n16 → n18 → n17 → n22(ε decay 6000)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n13_pdelta_ut/GRID_DONE ]; do sleep 120; done
for g in n21_highK_ui1 n14_cd n19_cfa_cd n13b_alpha n16_obsdiv n18_evalnet n17_net32 n22_epsdecay; do
  [ -f configs/grids/$g.yaml ] || { echo "[$(date '+%m-%d %H:%M')] chain_all: $g yaml 없음 → 건너뜀" >> $L; continue; }
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_all 완료" >> $L
