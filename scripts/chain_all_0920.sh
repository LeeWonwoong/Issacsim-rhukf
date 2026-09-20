#!/usr/bin/env bash
# 09-20 밤 체인(재정렬): n12 완료 → n13(pΔ 표준UT) → n14(c_d) → n19(c_fa×c_d) → n15(q 사다리) → n16(obs /4) → n18(eval_net). yaml 없으면 건너뜀.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n12_gamma_nstep/GRID_DONE ]; do sleep 120; done
for g in n13_pdelta_ut n14_cd n19_cfa_cd n15_ladder_q n16_obsdiv n18_evalnet; do
  [ -f configs/grids/$g.yaml ] || { echo "[$(date '+%m-%d %H:%M')] chain_all: $g yaml 없음 → 건너뜀" >> $L; continue; }
  [ -f results/claudecodefortest/$g/GRID_DONE ] && continue
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 시작" >> $L
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/$g.launch.log 2>&1
  until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] chain_all: $g 완료" >> $L
done
echo "[$(date '+%m-%d %H:%M')] chain_all 완료" >> $L
