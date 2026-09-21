#!/usr/bin/env bash
# n24 완료 → n27(B 튜닝 R·N) — 끝나면 판정 후 Isaac v3 를 수동 시작(scripts/chain_isaac_v3.sh)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n24_pdelta02_ui1/GRID_DONE ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] chain_all: n24 완료 → n27_B_tune 시작" >> $L
python3 scripts/grid.py run configs/grids/n27_B_tune.yaml --gpu 4 --cpu 4 --omp-cpu 2 > results/claudecodefortest/n27_B_tune.launch.log 2>&1
until [ -f results/claudecodefortest/n27_B_tune/GRID_DONE ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] chain_all: n27_B_tune 완료 (Isaac v3 수동 시작 대기)" >> $L
