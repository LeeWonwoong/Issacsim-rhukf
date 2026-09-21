#!/usr/bin/env bash
# 09-21 18:40 사용자 지시: n35b 완료 후 n36(바람 펄스: 약풍→20에피 강풍→약풍) SWIRL pΔ0.2/0.1 + Adam 3시드
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n35b_axisB_pd01/GRID_DONE ]; do sleep 120; done
echo "- $(date '+%H:%M') (09-21) chain_0921f: n36_windpulse 시작" >> $L
python3 scripts/grid.py run configs/grids/n36_windpulse.yaml --gpu 3 --cpu 3 --omp-cpu 2 > results/claudecodefortest/n36_windpulse.launch.log 2>&1
until [ -f results/claudecodefortest/n36_windpulse/GRID_DONE ]; do sleep 120; done
echo "- $(date '+%H:%M') (09-21) chain_0921f: n36_windpulse 완료" >> $L
