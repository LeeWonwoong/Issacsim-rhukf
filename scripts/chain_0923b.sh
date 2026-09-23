#!/usr/bin/env bash
# 09-23 01:30 순차 실행: n55(ρ 스윕 2시드) → n54(정합검증 2런) → chain_0923 이 Isaac 짝을 집는다.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
python3 scripts/grid.py run configs/grids/n55_rho_sweep.yaml --gpu 6 --cpu 3 --omp-cpu 2 >> results/claudecodefortest/n55.launch.log 2>&1
echo "- $(date '+%H:%M') chain_0923b: n55 완료 → n54 정합검증" >> $L
python3 scripts/grid.py run configs/grids/n54_fidelity.yaml --gpu 2 --cpu 0 --omp-cpu 4 >> results/claudecodefortest/n54.launch.log 2>&1
echo "- $(date '+%H:%M') chain_0923b: n54 완료" >> $L
