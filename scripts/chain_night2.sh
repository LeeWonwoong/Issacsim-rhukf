#!/usr/bin/env bash
# GPU 부하가 내려가면(train.py ≤ 28) 밤샘 격자 n3 → n5 → n7 을 순서대로 시작 (각각 --gpu 3)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
for g in n3_pdelta n5_sparse n7_cfa; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 28 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 3 --cpu 3 --omp-cpu 2
done
