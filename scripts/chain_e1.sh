#!/usr/bin/env bash
# night10 SWIRL 9런 완료 대기 → E1(직전 프레임워크) 격자
cd /home/acsl/projects/Issacsim-rhukf || exit 1
until [ "$(ls results/claudecodefortest/night10/a_s*/hist.json 2>/dev/null | wc -l)" -ge 9 ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] night10 완료 → E1 시작" >> results/claudecodefortest/NIGHT_0919.md
python3 scripts/grid.py run configs/grids/e1_oldframework.yaml --gpu 5 --cpu 6 --omp-cpu 2
