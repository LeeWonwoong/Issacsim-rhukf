#!/usr/bin/env bash
# chain_night2b 후속: n7 → n2/n4 결손 재실행 (n1·n3·n5 는 별도 grid.py 로 실행 중), 동시 ≤24
cd /home/acsl/projects/Issacsim-rhukf || exit 1
for g in n7_cfa n2_b128n10 n4_wind8; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 24 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] chain_night2c: $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2
done
until [ "$(ps -eo args | grep -v grep | grep -c 'grid.py run configs/grids/n[135]_')" -eq 0 ]; do sleep 300; done
echo "[$(date '+%m-%d %H:%M')] chain_night2 완료" >> results/claudecodefortest/NIGHT_0919.md
