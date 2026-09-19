#!/usr/bin/env bash
# chain_night2 후속(n6 는 별도 grid.py 로 계속 실행 중): n1 → n3 → n5 → n7 → n2/n4 결손 재실행, 동시 ≤24
cd /home/acsl/projects/Issacsim-rhukf || exit 1
for g in n1_ktd_std n3_pdelta n5_sparse n7_cfa n2_b128n10 n4_wind8; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 24 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] chain_night2b: $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2
done
echo "[$(date '+%m-%d %H:%M')] chain_night2 완료" >> results/claudecodefortest/NIGHT_0919.md
