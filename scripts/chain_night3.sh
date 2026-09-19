#!/usr/bin/env bash
# chain_night2 완료 뒤(또는 GPU 여유) n8 → n9 → n11 → n10 (각 --gpu 4)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
until grep -q "chain_night2 완료" results/claudecodefortest/NIGHT_0919.md; do sleep 600; done
for g in n8_center_pdelta n9_anchor_current n11_inep_shift n10_cd0; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 24 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] chain_night3: $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 3 --omp-cpu 2
done
echo "[$(date '+%m-%d %H:%M')] chain_night3 완료" >> results/claudecodefortest/NIGHT_0919.md
