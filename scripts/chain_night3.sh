#!/usr/bin/env bash
# chain_night2 완료 뒤 n8 → n11 → n10 (각 --gpu 4). n9(앵커=현재 추정)는 사용자 판단으로 제외: 창 추정치를 연쇄시키면 반복 UKF(IIR)가 되어 FIR 이 아님
cd /home/acsl/projects/Issacsim-rhukf || exit 1
until grep -q "chain_night2 완료" results/claudecodefortest/NIGHT_0919.md; do sleep 600; done
for g in n8_center_pdelta n11_inep_shift n10_cd0; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 24 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] chain_night3: $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 3 --omp-cpu 2
done
echo "[$(date '+%m-%d %H:%M')] chain_night3 완료" >> results/claudecodefortest/NIGHT_0919.md
