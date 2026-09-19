#!/usr/bin/env bash
# GPU 동시 실행 ≤ 24 일 때만 다음 격자 시작 (OOM 방지: 런당 ~1.2 GB, 48 GB) — 순서: 프로브16 → 칼만-TD 표준 → pΔ → 희소 → c_fa → (OOM 결손 재실행) B128N10 → 바람8
cd /home/acsl/projects/Issacsim-rhukf || exit 1
for g in n6_probe16 n1_ktd_std n3_pdelta n5_sparse n7_cfa n2_b128n10 n4_wind8; do
  until [ "$(ps -eo args | grep -v grep | grep -c train.py)" -le 24 ]; do sleep 300; done
  echo "[$(date '+%m-%d %H:%M')] chain_night2: $g 시작" >> results/claudecodefortest/NIGHT_0919.md
  python3 scripts/grid.py run configs/grids/$g.yaml --gpu 4 --cpu 4 --omp-cpu 2
done
echo "[$(date '+%m-%d %H:%M')] chain_night2 완료" >> results/claudecodefortest/NIGHT_0919.md
