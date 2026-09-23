#!/usr/bin/env bash
# 09-22 02:50 사용자: n41(B128·N10/12, Adam B128) 2시드 → 최선 config 로 Isaac 3시드 밤샘 + Adam 창에서 n29b(칼만-TD) 튜닝.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n41_batch128/GRID_DONE ]; do sleep 120; done
read B N < <(python3 scripts/pick_n41.py) || { echo "- $(date '+%H:%M') chain_0922: 판정 실패 → 기본 B256·N6" >> $L; B=256; N=6; }
sed -i "s/agent.batch: [0-9]*/agent.batch: $B/" configs/grids/n29b_s42.yaml configs/grids/n29b_s43.yaml configs/grids/n29b_s44.yaml
echo "- $(date '+%H:%M') chain_0922: Isaac 3시드 기동 SWIRL B$B·N$N·pΔ0.1·α0.05, Adam B$B, 바람 U(0,9), n29b batch $B" >> $L
AD_WAIT=1 WIND="[0.0, 9.0]" SW_SET="--set agent.batch=$B --set agent.swirl.N=$N" AD_SET="--set agent.batch=$B" bash scripts/chain_isaac_v4.sh
