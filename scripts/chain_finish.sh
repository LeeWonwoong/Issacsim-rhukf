#!/usr/bin/env bash
# 모든 grid.py 가 끝나면 'chain_night3 완료' 표식을 남겨 chain_isaac 이 Isaac 본선을 시작하게 한다
cd /home/acsl/projects/Issacsim-rhukf || exit 1
sleep 120
until [ "$(ps -eo args | grep -v grep | grep -c 'grid.py run')" -eq 0 ]; do sleep 300; done
echo "[$(date '+%m-%d %H:%M')] 밤샘 격자 전부 종료 → chain_night3 완료" >> results/claudecodefortest/NIGHT_0919.md
