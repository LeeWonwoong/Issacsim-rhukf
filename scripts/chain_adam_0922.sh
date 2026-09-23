#!/usr/bin/env bash
# 09-22 02:55: n41(Adam B128)·n42(lr×β) 완료 → pick_adam.py → isaac_v4/ADAM_SET (chain_isaac_v4 가 Adam 런 직전에 읽는다)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
until [ -f results/claudecodefortest/n41_batch128/GRID_DONE ] && [ -f results/claudecodefortest/n42_adam_lr_huber/GRID_DONE ]; do sleep 120; done
mkdir -p results/claudecodefortest/isaac_v4
python3 scripts/pick_adam.py > results/claudecodefortest/isaac_v4/ADAM_SET.tmp && mv results/claudecodefortest/isaac_v4/ADAM_SET.tmp results/claudecodefortest/isaac_v4/ADAM_SET \
  || { echo "--set agent.batch=256 --set agent.adam.lr=0.0003 --set agent.adam.huber_beta=5.0" > results/claudecodefortest/isaac_v4/ADAM_SET; echo "- $(date '+%H:%M') chain_adam_0922: 판정 실패 → 기본" >> results/claudecodefortest/NIGHT_0919.md; }
echo "- $(date '+%H:%M') chain_adam_0922: ADAM_SET = $(cat results/claudecodefortest/isaac_v4/ADAM_SET)" >> results/claudecodefortest/NIGHT_0919.md
