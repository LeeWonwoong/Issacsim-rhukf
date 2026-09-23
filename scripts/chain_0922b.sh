#!/usr/bin/env bash
# 09-22 06:12 Claude 판단: Isaac s42 에서 B128·N10 SWIRL 이 Adam(lr1e-3·β5) 에 열세(리턴 273.7 vs 281.8, 공격F1 0.656 vs 0.761).
#   n41 판정은 0.005 차 동률이었고 꼬리는 B256·N6 이 좋았으므로, n29b s42 가 GPU 를 비우면 SWIRL s42 를 B256·N6 으로 재실행(Adam s42 는 그대로 짝비교)
#   → 이어서 시드 43·44 도 B256·N6. 원 s42(B128·N10) 결과는 swirl_s42_B128N10 으로 보관.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ -f results/claudecodefortest/n29b_ktd_q_v4_s42/GRID_DONE ]; do sleep 120; done
[ -d results/claudecodefortest/isaac_v4/swirl_s42 ] && [ ! -d results/claudecodefortest/isaac_v4/swirl_s42_B128N10 ] && mv results/claudecodefortest/isaac_v4/swirl_s42 results/claudecodefortest/isaac_v4/swirl_s42_B128N10
echo "- $(date '+%H:%M') chain_0922b: n29b s42 완료 → SWIRL B256·N6 로 Isaac 재개(s42 재실행 → s43 → s44), Adam 은 ADAM_SET(lr1e-3·β5·B256)" >> $L
AD_WAIT=1 WIND="[0.0, 9.0]" SW_SET="--set agent.batch=256 --set agent.swirl.N=6" AD_SET="$(cat results/claudecodefortest/isaac_v4/ADAM_SET)" bash scripts/chain_isaac_v4.sh
