#!/usr/bin/env bash
# 09-22 21:30 사용자: R 을 먼저 맞춘 뒤 P·N. 순서: (③ SWIRL FN1.0 진행 중) → ③ Adam FN1.0 → ③b bonus6 S·A → ④ v5a S·A → R0.35 → R0.25  (P·N 은 아침에 R 확정 후)
#   CP/LL 근거: Isaac 잔차분산(loss) R0.5 에서 0.32, v2 승리 레짐은 R≈1.05×잔차분산 → R≈0.35 가 규칙상 목표, 0.25 는 아래쪽 브래킷(K≈0.40 예상).
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v8
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=256 --set agent.swirl.N=6"
SW="--set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1"
until [ -f $B/swirl_ui1_P01_R1_FN10_s42/RUN_DONE ]; do sleep 60; done
[ -f $B/swirl_ui1_P01_R1_FN10_s42/final_model.pt ] && [ ! -s $B/swirl_ui1_P01_R1_FN10_s42/eval_surrogate.log ] && python3 scripts/eval_model.py $B/swirl_ui1_P01_R1_FN10_s42 --n 100 --device cpu >> $B/swirl_ui1_P01_R1_FN10_s42/eval_surrogate.log 2>&1
run(){ local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return
  echo "- $(date '+%H:%M') chain_0922k: $(basename $OUT) 시작" >> $L
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922k: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1; }
run $B/adam_FN10_s42             configs/isaac_v5_adam.yaml   --set reward.c_d=1.0
run $B/swirl_ui1_P01_R1_B6_s42   configs/isaac_v5_swirl.yaml  $UI $SW --set agent.swirl.R=1.0 --set reward.bonus=6.0
run $B/adam_B6_s42               configs/isaac_v5_adam.yaml   --set reward.bonus=6.0
run $B/swirl_v5a_s42             configs/isaac_v5a_swirl.yaml $UI $SW --set agent.swirl.R=1.0
run $B/adam_v5a_s42              configs/isaac_v5a_adam.yaml
run $B/swirl_ui1_P01_R035_s42    configs/isaac_v5_swirl.yaml  $UI $SW --set agent.swirl.R=0.35
run $B/swirl_ui1_P01_R025_s42    configs/isaac_v5_swirl.yaml  $UI $SW --set agent.swirl.R=0.25
echo "- $(date '+%H:%M') chain_0922k 완료 (P·N 은 R 확정 후)" >> $L; touch $B/CHAIN_DONE
