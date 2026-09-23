#!/usr/bin/env bash
# 09-22 19:45 사용자: SWIRL 은 Adam 과 같은 스케줄(ui1·τ0.005·B256) 고정, 그 안에서 P·R·N 정밀 튜닝. Isaac s42 순서: ② R0.5 → ③ 보상 FN1.0(SWIRL+Adam) → ④ P0.05·R1 → ⑤ P0.05·R0.5 → ⑥ P0.1·R1
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v8; mkdir -p $B
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=256 --set agent.swirl.N=6"
run(){ local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return
  echo "- $(date '+%H:%M') chain_0922h: $(basename $OUT) 시작" >> $L
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922h: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1; }
run $B/swirl_ui1_P01_R05_s42  configs/isaac_v5_swirl.yaml $UI --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.R=0.5
run $B/swirl_ui1_P01_R1_FN10_s42 configs/isaac_v5_swirl.yaml $UI --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.R=1.0 --set reward.c_d=1.0
run $B/adam_FN10_s42          configs/isaac_v5_adam.yaml  --set reward.c_d=1.0
run $B/swirl_ui1_P05_R1_s42   configs/isaac_v5_swirl.yaml $UI --set agent.swirl.p_delta=0.05 --set agent.swirl.alpha=0.05 --set agent.swirl.R=1.0
run $B/swirl_ui1_P05_R05_s42  configs/isaac_v5_swirl.yaml $UI --set agent.swirl.p_delta=0.05 --set agent.swirl.alpha=0.05 --set agent.swirl.R=0.5
run $B/swirl_ui1_P1_R1_s42    configs/isaac_v5_swirl.yaml $UI --set agent.swirl.p_delta=0.1 --set agent.swirl.alpha=0.05 --set agent.swirl.R=1.0
echo "- $(date '+%H:%M') chain_0922h 완료" >> $L; touch $B/CHAIN_DONE
