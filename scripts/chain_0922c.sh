#!/usr/bin/env bash
# 09-22 10:15 사용자 승인: Isaac s42 · U(0,8) · B256 · 희소 보상 (R₀,C_FP,C_FN,R_d)=(1,2,1,5) 로 3런
#   ① Adam 3e-4·β5 (CPU, 지금 즉시)  ② SWIRL 현 노브 pΔ0.1·α0.05·τ0.005·ui1 (n43 GPU 비운 뒤)  ③ SWIRL v2 노브 pΔ0.01·α0.1·τ0.02·ui4
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v5
COMMON=(--config configs/overlays/isaac.yaml --set run.seed=42 "--set" "scenario.wind.range=[0.0, 8.0]" --set reward.c_d=1.0 --set reward.bonus=5.0 --set agent.batch=256)
run(){ # $1 outdir  $2 base yaml  $3... extra sets
  local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" "${COMMON[@]}" --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922c: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
}
echo "- $(date '+%H:%M') chain_0922c 시작: Adam s42 (1,2,1,5)·U(0,8)·B256" >> $L
run $B/adam_s42 configs/isaac_v4_adam.yaml
until [ -f results/claudecodefortest/n43_wind08_b256/GRID_DONE ]; do sleep 60; done
echo "- $(date '+%H:%M') chain_0922c: n43 완료 → SWIRL 현 노브" >> $L
run $B/swirl_s42 configs/isaac_v4_swirl.yaml --set agent.swirl.N=6
echo "- $(date '+%H:%M') chain_0922c: SWIRL v2 노브" >> $L
run $B/swirl_v2knobs_s42 configs/isaac_v4_swirl.yaml --set agent.swirl.N=6 --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.tau=0.02 --set agent.update_interval=4
echo "- $(date '+%H:%M') chain_0922c 완료" >> $L; touch $B/CHAIN_DONE
