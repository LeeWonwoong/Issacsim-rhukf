#!/usr/bin/env bash
# 09-22 16:05 사용자 순서: (A′ 판정·조종기 리허설 끝난 뒤 '고') → ④ Isaac s42 SWIRL v5 B 노브 + Adam 과 같은 갱신 스케줄(τ0.005·ui1) → ⑤ Isaac s42 Adam v5 → v5 짝 완성
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v7
echo "- $(date '+%H:%M') chain_0922g 시작: Isaac s42 SWIRL v5 B-ui1 (pΔ0.01·α0.1·R1·τ0.005·ui1)" >> $L
bash scripts/isaac_run.sh $B/swirlBui1_s42 --config configs/isaac_v5_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/swirlBui1_s42 --set agent.tau=0.005 --set agent.update_interval=1
echo "- $(date '+%H:%M') chain_0922g: swirlBui1_s42 rc=$(cat $B/swirlBui1_s42/RUN_DONE)" >> $L
[ -f $B/swirlBui1_s42/final_model.pt ] && python3 px4_field/export_policy.py $B/swirlBui1_s42/final_model.pt px4_field/models/swirl_v5Bui1_s42.npz >> $B/export.log 2>&1
echo "- $(date '+%H:%M') chain_0922g: Isaac s42 Adam v5 (3e-4·β5·clip10·B256·ui1·τ0.005) 시작" >> $L
bash scripts/isaac_run.sh $B/adam_s42 --config configs/isaac_v5_adam.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/adam_s42
echo "- $(date '+%H:%M') chain_0922g: adam_s42 rc=$(cat $B/adam_s42/RUN_DONE)" >> $L
for d in swirlB_s42 swirlA2_s42 swirlBui1_s42 adam_s42; do [ -f $B/$d/final_model.pt ] && [ ! -s $B/$d/eval_surrogate.log ] && python3 scripts/eval_model.py $B/$d --n 100 --device cpu >> $B/$d/eval_surrogate.log 2>&1; done
echo "- $(date '+%H:%M') chain_0922g 완료 — v5 Isaac 짝(SWIRL B / A′ / B-ui1 vs Adam, s42)" >> $L; touch $B/PAIR_DONE
