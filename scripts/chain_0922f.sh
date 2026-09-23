#!/usr/bin/env bash
# 09-22 15:12 사용자: "v5 에서 Isaac 짝을 맞춘 후 논의" → chain_0922e(SWIRL B → A′) 끝나면 Adam v5 Isaac s42 (짝 완성)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v7
until [ -f $B/CHAIN_DONE ]; do sleep 60; done
echo "- $(date '+%H:%M') chain_0922f: Isaac s42 Adam v5 (3e-4·β5·clip10·B256) 시작 — v5 짝 완성" >> $L
bash scripts/isaac_run.sh $B/adam_s42 --config configs/isaac_v5_adam.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/adam_s42
echo "- $(date '+%H:%M') chain_0922f: adam_s42 rc=$(cat $B/adam_s42/RUN_DONE)" >> $L
[ -f $B/adam_s42/final_model.pt ] && python3 scripts/eval_model.py $B/adam_s42 --n 100 --device cpu >> $B/adam_s42/eval_surrogate.log 2>&1
[ -f $B/swirlB_s42/final_model.pt ] && [ ! -f $B/swirlB_s42/eval_surrogate.log ] && python3 scripts/eval_model.py $B/swirlB_s42 --n 100 --device cpu >> $B/swirlB_s42/eval_surrogate.log 2>&1
[ -f $B/swirlA2_s42/final_model.pt ] && [ ! -f $B/swirlA2_s42/eval_surrogate.log ] && python3 scripts/eval_model.py $B/swirlA2_s42 --n 100 --device cpu >> $B/swirlA2_s42/eval_surrogate.log 2>&1
echo "- $(date '+%H:%M') chain_0922f 완료 (v5 Isaac 짝: swirlB·swirlA2·adam s42)" >> $L; touch $B/PAIR_DONE
