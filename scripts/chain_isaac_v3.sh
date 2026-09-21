#!/usr/bin/env bash
# Isaac 본선 v3(09-21 확정): GPU 가 비면(train.py 0) 시드 42→43→44 순으로 SWIRL(B)·Adam(c5·β5·clip10) 순차 + surrogate 교차평가. 실행 전 surrogate 체인을 멈춰 GPU 를 비울 것.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md
until [ "$(ps -eo args | grep -v grep | grep -c 'train.py')" -eq 0 ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] chain_isaac_v3: GPU 비움 → 시작" >> $L
for s in 42 43 44; do
  for Lr in swirl adam; do
    OUT=results/claudecodefortest/isaac_v3/${Lr}_s$s
    [ -f "$OUT/RUN_DONE" ] && continue
    bash scripts/isaac_run.sh "$OUT" --config configs/isaac_v3_$Lr.yaml --config configs/overlays/isaac.yaml --set run.seed=$s
    echo "[$(date '+%m-%d %H:%M')] chain_isaac_v3: $Lr s$s rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
    [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
  done
done
echo "[$(date '+%m-%d %H:%M')] chain_isaac_v3 완료" >> $L
