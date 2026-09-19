#!/usr/bin/env bash
# Isaac 본선(v2 무대): surrogate 격자가 전부 끝나 GPU 가 비면(train.py 0 · chain_night3 완료) 시드 42→43→44 순으로 SWIRL·Adam 을 순차 실행.
#   각 런 뒤 저장 모델을 surrogate greedy 100ep 로 교차평가(eval_model.py). SWIRL 은 GPU 단독 필수(learn ≤ 40 ms 예산).
cd /home/acsl/projects/Issacsim-rhukf || exit 1
until grep -q "chain_night3 완료" results/claudecodefortest/NIGHT_0919.md && [ "$(ps -eo args | grep -v grep | grep -c 'train.py')" -eq 0 ]; do sleep 300; done
echo "[$(date '+%m-%d %H:%M')] chain_isaac: GPU 비움 확인 → Isaac 본선 시작" >> results/claudecodefortest/NIGHT_0919.md
for s in 42 43 44; do
  for L in swirl adam; do
    OUT=results/claudecodefortest/isaac_v2/${L}_s$s
    [ -f "$OUT/hist.json" ] && continue
    bash scripts/isaac_run.sh "$OUT" --config configs/isaac_v2_$L.yaml --config configs/overlays/isaac.yaml --set run.seed=$s
    echo "[$(date '+%m-%d %H:%M')] chain_isaac: $L s$s rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> results/claudecodefortest/NIGHT_0919.md
    [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
  done
done
echo "[$(date '+%m-%d %H:%M')] chain_isaac 완료" >> results/claudecodefortest/NIGHT_0919.md
