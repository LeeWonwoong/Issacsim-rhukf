#!/usr/bin/env bash
# 풀 A 종료 대기 → Isaac Adam 학습(스텝 기록) → 풀 B(강풍)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
R=results/claudecodefortest
until [ -f $R/capture_pool/RUN_DONE ]; do sleep 30; done
echo "[$(date '+%m-%d %H:%M')] 풀 A 종료 rc=$(cat $R/capture_pool/RUN_DONE) → Isaac Adam 학습" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/isaac_adam_probe --config configs/isaac_adam_probe.yaml --config configs/overlays/isaac.yaml
echo "[$(date '+%m-%d %H:%M')] Isaac Adam 종료 rc=$(cat $R/isaac_adam_probe/RUN_DONE) → 풀 B" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/capture_pool_b --config configs/capture_pool_b.yaml --config configs/overlays/isaac.yaml --config configs/overlays/wind_strong.yaml
echo "[$(date '+%m-%d %H:%M')] 풀 B 종료 rc=$(cat $R/capture_pool_b/RUN_DONE)" >> $R/chain_capture.log
