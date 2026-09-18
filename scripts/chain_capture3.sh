#!/usr/bin/env bash
# Isaac Adam 학습(되는지 확인 + 실제 분포, 스텝 기록) → 풀 A(U(0,6) 300) → 풀 B(U(6,10) 100)
cd /home/acsl/projects/Issacsim-rhukf || exit 1
R=results/claudecodefortest
echo "[$(date '+%m-%d %H:%M')] Isaac Adam 학습 시작" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/isaac_adam_probe --config configs/isaac_adam_probe.yaml --config configs/overlays/isaac.yaml
echo "[$(date '+%m-%d %H:%M')] Isaac Adam 종료 rc=$(cat $R/isaac_adam_probe/RUN_DONE) → 풀 A" >> $R/chain_capture.log
rm -rf $R/capture_pool
bash scripts/isaac_run.sh $R/capture_pool --config configs/capture_pool.yaml --config configs/overlays/isaac.yaml
echo "[$(date '+%m-%d %H:%M')] 풀 A 종료 rc=$(cat $R/capture_pool/RUN_DONE) → 풀 B" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/capture_pool_b --config configs/capture_pool_b.yaml --config configs/overlays/isaac.yaml --config configs/overlays/wind_strong.yaml
echo "[$(date '+%m-%d %H:%M')] 풀 B 종료 rc=$(cat $R/capture_pool_b/RUN_DONE)" >> $R/chain_capture.log
