#!/usr/bin/env bash
# 분기 캡처 종료 대기 → 풀 수집 A(U(0,6) 300) → 풀 수집 B(U(6,10) 100). 각 단계는 scripts/isaac_run.sh.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
R=results/claudecodefortest
until [ -f $R/capture_branching/RUN_DONE ]; do sleep 30; done
echo "[$(date '+%m-%d %H:%M')] 분기 캡처 종료 → 풀 A" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/capture_pool --config configs/capture_pool.yaml --config configs/overlays/isaac.yaml
echo "[$(date '+%m-%d %H:%M')] 풀 A 종료 rc=$(cat $R/capture_pool/RUN_DONE) → 풀 B" >> $R/chain_capture.log
bash scripts/isaac_run.sh $R/capture_pool_b --config configs/capture_pool_b.yaml --config configs/overlays/isaac.yaml --config configs/overlays/wind_strong.yaml
echo "[$(date '+%m-%d %H:%M')] 풀 B 종료 rc=$(cat $R/capture_pool_b/RUN_DONE)" >> $R/chain_capture.log
