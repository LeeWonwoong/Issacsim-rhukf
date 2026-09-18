#!/usr/bin/env bash
# v29 (09-14 오후): 새 무대(v5c 풀·env8 티어 혼합·하한 DELTA_LO) SWIRL 격자 24(N5·6·7) + UKF/EKF-TD 4 (GPU 84 런) + Adam(CPU 3 런), 시드 3.
# 사용: TIER_P="0:0.4,6:0.2,7:0.2,10:0.2" DELTA_LO=0.05 NW_GPU=8 bash etc/scripts/launch_v29.sh
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
export TIER_P="${TIER_P:-0:0.4,7:0.3,10:0.3}" SURR_PLOC_TIERS="${SURR_PLOC_TIERS:-7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1}" DELTA_LO="${DELTA_LO:-0.05}" POOL="${POOL:-$N/train_pool_v5c.npz}"
NW_GPU=${NW_GPU:-8}
[ -f "$POOL" ] || { echo "풀 없음: $POOL"; exit 1; }
rm -f $N/v29*_w*.json $N/V29*_DONE
for w in $(seq 0 $((NW_GPU-1))); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v29.py $w $NW_GPU > $N/v29gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v29.py $w 3 > $N/v29cpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v29 새 무대 SWIRL 격자(TIER_P=$TIER_P DELTA_LO=$DELTA_LO) GPU $NW_GPU 워커 84 런 + CPU 3 START" >> $N/v5_launch.log
