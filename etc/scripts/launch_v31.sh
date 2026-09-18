#!/usr/bin/env bash
# v31 (09-14 밤): v30 완료(GPU DONE 8개) 대기 → aliasing 다이얼(g4·ws10 0.5·스텔스 off·P0.2·가족 v6) × SW·UKF·EKF(GPU 75) + Adam(CPU 25), 시드 5.
# 실행: setsid nohup bash etc/scripts/launch_v31.sh > results/claudecodefortest/night/v30_autolaunch.log 2>&1 < /dev/null &
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
NW_GPU=${NW_GPU:-8}
# v29 완료 + 풀 v5d(cert_SW 반영) 존재를 기다림. 23:30 까지 v5d 가 없으면 v5c 로 진행(env_scan_v30 이 존재 여부로 선택).
while [ "$(ls $N | grep -c 'V30gpu_w.*_DONE')" -lt 8 ]; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] v30 완료 → v31 착수"
rm -f $N/v31*_w*.json $N/V31*_DONE
for w in $(seq 0 $((NW_GPU-1))); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v31.py $w $NW_GPU > $N/v31gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2 3; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v31.py $w 4 > $N/v31cpu_w$w.log 2>&1 < /dev/null & done
sleep 20; echo "v30 workers=$(ps aux | grep -c '[e]nv_scan_v31.py')  SWIRL cfg: $(cat $N/V31_SWIRL_CFG 2>/dev/null)"
echo "[$(date '+%m-%d %H:%M')] v31 aliasing 다이얼+ablation (GPU $NW_GPU 워커 75 런 + CPU 25) START" >> $N/v5_launch.log
