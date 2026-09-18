#!/usr/bin/env bash
# v30 (09-14 저녁): v29 완료(GPU DONE 8개) 대기 → 강풍 블록 급변 축. GPU 60 런(블록/혼합 × 버퍼 50k/10k × SW·UKF·EKF × 시드 5) + CPU Adam 20 런.
# 실행: setsid nohup bash etc/scripts/launch_v30.sh > results/claudecodefortest/night/v30_autolaunch.log 2>&1 < /dev/null &
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
NW_GPU=${NW_GPU:-8}
# v29 완료 + 풀 v5d(cert_SW 반영) 존재를 기다림. 23:30 까지 v5d 가 없으면 v5c 로 진행(env_scan_v30 이 존재 여부로 선택).
while [ "$(ls $N | grep -c 'V29gpu_w.*_DONE')" -lt 8 ] || { [ ! -f $N/train_pool_v5d.npz ] && [ "$(date +%H%M)" -lt 2330 ]; }; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] v29 완료 → v30 착수"
rm -f $N/v30*_w*.json $N/V30*_DONE
for w in $(seq 0 $((NW_GPU-1))); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v30.py $w $NW_GPU > $N/v30gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2 3; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup python3 etc/scripts/env_scan_v30.py $w 4 > $N/v30cpu_w$w.log 2>&1 < /dev/null & done
sleep 20; echo "v30 workers=$(ps aux | grep -c '[e]nv_scan_v30.py')  SWIRL cfg: $(cat $N/V30_SWIRL_CFG 2>/dev/null)"
echo "[$(date '+%m-%d %H:%M')] v30 강풍 블록 급변 축 (GPU $NW_GPU 워커 60 런 + CPU 20) START" >> $N/v5_launch.log
