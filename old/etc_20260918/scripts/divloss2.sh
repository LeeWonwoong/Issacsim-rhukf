#!/usr/bin/env bash
# 2번만 재실행. 앞선 실패는 STOP_DATA 미지정으로 구 8D 데이터셋(stop_dataset_50k.npz)이
# 잡혀 dimS 12 와 불일치한 것이다. 어젯밤 파이프라인은 _L6 파일을 명시하고 있었다.
set -u
cd /home/acsl/projects/Issacsim-rhukf
N=results/claudecodefortest/night
L=SWIRL,Adam3e-4ams,EKF-TD,UKF-TD
for SD in 42 43 44; do
  for DIV in 1.0 4.0; do
    TAG=$(echo $DIV | tr -d .)
    STOP_DATA=$N/stop_dataset_50k_L6.npz \
    STOP_EPISODES=300 STOP_EPS_FRAC=0.50 STOP_PSTOP=0.10 STOP_L=6 \
    STOP_OBS_DIV=$DIV SW_P=0.30 SW_R=1 STOP_SEED=$SD STOP_LEARNERS=$L \
    STOP_OUT=results/claudecodefortest/divloss/f2_d${TAG}_s${SD}.json \
      python3 etc/scripts/offline_stop.py > logs/divloss/f2_d${TAG}_s${SD}.log 2>&1 &
  done
done
wait
echo "DIVLOSS2_DONE $(date +%H:%M)"
