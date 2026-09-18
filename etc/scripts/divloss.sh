#!/usr/bin/env bash
# ★09-17 사용자 기준으로 재측정: ①reward 수렴속도·최고성능 유지 ②loss 감소 패턴.
#   CartPole/LL 은 항상 0~1 입력 정규화 위에서 검증됐다. 드론 세 프레임워크에서
#   관측 /4 유무가 loss 에 어떤 차이를 만드는지 본다. loss·T_Var 로깅은 오늘 추가했다
#   (learn() 이 (loss, dt_ms, q_target.var()) 를 반환하는데 두 스크립트가 버리고 있었다).
# 학습기는 사용자 지정: SWIRL 기본 · Adam3e-4ams · EKF-TD · UKF-TD
set -u
cd /home/acsl/projects/Issacsim-rhukf
L=SWIRL,Adam3e-4ams,EKF-TD,UKF-TD
for SD in 42 43 44; do
  for DIV in 1.0 4.0; do
    TAG=$(echo $DIV | tr -d .)
    # 3번 온라인 정지 — 교정 무대(v5e·EP_RNG=1), P₀0.01·R3(오늘 3시드 최적)
    STOP_EPISODES=300 STOP_EPS_FRAC=0.50 STOP_EP_STEPS=300 STOP_PSTOP=0.10 \
    STOP_L=6 STOP_GAMMA=0.9 SURR_CLIP=1e9 SURR_OBS_CLIP=1e9 \
    SURR_POOL=/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/train_pool_v5e.npz SURR_EP_RNG=1 SURR_OBS_DIV=$DIV \
    ATK_DELTA_LO=0.10 ATK_START_LO=30 ATK_START_HI=120 \
    SW_P=0.01 SW_R=3 STOP_SEED=$SD STOP_LEARNERS=$L \
    STOP_OUT=results/claudecodefortest/divloss/q3_d${TAG}_s${SD}.json \
      python3 etc/scripts/online_stop.py > logs/divloss/q3_d${TAG}_s${SD}.log 2>&1 &
  done
done
wait
echo "DIVLOSS_DONE $(date +%H:%M)"
