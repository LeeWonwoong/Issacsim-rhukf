#!/usr/bin/env bash
# P₀ 를 3번(online_stop) 팔 B 무대에서 직접 측정한다.
# 3번 본선은 P₀=0.01 로 돌았는데, 그 값은 튜닝 결과가 아니라 파이프라인 상수였다
# (g2on.log 에 P 스캔 축 자체가 없음). online_stop.py 기본값 0.03 을 덮어쓴 것이다.
#
# ★09-17 수정: 3번 전체가 구 v3 풀(42키·최대3.0)로 돌고 있었다. surrogate_env8.py:21 기본값이고
#   online_stop.py 는 SURR_POOL 을 설정하지 않는다. 1번 무대(chain_hz.py:26)는 v5e 를 쓴다.
#   SURR_EP_RNG 도 기본 0 이라 짝 t검정이 진짜 짝이 아니었다(chain_hz.py:30 은 1 로 켬).
#   두 결함 모두 1번 무대에 맞춰 교정.
# 300ep · EPS_FRAC=0.50  →  rate 는 n×FRAC=150 에만 의존하므로
# 500ep·0.30 본선과 ε(ep) 곡선이 완전히 동일 = 기존 런의 정확한 앞 300에피.
set -u
cd /home/acsl/projects/Issacsim-rhukf
OUT=results/claudecodefortest/p0stage3
LOG=logs/p0stage3
for P0 in 0.003 0.01 0.03 0.10; do
  for SD in 42 43 44; do
    STOP_EPISODES=300 STOP_EPS_FRAC=0.50 \
    STOP_EP_STEPS=300 STOP_PSTOP=0.10 STOP_L=6 STOP_GAMMA=0.9 \
    SURR_CLIP=1e9 SURR_OBS_CLIP=1e9 SURR_OBS_DIV=4.0 \
    SURR_POOL=/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/train_pool_v5e.npz SURR_EP_RNG=1 \
    ATK_DELTA_LO=0.10 ATK_START_LO=30 ATK_START_HI=120 \
    SW_P=$P0 SW_R=2 \
    STOP_SEED=$SD STOP_LEARNERS=SWIRL,Adam1e-3 \
    STOP_OUT=$OUT/p${P0}_s${SD}.json \
      python3 etc/scripts/online_stop.py > $LOG/p${P0}_s${SD}.log 2>&1 &
  done
done
wait
echo "P0_STAGE3_DONE $(date +%H:%M)"
