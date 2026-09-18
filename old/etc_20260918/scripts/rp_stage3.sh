#!/usr/bin/env bash
# ★09-17 배치2: 본선 전 마지막 확인. 두 축을 교정된 무대(v5e·EP_RNG=1)에서 직접 잰다.
#   ⓐ P₀ 저단 확장 — 0.10 −0.50 / 0.03 −0.09 / 0.01 +1.07 / 0.003 +1.31 로 단조 개선 중이라
#      0.001 을 재서 평탄해지는지 확인한다.
#   ⓑ R 스캔 — 본선이 쓴 SW_R=2 는 구 v3 풀에서 1시드로 고른 값이고, 그 스캔은
#      R1 26.68 / R2 29.36 / R3 20.31 / R5 22.48 로 비단조(R3<R5) = 잡음 신호였다.
#      P₀=0.01 에 고정하고 R{1,2,3} 을 3시드로 다시 잰다. R1 은 online_stop.py 기본값이다.
set -u
cd /home/acsl/projects/Issacsim-rhukf
OUT=results/claudecodefortest/rp_stage3
LOG=logs/rp_stage3
run() {  # $1=P0 $2=R $3=seed $4=tag
  STOP_EPISODES=300 STOP_EPS_FRAC=0.50 \
  STOP_EP_STEPS=300 STOP_PSTOP=0.10 STOP_L=6 STOP_GAMMA=0.9 \
  SURR_CLIP=1e9 SURR_OBS_CLIP=1e9 SURR_OBS_DIV=4.0 \
  SURR_POOL=/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/train_pool_v5e.npz SURR_EP_RNG=1 \
  ATK_DELTA_LO=0.10 ATK_START_LO=30 ATK_START_HI=120 \
  SW_P=$1 SW_R=$2 \
  STOP_SEED=$3 STOP_LEARNERS=SWIRL,Adam1e-3 \
  STOP_OUT=$OUT/$4_s$3.json \
    python3 etc/scripts/online_stop.py > $LOG/$4_s$3.log 2>&1 &
}
for SD in 42 43 44; do
  run 0.001 2 $SD p0.001_r2      # ⓐ P₀ 저단
  run 0.01  1 $SD p0.01_r1       # ⓑ R 스캔 (R1 = 기본값)
  run 0.01  2 $SD p0.01_r2       #    R2 = 본선이 쓴 값 (재확인)
  run 0.01  3 $SD p0.01_r3
done
wait
echo "RP_STAGE3_DONE $(date +%H:%M)"
