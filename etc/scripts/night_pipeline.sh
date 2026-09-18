#!/usr/bin/env bash
# night_pipeline — 1번→2번→3번 순차. 각 단계: 튜닝 완료 대기 → 집계·기록 → 본선 → 기록 → 다음.
cd /home/acsl/projects/Issacsim-rhukf
N=results/claudecodefortest/night
Q=$N/tonight/q
REP=$N/NIGHT_PIPELINE_0916.md
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $REP; }

say "## 밤샘 파이프라인 시작"

# ─────────── 1번: 온라인 RL 튜닝 ───────────
say "### 1번 온라인 RL — 튜닝 대기"
while :; do
  n=$(ls $Q 2>/dev/null | grep -c '^tune1_'); [ "$n" -ge 10 ] && break; sleep 60
done
say "1번 튜닝 10/10 완료 → 집계"
python3 etc/scripts/agg_tune1.py >> $REP 2>&1
say "1번 기록 완료 → 본선은 별도 큐(QSET=main1)로 병행 실행 중"

# ─────────── 2번: 오프라인 정지 튜닝 → 본선 ───────────
say "### 2번 오프라인 정지 — 튜닝 대기"
while :; do grep -q "오프라인 완료" /tmp/g2of.log 2>/dev/null && break; sleep 60; done
say "2번 튜닝 완료 → 집계"
# ★09-17 N 덮어쓰기 버그 수정본(g3ofN.log)을 합쳐서 최적 선택
  while :; do grep -q "N 재스캔 완료" /tmp/g3ofN.log 2>/dev/null && break; sleep 30; done
  cat /tmp/g2of.log /tmp/g3ofN.log > /tmp/g2of_merged.log
  BEST2=$(python3 etc/scripts/pick_best.py /tmp/g2of_merged.log 0.30)
say "2번 최적: $BEST2"
grep -E "^  (R|N|A|base)" /tmp/g2of_merged.log >> $REP
say "2번 본선 착수 (6학습기 × 5시드, 500ep, 병렬)"
for SD in 42 43 44 45 46; do
  env $BEST2 STOP_EPISODES=500 STOP_PSTOP=0.10 STOP_L=6 STOP_OBS_DIV=4.0 \
      STOP_DATA=$N/stop_dataset_50k_L6.npz STOP_SEED=$SD \
      STOP_LEARNERS=SWIRL,Adam3e-4,Adam3e-4ams,Adam1e-3,Adam1e-3ams,EKF-TD,UKF-TD \
      STOP_OUT=$N/main2_s${SD}.json python3 etc/scripts/offline_stop.py > /tmp/main2_$SD.log 2>&1 &
done
wait
say "2번 본선 완료 → 집계"
python3 etc/scripts/agg_stop.py "$N/main2_s*.json" "2번 오프라인 정지" >> $REP 2>&1

# ─────────── 3번: 온라인 정지 튜닝 → 본선 ───────────
say "### 3번 온라인 정지 — 튜닝 대기"
while :; do grep -q "온라인 완료" /tmp/g2on.log 2>/dev/null && break; sleep 60; done
say "3번 튜닝 완료 → 집계"
BEST3=$(python3 etc/scripts/pick_best.py /tmp/g2on.log 0.01)
say "3번 최적: $BEST3"
grep -E "^  (R|N|A)" /tmp/g2on.log >> $REP
say "3번 본선 착수 (7학습기 × 5시드 × 팔 A/B, 500ep)"
for ARM in A B; do
  [ "$ARM" = B ] && DIV=4.0 || DIV=1.0
  for SD in 42 43 44 45 46; do
    env $BEST3 STOP_EPISODES=500 STOP_EP_STEPS=300 STOP_PSTOP=0.10 STOP_EPS_FRAC=0.30 \
        STOP_L=6 STOP_GAMMA=0.9 SURR_CLIP=1e9 SURR_OBS_CLIP=1e9 SURR_OBS_DIV=$DIV \
        ATK_DELTA_LO=0.10 ATK_START_LO=30 ATK_START_HI=120 STOP_SEED=$SD \
        STOP_LEARNERS=SWIRL,Adam3e-4,Adam3e-4ams,Adam1e-3,Adam1e-3ams,EKF-TD,UKF-TD \
        STOP_OUT=$N/main3_${ARM}_s${SD}.json python3 etc/scripts/online_stop.py > /tmp/main3_${ARM}_$SD.log 2>&1 &
  done
  wait
  say "3번 팔 $ARM 완료"
done
say "3번 본선 완료 → 집계"
python3 etc/scripts/agg_stop.py "$N/main3_*_s*.json" "3번 온라인 정지" >> $REP 2>&1
say "## 파이프라인 전체 완료"
