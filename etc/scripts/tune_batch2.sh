#!/usr/bin/env bash
# ============================================================
# tune_batch2.sh — config E 확정환경 첫 학습 배치 (2026-08-27)
#   환경: config E(ukf 기본값) + 신기동(R1.6/ω0.75·agg3.35·SPEED_MOD 편향수정)
#        + 탐지강조 reward(r_tp1.0/r_fp-0.7) + burst + 회전바람
#   4런: RHUKF 고-K레짐 본명(t02u4) 먼저 → Adam 3e-4 → Adam 1e-3 → RHUKF 저-K 대조(t005u1)
#   RHUKF r=1.0 고정 (T_Var 13~40 고-K → r↑=K↓ 는 원리 역행. R1.5 변형 기각)
#   개선: TAKEOFF 3연속 → HARD 에스컬레이션(코드) · timeout 12600(=5h, 구 12000은 283ep 컷)
# ============================================================
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=1.0 COM_BIAS_STD=0.05
# UKF = config E (코드 기본값. env 미지정 = E 그대로)
LOG=tune_batch2.log
echo "════ tune_batch2 시작 $(date '+%m-%d %H:%M:%S') (config E + 신기동) ════" | tee $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){  # $1=tag $2=agent $3=extra_env(k=v k=v)
  local OUT=results_e1_$1
  echo "[$(date '+%H:%M:%S')] $1 agent=$2 env='$3' START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 20 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)" | tee -a $LOG
}
# ── 본 4런 (10h 예산 목표: 200ep×4 ≈ 9.5h) ──
# ★2026-08-27 재배열: MATLAB 이 GPU 점유 중 RHUKF learn 9ms→138ms (업데이트/스텝 0.29 = 불공정).
#   Adam 은 learn 1-3ms 라 무영향 → Adam 먼저. RHUKF 는 MATLAB 종료 후 도달하도록 뒤로.
#   가드: RHUKF 런 시작 전 GPU util 이 높으면 로그에 경고 남김.
# RHUKF 공정성 게이트: GPU util<30% 이 5분 지속될 때까지 대기(최대 6h). 초과 시 경고 후 강행.
#   근거(2026-08-27 실측): GPU 점유 시 learn 9→138ms·push 0.3→13.6ms → 업데이트/스텝 0.29 (불공정)
gpu_guard(){
  local waited=0 ok=0
  while [ $waited -lt 21600 ]; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null||echo 0)
    if [ "${u:-100}" -lt 30 ]; then ok=$((ok+60)); else ok=0; fi
    if [ $ok -ge 300 ]; then echo "[$(date '+%H:%M:%S')] GPU 한가(util=${u}%, 5분 지속) → RHUKF 진행" | tee -a $LOG; return; fi
    [ $((waited % 1800)) -eq 0 ] && echo "[$(date '+%H:%M:%S')] GPU 대기 중 util=${u}% (${waited}s 경과)" | tee -a $LOG
    sleep 60; waited=$((waited+60))
  done
  echo "[$(date '+%H:%M:%S')] ⚠ GPU 대기 6h 초과 (util=${u}%) → 불공정 조건으로 강행. 결과 해석 주의" | tee -a $LOG
}
run adam_lr3e4   adam  "ADAM_LR=3e-4"
run adam_lr1e3   adam  "ADAM_LR=1e-3"
gpu_guard
run rhukf_t02u4  rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0"
run rhukf_t005u1 rhukf "RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.0"
touch TUNE_BATCH2_MAIN4_DONE
# ── 후속 변형 (예산 밖 — 이어서 자동 진행, 필요시 중단) ──
gpu_guard
run rhukf_pd03   rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.03"
run rhukf_R15    rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.5"
run rhukf_pd001  rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.01"   # R2.0 대체: pΔ 양방향(0.01/0.02/0.03) 완성
kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch TUNE_BATCH2_DONE
