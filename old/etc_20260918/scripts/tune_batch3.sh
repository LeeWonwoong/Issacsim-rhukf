#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# tune_batch3 — RHUKF "의도된 옵티마이저" 격자 1파 (2026-08-27 사용자 확정)
#
#  용어 확정: ui = update_interval / N = N_horizon.  세트 커플링(항상 함께):
#    세트A = (tau 0.02,  ui 4)   세트B = (tau 0.005, ui 1)
#  ⚠ 이전 RHUKF 런 전부(t02u4, n16u6 등)는 RHUKF_UI 가 N 을 바꾸고 ui=1 고정
#    → 비의도 옵티마이저. 기록만 보존, 주장 불가.
#
#  전체 격자: 세트{A,B} × N{5,6} × pΔ{0.01,0.02,0.05} × Q{1e-4,1e-3,1e-2}
#             × R{1,1.5,2,2.5,3} × α{0.1,0.5,0.9}  = 540런 (불가)
#  1파 선정(5런): 세트×N 4점 전수 + Adam 짝. 나머지 축은 재직값(pΔ.02 q1e-4 r1 α.1) 고정.
#  2파(결과 보고): 1파 승자 세트에 pΔ{0.01,0.05}·R{1.5,2}·Q{1e-3}·α{0.5} OFAT.
# ════════════════════════════════════════════════════════════════════
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=1.0 COM_BIAS_STD=0.05
LOG=tune_batch3.log
echo "════ batch3 (의도 옵티마이저 격자 1파) 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){
  local OUT=results_g1_$1
  local u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null||echo '?')
  echo "[$(date '+%H:%M:%S')] $1 agent=$2 env='$3' START (GPU util=${u}%)" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 20 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)" | tee -a $LOG
}
run adam_n16 adam "ADAM_LR=1e-3 NET_HIDDEN=16"
run A_N6 rhukf "NET_HIDDEN=16 RHUKF_TAU=0.02  RHUKF_UI=4 RHUKF_N=6 RHUKF_R=1.0"
run A_N5 rhukf "NET_HIDDEN=16 RHUKF_TAU=0.02  RHUKF_UI=4 RHUKF_N=5 RHUKF_R=1.0"
run B_N6 rhukf "NET_HIDDEN=16 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_N=6 RHUKF_R=1.0"
run B_N5 rhukf "NET_HIDDEN=16 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_N=5 RHUKF_R=1.0"
kill_isaac
echo "════ batch3 1파 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch TUNE_BATCH3_DONE
