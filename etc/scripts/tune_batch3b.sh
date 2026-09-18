#!/usr/bin/env bash
# batch3b — 확정 격자 8런 (2026-08-27 사용자 확정): [16,16]·N5·Q1e-3 고정,
#   세트{A: tau0.02·ui4 / B: tau0.005·ui1} × R{1.0,1.5} × pΔ{0.01,0.05}
#   adam_n16(진행 중, 고아) 종료 대기 후 시작. 관측/파라미터 = 640/514 = 1.24 ✓
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=tune_batch3.log
echo "[$(date '+%H:%M:%S')] batch3b 격자 시작 (1-8 RHUKF → 9 Adam)" | tee -a $LOG
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){
  local OUT=results_g1_$1
  if [ -f $OUT/RUN_DONE ]; then echo "[$(date '+%H:%M:%S')] $1 SKIP (이미 완료)" | tee -a $LOG; return; fi
  local u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null||echo '?')
  echo "[$(date '+%H:%M:%S')] $1 agent=$2 env='$3' START (GPU util=${u}%)" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 1 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%H:%M:%S')] $1 END rc=$rc ep=$ep" | tee -a $LOG
  # 200 에피 도달(또는 정상 rc)면 완료 마커 — 워치독이 재시작해도 skip
  if [ "$ep" -ge 195 ] || [ "$rc" = "0" ]; then touch $OUT/RUN_DONE; fi
}
CA="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.02 RHUKF_UI=4"
CB="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1"
run A_r10_pd001 rhukf "$CA RHUKF_R=1.0 RHUKF_PD=0.01"
run A_r10_pd005 rhukf "$CA RHUKF_R=1.0 RHUKF_PD=0.05"
run A_r15_pd001 rhukf "$CA RHUKF_R=1.5 RHUKF_PD=0.01"
run A_r15_pd005 rhukf "$CA RHUKF_R=1.5 RHUKF_PD=0.05"
run B_r10_pd001 rhukf "$CB RHUKF_R=1.0 RHUKF_PD=0.01"
run B_r10_pd005 rhukf "$CB RHUKF_R=1.0 RHUKF_PD=0.05"
run B_r15_pd001 rhukf "$CB RHUKF_R=1.5 RHUKF_PD=0.01"
run B_r15_pd005 rhukf "$CB RHUKF_R=1.5 RHUKF_PD=0.05"
run adam_n16    adam  "ADAM_LR=1e-3 NET_HIDDEN=16"       # 9번 (사용자 지정 순서)
kill_isaac
echo "════ batch3b 격자 8런 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch TUNE_BATCH3_DONE
