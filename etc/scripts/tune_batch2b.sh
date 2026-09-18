#!/usr/bin/env bash
# tune_batch2b — RHUKF 5런 (사용자 지시로 GPU 게이트 해제·강행, 2026-08-27 06:4x).
#   ⚠ MATLAB GPU 점유 중 시작 → learn 138ms·업데이트/스텝 0.29 = Adam 과 비교 무효 가능.
#   각 런 시작 시 GPU util 을 기록해 사후 공정성 판별.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=1.0 COM_BIAS_STD=0.05
LOG=tune_batch2.log
echo "════ batch2b (RHUKF 강행) 시작 $(date '+%m-%d %H:%M:%S') ════" | tee -a $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){
  local OUT=results_e1_$1
  local u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null||echo '?')
  echo "[$(date '+%H:%M:%S')] $1 agent=$2 env='$3' START (GPU util=${u}%)" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 20 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)" | tee -a $LOG
}
run rhukf_t02u4  rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0"
run rhukf_t005u1 rhukf "RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.0"
touch TUNE_BATCH2_MAIN4_DONE
run rhukf_pd03   rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.03"
run rhukf_R15    rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.5"
run rhukf_pd001  rhukf "RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.01"
kill_isaac
echo "════ batch2b 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch TUNE_BATCH2_DONE
