#!/usr/bin/env bash
# 최고 단일-seed RHUKF 찾기: 탐지강조 reward(r_tp1.0·r_fp-0.7) + POMDP환경. RHUKF 하이퍼 비교.
#   1. Adam (새 reward baseline)
#   2. RHUKF tau0.005 ui1  (느린타깃·짧은호라이즌)
#   3. RHUKF tau0.02  ui4  (빠른타깃·긴호라이즌)
#   4. RHUKF tau0.02  ui4 R1.5 (3번 유망시 R변형)
# 공통 env: 가감속0.7 + Q_gyro5e-3 + COM0.05 + 바람 + burst.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=5e-3 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.7 SPEED_MOD_FREQ=0.25 COM_BIAS_STD=0.05
LOG=rhukf_tune.log
echo "════ RHUKF 튜닝 배치 시작 $(date '+%m-%d %H:%M:%S') (탐지강조 reward) ════" | tee $LOG
kill_isaac(){ pkill -9 -f "run_sim.py --headless" 2>/dev/null||true; for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done; sleep 5; }
run(){  # $1=tag $2=agent $3=tau $4=ui $5=R
  local OUT=results_tune_$1
  echo "[$(date '+%H:%M:%S')] $1 agent=$2 tau=$3 ui=$4 R=$5 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  RHUKF_TAU=$3 RHUKF_UI=$4 RHUKF_R=$5 timeout 12000 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 300 --speed 20 --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -cE 'Ep [0-9]+: ' $OUT/train.log 2>/dev/null||echo 0)" | tee -a $LOG
}
run adam         adam  0.005 5 1.0
run rhukf_t005u1 rhukf 0.005 1 1.0
run rhukf_t02u4  rhukf 0.02  4 1.0
run rhukf_t02u4R15 rhukf 0.02 4 1.5
kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch RHUKF_TUNE_DONE
