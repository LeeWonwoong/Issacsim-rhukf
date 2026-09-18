#!/usr/bin/env bash
# Adam 옵티마이저 전수(2⁴=16 config): lr{3e-4,1e-3} × amsgrad{0,1} × obs스케일{0,1} × net{24,16}. 각 300ep.
# 확정 학습env: Q_gyro2e-3·회전바람arm0.02·에피400·δ0.1-0.8·burst5-20/10-30·추락bootstrap. speed=20(최대).
# 지표: eval_history.npz (det_delay·FP·survival per eval).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 EP_MAX_STEPS=400
SPEED=20
LOG=adam_opt.log
echo "════ Adam 전수 sweep(16) 시작 $(date '+%m-%d %H:%M:%S') speed=$SPEED ════" | tee $LOG

kill_isaac(){ pkill -9 -f "run_sim.py --headless" 2>/dev/null||true; for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done; sleep 5; }

run(){  # $1=tag $2=lr $3=ams $4=obs $5=net(24|16)
  local OUT=results_opt_$1
  echo "[$(date '+%H:%M:%S')] $1 lr=$2 ams=$3 obs=$4 net=$5 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  local NETENV=""; [ "$5" = "16" ] && NETENV="16"
  ADAM_LR=$2 ADAM_AMSGRAD=$3 OBS_NORM=$4 NET_HIDDEN=$NETENV \
    timeout 21600 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 300 --speed $SPEED --outdir $OUT > $OUT/train.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? ep=$(grep -cE 'Ep [0-9]+: ' $OUT/train.log 2>/dev/null||echo '?')" | tee -a $LOG
}

for LR in 3e-4 1e-3; do
  LRT=$(echo $LR | tr -d '-')
  for AMS in 1 0; do
    for OBS in 0 1; do
      for NET in 24 16; do
        run "lr${LRT}_ams${AMS}_obs${OBS}_net${NET}" $LR $AMS $OBS $NET
      done
    done
  done
done

kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch ADAM_OPT_DONE
