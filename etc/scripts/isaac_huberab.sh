#!/usr/bin/env bash
# isaac_huberab (2026-09-05) — Isaac 에서 Adam 의 Huber ON/OFF 대응비교.
#   같은 lr(3e-4)·같은 시드로 ADAM_LOSS=huber vs mse.  Huber ON 은 s42,43,44,45 확보 예정.
#   → OFF(mse 3e-4) 를 s43,44,45 추가하면 4시드 대응표본이 완성된다. (s42 는 보유)
#   ★ isaac_final4 완료 후 자동 시작.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_huberab.log
echo "[$(date '+%m-%d %H:%M')] final4 완료 대기..." | tee -a $LOG
while [ ! -f FINAL4_DONE ]; do sleep 180; done
sleep 30
kill_isaac(){ for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
for SD in 43 44 45; do
  OUT=results_o3_adammse_s$SD
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $SD"|tee -a $LOG; continue; }
  echo "[$(date '+%m-%d %H:%M')] adammse_s$SD START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env NET_HIDDEN=16 ADAM_LOSS=mse ADAM_AMSGRAD=0 ADAM_LR=3e-4 \
      timeout 10800 python3 -u online_rl_main.py --headless --agent adam \
      --max-ep 200 --speed 2.5 --seed $SD --outdir $OUT > $OUT/train.log 2>&1
  rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  er=$(grep -c 'sequence size exceeds' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] adammse_s$SD END rc=$rc ep=$ep ddserr=$er"|tee -a $LOG
  touch $OUT/RUN_DONE
done
kill_isaac; touch HUBERAB_DONE
echo "[$(date '+%m-%d %H:%M')] Huber ON/OFF 4시드 완료"|tee -a $LOG
