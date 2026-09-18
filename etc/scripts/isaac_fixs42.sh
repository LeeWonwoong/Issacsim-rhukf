#!/usr/bin/env bash
# isaac_fixs42 (2026-09-05) — results_o3_adam_huber_3e4 는 CSV 가 185에피에서 끊겨 있다
#   (스크립트가 죽었을 때 RUN_DONE 만 수동 생성한 런). 4시드 대응표본을 완성하려면 s42 재실행 필요.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_fixs42.log
echo "[$(date '+%m-%d %H:%M')] huberab 완료 대기..." | tee -a $LOG
while [ ! -f HUBERAB_DONE ]; do sleep 180; done
sleep 30
for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
sleep 5
OUT=results_o3_adamhub_s42fix; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] adamhub_s42fix START"|tee -a $LOG
env NET_HIDDEN=16 ADAM_LOSS=huber ADAM_AMSGRAD=0 ADAM_LR=3e-4 \
    timeout 10800 python3 -u online_rl_main.py --headless --agent adam \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
n=$(wc -l < $(ls $OUT/metrics_*.csv|head -1) 2>/dev/null||echo 0)
echo "[$(date '+%m-%d %H:%M')] adamhub_s42fix END rc=$rc ep=$ep csv=$n"|tee -a $LOG
touch $OUT/RUN_DONE FIXS42_DONE
