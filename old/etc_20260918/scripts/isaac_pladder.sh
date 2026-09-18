#!/usr/bin/env bash
# isaac_pladder (2026-09-04) — Isaac 에서 P(p_init) 사다리. 최적점이 아래쪽에 있는지 확인.
#   기존 Isaac 실측:  P0.01 abs = 0.925(최고) > P0.1 abs = 0.919 > P0.2 err = 0.907(최저)
#   → 최적이 0.01 이하일 수 있음. 아래로 브래킷: {0.002, 0.005, 0.02, 0.05}
#   나머지 전부 최고 config 와 동일: absolute · R1.5 · Q1e-3 · tau0.005/ui1 · N5 · alpha0.1 · [16,16] · seed42
#   ★ rs2 런 완료 후 자동 시작. Isaac 은 동시 1개만 가능하므로 순차.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_pladder.log
echo "[$(date '+%m-%d %H:%M')] rs2 완료 대기..." | tee -a $LOG
while [ ! -f results_o3_abs001_rs2/RUN_DONE ]; do sleep 120; done
sleep 30
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
for P in 0.005 0.02 0.002 0.05; do
  OUT=results_o3_pl_P${P}
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $P" | tee -a $LOG; continue; }
  echo "[$(date '+%m-%d %H:%M')] P=$P START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 \
      RHUKF_FORM=absolute RHUKF_PINIT=$P RHUKF_ALPHA=0.10 \
      timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] P=$P END rc=$rc ep=$ep" | tee -a $LOG
  touch $OUT/RUN_DONE
done
kill_isaac; touch PLADDER_DONE
echo "[$(date '+%m-%d %H:%M')] P 사다리 완료" | tee -a $LOG
