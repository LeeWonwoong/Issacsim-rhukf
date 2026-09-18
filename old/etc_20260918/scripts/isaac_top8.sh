#!/usr/bin/env bash
# isaac_top8 (2026-09-05) — surrogate env5.1 상위 8 SWIRL config 를 Isaac 에서 검증.
#   공통: absolute · N5 · tau0.005/ui1 · alpha0.10 · Q1e-3 · huber_c3 · [16,16] · seed42
#   ★2026-09-05 Isaac 은 앞으로 전부 RHUKF_SPAS=1 (h=0 argmax = 시그마 앙상블).
#     기존 Isaac abs 런들(CALwin·P사다리·SW2)은 SPAS OFF 로 측정된 것이라 직접 비교 불가 →
#     비교 기준선으로 CALwin 을 SPAS ON 으로 한 번 더 돌린다.
#   이미 Isaac 보유:  #2 P0.05·R1.5(P사다리) · #3 P0.03·R1(SW2 s42) · #4 P0.02·R1.5(P사다리)
#   신규 5개:  #1 P0.03·R1.5 · #5 P0.02·R1 · #6 P0.07·R1 · #7 P0.02·R2 · #8 P0.1·R1
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_top8.log
kill_isaac(){ for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local P=$1 R=$2; local TAG=t8s_P${P}_R${R}; local OUT=results_o3_$TAG
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $TAG"|tee -a $LOG; return 0; }
  echo "[$(date '+%m-%d %H:%M')] $TAG START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=$R \
      RHUKF_FORM=absolute RHUKF_PINIT=$P RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
      timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  local n=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)
  local er=$(grep -c 'sequence size exceeds' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $TAG END rc=$rc ep=$ep csv=$n ddserr=$er"|tee -a $LOG
  [ "$n" -ge 201 ] && touch $OUT/RUN_DONE || echo "  ⚠ CSV 불완전 — RUN_DONE 미생성"|tee -a $LOG
  return 0; }
# 기준선: CALwin 을 SPAS ON 으로 (기존 4시드는 SPAS OFF 라 직접 비교 불가)
OUT=results_o3_calwin_spas_s42
if [ ! -f $OUT/RUN_DONE ]; then
  echo "[$(date '+%m-%d %H:%M')] calwin_spas START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-2 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=2 \
      RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
      timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  n=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] calwin_spas END csv=$n"|tee -a $LOG
  [ "$n" -ge 201 ] && touch $OUT/RUN_DONE
fi
run 0.03 1.5     # #1
run 0.02 1       # #5
run 0.07 1       # #6
run 0.02 2       # #7
run 0.1  1       # #8
kill_isaac; touch TOP8_DONE
echo "[$(date '+%m-%d %H:%M')] Isaac 상위8 검증 완료"|tee -a $LOG
