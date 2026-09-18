#!/usr/bin/env bash
# isaac_spasab (2026-09-05) — Isaac 에서 surrogate 상위4 × SPAS {ON, OFF} 대조.
#   SPAS = h=0 DDQN argmax 출처. OFF=θ_target argmax(기존 abs) / ON=시그마 앙상블(err 과 동일).
#   공통: absolute · N5 · Q1e-3 · tau0.005/ui1 · alpha0.10 · huber_c3 · [16,16] · seed42
#   상위4: #1 P0.03·R1.5  #2 P0.05·R1.5  #3 P0.03·R1  #4 P0.02·R1.5
#   SPAS OFF 3개는 기존 런 재사용(P사다리 P0.05/P0.02, SW2 s42) → 신규는 OFF 1 + ON 4 = 5런
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_spasab.log
kill_isaac(){ for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local P=$1 R=$2 SP=$3; local TAG=sp${SP}_P${P}_R${R}; local OUT=results_o3_$TAG
  [ -f $OUT/RUN_DONE ] && { echo "SKIP $TAG"|tee -a $LOG; return 0; }
  echo "[$(date '+%m-%d %H:%M')] $TAG START (SPAS=$SP)"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=$R \
      RHUKF_FORM=absolute RHUKF_PINIT=$P RHUKF_ALPHA=0.10 RHUKF_SPAS=$SP \
      timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
      --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local n=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)
  local er=$(grep -c 'sequence size exceeds' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $TAG END csv=$n ddserr=$er"|tee -a $LOG
  [ "$n" -ge 201 ] && touch $OUT/RUN_DONE || echo "  ⚠ CSV 불완전"|tee -a $LOG
  return 0; }
run 0.03 1.5 0     # 상위1 OFF (유일하게 미보유)
run 0.03 1.5 1     # 상위1 ON
run 0.05 1.5 1     # 상위2 ON  (OFF 는 results_o3_pl_P0.05 재사용)
run 0.03 1   1     # 상위3 ON  (OFF 는 results_o3_sw2_s42 재사용)
run 0.02 1.5 1     # 상위4 ON  (OFF 는 results_o3_pl_P0.02 재사용)
# ★추가: surrogate P=0.1 계열 최고 (P0.1·R1·Q1e-3, F1 0.929 — P0.1 중 1위).
#   Isaac 에 P0.1 은 R1.5 만 있고 R1 은 미보유 → on/off 둘 다 신규.
#   큰 P₀(h=0.72) 에서도 init 효과가 같은 방향인지 보는 대조점.
run 0.1  1   0     # P0.1 최고 OFF
run 0.1  1   1     # P0.1 최고 ON
kill_isaac; touch SPASAB_DONE
echo "[$(date '+%m-%d %H:%M')] SPAS ON/OFF 대조 완료"|tee -a $LOG
