#!/usr/bin/env bash
# night_ablation (2026-08-31): ①gyro-only ablation(Adam 8D vs 12D) ②버스트길이 사다리(ON{40,15,6}×{SWIRL,Adam})
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=night_ablation.log
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ local OUT=results_ab_$1
  if [ -f $OUT/RUN_DONE ]; then echo "[$(date '+%H:%M')] $1 SKIP"|tee -a $LOG; return; fi
  echo "[$(date '+%m-%d %H:%M')] $1 agent=$2 env='$3' START"|tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $3 timeout 12600 python3 -u online_rl_main.py --headless --agent $2 \
    --max-ep 200 --speed 1 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $1 END rc=$rc ep=$ep"|tee -a $LOG
  if [ "$ep" -ge 150 ] || [ "$rc" = "0" ]; then touch $OUT/RUN_DONE; fi
}
CB="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.01"
CA="ADAM_LR=1e-3 NET_HIDDEN=16"
# 공통 공격: δ0.4 고정, OFF 25-40
ATK="ATK_ON_LO=REPLACE_ON_LO ATK_ON_HI=REPLACE_ON_HI ATK_OFF_LO=25 ATK_OFF_HI=40 ATK_DELTA_LO=0.4 ATK_DELTA_HI=0.4"
# ── ① gyro-only ablation (Adam, ON10-20 = 기존 강건셋과 동일조건) ──
AB="ATK_ON_LO=10 ATK_ON_HI=20 ATK_OFF_LO=25 ATK_OFF_HI=40"
run gyroonly_adam adam "$CA $AB GYRO_ONLY=1"
run full12_adam   adam "$CA $AB"
# ── ② 버스트 길이 사다리 (δ0.4, OFF25-40) × {SWIRL, Adam} ──
for ON in 40:60 15:25 6:12; do
  L=${ON%:*}; H=${ON#*:}; TAG=on${L}
  A2="ATK_ON_LO=$L ATK_ON_HI=$H ATK_OFF_LO=25 ATK_OFF_HI=40 ATK_DELTA_LO=0.4 ATK_DELTA_HI=0.4"
  run ${TAG}_rhukf rhukf "$CB $A2"
  run ${TAG}_adam  adam  "$CA $A2"
done
kill_isaac
echo "════ night_ablation 완료 $(date '+%m-%d %H:%M') ════"|tee -a $LOG
touch NIGHT_ABLATION_DONE
