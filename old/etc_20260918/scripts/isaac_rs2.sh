#!/usr/bin/env bash
# isaac_rs2 (2026-09-04) — Isaac 최고 SWIRL config 에 보상 스케일 ×2.
#   config: P0.01 absolute · R1.5 · Q1e-3 · tau0.005/ui1 · N5 · alpha0.1 · [16,16]  (= results_o3_swirl_abs001)
#   변경:   --reward-scale 2.0  (곱하기 2). huber_c 는 3 그대로 → 잔차 대비 knee 가 상대적으로 절반
#           = 유계영향이 더 자주 발동하는 영역. T_Var 는 약 4배.
#   대조군: results_o3_swirl_abs001 (F1 .925 prec .900 rec .952 fpr .148 evalFA .444 rr40-100 .270)
#   ※ 탐지지표(F1/prec/rec/fpr/delay)는 reward 무관이라 직접 비교 가능. reward 컬럼만 2배 팽창.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0   # FROZEN-ENV v2
OUT=results_o3_abs001_rs2
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
sleep 5
rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] abs001_rs2 START (reward_scale=2.0)" | tee -a isaac_rs2.log
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 \
    timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --reward-scale 2.0 --outdir $OUT > $OUT/train.log 2>&1
rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
echo "[$(date '+%m-%d %H:%M')] abs001_rs2 END rc=$rc ep=$ep" | tee -a isaac_rs2.log
touch $OUT/RUN_DONE
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
