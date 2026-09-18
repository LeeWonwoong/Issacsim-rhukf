#!/usr/bin/env bash
# isaac_winner (2026-09-03) — surrogate 그리드 승자 config 를 Isaac 에서 확인.
#   승자: alpha0.10 · P0(p_delta)=0.2 · R1 · Q1e-2 · N5 · tau0.005/ui1 · error-state · [16,16]
#   대조: results_o3_swirl_base (동일 환경/시드, 단 R1.5 · Q1e-3 · PD0.01)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0   # FROZEN-ENV v2
OUT=results_o3_abs001_nohuber
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
sleep 5
rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] abs001_nohuber START" | tee -a isaac_nohuber.log
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 RHUKF_HUBER_C=1e9 \
    timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
echo "[$(date '+%m-%d %H:%M')] abs001_nohuber END rc=$rc ep=$ep" | tee -a isaac_nohuber.log
touch $OUT/RUN_DONE
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
