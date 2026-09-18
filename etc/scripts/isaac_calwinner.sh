#!/usr/bin/env bash
# isaac_calwinner (2026-09-04) — 정합(CAL) grid64 승자를 Isaac 에서 검증.
#   승자: t005u1_R2_Q2_P0.01_abs  (pFP≤0.03 제약 하 약공격 recall 최대 + 64개 중 F1 최고 0.928)
#         = tau0.005/ui1 · R2 · Q1e-2 · P(p_init)0.01 · absolute · N5 · alpha0.1 · [16,16]
#   대조: results_o3_swirl_abs001 (tau/ui/P/mode 동일, R1.5·Q1e-3) → Isaac F1 .925 fpr .148 evalFA .444
#   ★ P 사다리 완료 후 자동 시작.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export WIND_MOMENT_ARM=0.07 SPEED_SCALE=1.25 MPC_ACC_HOR=4.0 SENSOR_NOISE_SCALE=1.0
LOG=isaac_calwinner.log
echo "[$(date '+%m-%d %H:%M')] P 사다리 완료 대기..." | tee -a $LOG
while [ ! -f PLADDER_DONE ]; do sleep 120; done
sleep 30
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
sleep 5
OUT=results_o3_calwinner; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] calwinner START (R2 Q1e-2 P0.01 abs)" | tee -a $LOG
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-2 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=2 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 \
    timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
rc=$?; ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
echo "[$(date '+%m-%d %H:%M')] calwinner END rc=$rc ep=$ep" | tee -a $LOG
touch $OUT/RUN_DONE
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
