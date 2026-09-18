#!/usr/bin/env bash
# isaac_v3check (2026-09-06) — FROZEN-v3 가 실제로 어려워지는지 단일 검증런.
#   config: CALwin (Isaac 최고, v2 에서 F1 0.929 · d′ 2.88 · 추락 0 · 4시드)
#   판정: F1 하락 · d′ 하락 · 탐지지연 증가 · **추락 발생 여부**
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
LOG=isaac_v3check.log
for p in $(pgrep -f MicroXRCEAgent 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
sleep 5
OUT=results_v3_calwin_s42; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] v3check START" | tee -a $LOG
env | grep -E "^(ATK_|SPEED_|COM_|EP_MAX|WIND_|MPC_|SENSOR_)" | sort | tee -a $LOG
env NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-2 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=2 \
    RHUKF_FORM=absolute RHUKF_PINIT=0.01 RHUKF_ALPHA=0.10 RHUKF_SPAS=1 \
    timeout 10800 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 2.5 --seed 42 --outdir $OUT > $OUT/train.log 2>&1
n=$(wc -l < $(ls $OUT/metrics_*.csv 2>/dev/null|head -1) 2>/dev/null||echo 0)
echo "[$(date '+%m-%d %H:%M')] v3check END csv=$n ddserr=$(grep -c 'sequence size exceeds' $OUT/train.log)" | tee -a $LOG
[ "$n" -ge 201 ] && touch $OUT/RUN_DONE V3CHECK_DONE
for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done
