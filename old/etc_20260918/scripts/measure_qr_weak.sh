#!/usr/bin/env bash
# 약공격 landing + 결과성 + Q_gyro/Q_vel sweep. 단일축·지속공격(steady NIS+consequence+decay).
# G2(baseline Qg2e-3)=full δ{0.05~0.8}×2pat×ws{0,8}. G1/G3=Qg1e-3/5e-3, V1/V2=Qv1e-3/2e-2 (aggressive δ{0.1,0.3,0.5}).
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
# SWEEP_BURST 미설정 = 지속공격 (steady NIS + 결과성 + 복귀감쇠 관측)
LOG=measure_qr.log
echo "════ QR/약공격 측정 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG

kill_isaac(){ pkill -9 -f "run_sim.py --headless" 2>/dev/null||true; for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}');do kill -9 $p 2>/dev/null;done; sleep 5; }

run(){ # $1=tag $2=Qg $3=Qv $4=pats $5=biases $6=dist
  local OUT=results_qr_$1
  echo "[$(date '+%H:%M:%S')] $1 Qg=$2 Qv=$3 START" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  UKF_Q_GYRO=$2 UKF_Q_VEL=$3 timeout 3000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
    --torque-yaw-ratio 0.0 --capture-mode hijack --capture-patterns $4 \
    --capture-disturbances $6 --capture-biases $5 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] $1 DONE rc=$? rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null||echo 0)" | tee -a $LOG
}

# δ→bias(single ×4.36): 0.05→0.218 0.1→0.436 0.15→0.654 0.2→0.872 0.3→1.308 0.5→2.18 0.8→3.488
FULL="0.218,0.436,0.654,0.872,1.308,2.18,3.488"
SMALL="0.436,1.308,2.18"   # δ0.1,0.3,0.5
D2="none:0,wind_turbulence:8"

# baseline (full δ, 2패턴) — 약공격 landing + 결과성 + 기준 감쇠
run G2_qg2e-3 2e-3 5e-3 circle,aggressive "$FULL" "$D2"
# Q_gyro sweep (aggressive, δ0.1/0.3/0.5)
run G1_qg1e-3 1e-3 5e-3 aggressive "$SMALL" "$D2"
run G3_qg5e-3 5e-3 5e-3 aggressive "$SMALL" "$D2"
# Q_vel sweep (감쇠; aggressive)
run V1_qv1e-3 2e-3 1e-3 aggressive "$SMALL" "$D2"
run V2_qv2e-2 2e-3 2e-2 aggressive "$SMALL" "$D2"

kill_isaac
echo "════ 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch MEASURE_QR_DONE
