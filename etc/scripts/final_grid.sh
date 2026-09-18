#!/usr/bin/env bash
# 최종 밤샘 grid: arm0.02, Q_gyro2e-3, 전패턴 × δ{0.3~0.8} × ws{0,4~12}, 시간창(clean→바람→겹침→공격→복귀).
# 관측 min(log(1+√NIS),3). 공격only(ws0)+공격+바람 구분 측정. 최종 확정용.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=2e-3 SWEEP_ATTACK_TYPE=tilt
export EP_MAX_STEPS=450 WIND_START=100 WIND_END=300 SWEEP_ATK_START=180 SWEEP_ATK_END=380
OUT=results_final; LOG=final.log
echo "════ 최종 grid 시작 $(date '+%m-%d %H:%M:%S') (arm0.02 Q2e-3, δ0.3-0.8 × ws0/4-12) ════" | tee $LOG
# 시작 정리
pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
sleep 5
rm -rf $OUT; mkdir -p $OUT
# ★ B: balanced 45° (roll=pitch). --torque-yaw-ratio 1.0 → (value,value) → roll=pitch.
#   magnitude δ=√2·value/A → value=δ·A/√2=δ×3.083. δ0.3-0.8 = 0.925/1.233/1.542/1.850/2.158/2.466.
timeout 43200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --torque-yaw-ratio 1.0 --capture-mode hijack \
  --capture-patterns circle,figure8,waypoint,aggressive \
  --capture-disturbances none:0,wind_turbulence:4,wind_turbulence:5,wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_turbulence:9,wind_turbulence:10,wind_turbulence:11,wind_turbulence:12 \
  --capture-biases 0.9249,1.2332,1.5415,1.8497,2.1580,2.4663 \
  --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
RC=$?
echo "════ 종료 $(date '+%H:%M:%S') rc=$RC rows=$(wc -l < $OUT/sweep_detail.csv 2>/dev/null || echo 0) ════" | tee -a $LOG
# 종료 정리
pkill -9 -f "run_sim.py --headless" 2>/dev/null || true
for pid in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
touch FINAL_DONE
