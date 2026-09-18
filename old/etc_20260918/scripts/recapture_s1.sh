#!/usr/bin/env bash
# ============================================================
# recapture_s1 — speed 1 정본 재캡처 (2026-08-27, 버그수정 B1~B5 반영 후)
#   전부 speed 1: 틱 50Hz=sim, GPS 5틱, UKF dt 정합 — 시간축 명세 일치 상태의 정본.
#   [1] track_check : 5패턴 평시 추종/기준선          (~10분)
#   [2] traj_wp     : waypoint δ0.6/0.8 × track/whover2, 공격창 15-25s, 무풍  (~8분)
#   [3] scurve_val  : 신형 aggressive(3D S-curve) 검증비행, 무풍+ws8         (~5분)
#   [4] zu_s1       : POMDP 기준선 (평시+burst공격 혼합 16에피, zu 로그)      (~35분)
# ============================================================
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=recapture_s1.log
echo "════ recapture_s1 시작 $(date '+%m-%d %H:%M:%S') ════" | tee $LOG
kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 COM_BIAS_STD=0.05 ACCEL_FF=1
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5

# [1] track_check
export EP_MAX_STEPS=400 CAPTURE_POLICIES=track
echo "[$(date '+%H:%M:%S')] [1] track_check START" | tee -a $LOG
kill_isaac; rm -rf results_s1_track; mkdir -p results_s1_track
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode normal --capture-patterns circle,figure8,waypoint,aggressive,scurve \
  --capture-disturbances none:0 --episodes 1 --speed 1 --log-zu \
  --outdir results_s1_track > results_s1_track/run.log 2>&1
echo "[$(date '+%H:%M:%S')] [1] DONE rc=$?" | tee -a $LOG

# [2] traj_wp (공격창)
export SWEEP_ATK_START=150 SWEEP_ATK_END=250 SWEEP_ATTACK_TYPE=tilt CAPTURE_POLICIES=track,whover2
echo "[$(date '+%H:%M:%S')] [2] traj_wp START" | tee -a $LOG
kill_isaac; rm -rf results_s1_trajwp; mkdir -p results_s1_trajwp
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode hijack --capture-patterns waypoint --capture-disturbances none:0 \
  --capture-biases 2.616,3.488 --episodes 1 --speed 1 --log-zu \
  --outdir results_s1_trajwp > results_s1_trajwp/run.log 2>&1
echo "[$(date '+%H:%M:%S')] [2] DONE rc=$?" | tee -a $LOG
unset SWEEP_ATK_START SWEEP_ATK_END

# [3] scurve_val
export CAPTURE_POLICIES=track
echo "[$(date '+%H:%M:%S')] [3] scurve_val START" | tee -a $LOG
kill_isaac; rm -rf results_s1_scurve; mkdir -p results_s1_scurve
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque \
  --capture-mode normal --capture-patterns scurve \
  --capture-disturbances none:0,wind_turbulence:8 --episodes 1 --speed 1 --log-zu \
  --outdir results_s1_scurve > results_s1_scurve/run.log 2>&1
echo "[$(date '+%H:%M:%S')] [3] DONE rc=$?" | tee -a $LOG
unset CAPTURE_POLICIES

# [4] zu 기준선 (학습 모드 캡처 — zu_v3 과 동일 방식, seed42)
echo "[$(date '+%H:%M:%S')] [4] zu_s1 START" | tee -a $LOG
kill_isaac; rm -rf results_zu_s1; mkdir -p results_zu_s1
timeout 5400 python3 -u online_rl_main.py --headless --agent adam --max-ep 16 \
  --speed 1 --log-zu --seed 42 --outdir results_zu_s1 > results_zu_s1/cap.log 2>&1
echo "[$(date '+%H:%M:%S')] [4] DONE rc=$?" | tee -a $LOG
kill_isaac
echo "════ recapture_s1 완료 $(date '+%H:%M:%S') ════" | tee -a $LOG
touch RECAPTURE_S1_DONE
