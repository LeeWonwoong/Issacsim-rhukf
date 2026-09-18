#!/usr/bin/env bash
# aggfix_verify (2026-09-15 저녁): aggressive 위상 수정 검증 캡처 — aggressive × 무공격(δ0) × ws{0,10} × track × 4ep (~15–20 min)
#   구 cert_WIND(수정 전) 같은 셀과 설정점 점프·추종오차·평시 NIS 비교. 무대 플래그는 cert_MID/cert_WIND 와 동일.
#   GPU 규칙: v33 종료(V33_ENDED) ∧ 사슬 워커 없음 ∧ 사전 분석 대기 중(ANALYSIS_V33_DONE 없음)일 때만 착수. 실행 중 chain_hz/ISAAC_BUSY 로 사슬 착수를 막는다.
cd /home/acsl/projects/Issacsim-rhukf
C=results/claudecodefortest/night/chain_hz; OUT=results/claudecodefortest/aggfix_verify
LOG=results/claudecodefortest/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
cnt(){ ps aux | grep -cE "$1"; }
log "aggfix_verify 대기 (V33_ENDED·사슬 워커 없음 조건)"
while :; do
  if [ -f $C/V33_ENDED ] && [ "$(cnt '[e]nv_scan_v33.py')" -eq 0 ] && [ "$(cnt '[c]hain_hz.py work')" -eq 0 ]; then
    if [ -f $C/ANALYSIS_V33_DONE ]; then log "aggfix_verify: 사전 분석이 이미 끝나 사슬이 GPU 를 쓸 차례 — 이번 창에서는 건너뜀"; exit 0; fi
    break
  fi
  sleep 60
done
touch $C/ISAAC_BUSY; trap 'rm -f $C/ISAAC_BUSY' EXIT
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.10 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f results/claudecodefortest/FS_PARAMS.env ] && source results/claudecodefortest/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
nvidia-smi >/dev/null 2>&1 || { log "GPU 불가 — 중단"; exit 1; }
rm -rf $OUT; mkdir -p $OUT; kill_all; log "aggfix_verify START (aggressive × δ0 × ws0/10 × track × 4ep, 수정 코드)"
env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track" \
timeout 2700 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
  --capture-patterns aggressive --capture-disturbances none:0,wind_turbulence:10 \
  --capture-biases 0 --episodes 4 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
rc=$?; kill_all
log "aggfix_verify END rc=$rc rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0) err=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log)"
touch $OUT/DONE
