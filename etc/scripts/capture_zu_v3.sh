#!/usr/bin/env bash
# ============================================================
# capture_zu_v3.sh — SPEED_MOD 편향수정(①) + FREQ 1.0(②) + 궤적 R1.6/ω0.75(③) 적용 후 캡처.
#   zu_v2(before)와 동일 시드/에피수 → 시나리오(패턴·바람·공격δ) 순서가 대응된다.
#   nohup ./etc/scripts/capture_zu_v3.sh > scratchpad/cap_v3.log 2>&1 &
# ============================================================
cd /home/acsl/projects/Issacsim-rhukf
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u

# ── 잔여 시뮬레이터 정리 (호출자 셸 명령줄과 패턴 충돌을 피하려 스크립트 안에서 수행) ──
for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done
for p in $(ps -eo pid,args | grep "isaacsim/kit" | grep -v grep | awk '{print $1}'); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 5

# ── 확정 POMDP env + 신규 기동 설정 ──
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 UKF_Q_GYRO=5e-3 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=1.0 COM_BIAS_STD=0.05

OUT=results_zu_v3
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] capture zu_v3 시작 (amp0.5 freq1.0 R1.6 omega0.75, seed42 ep16)"
timeout 3600 python3 -u online_rl_main.py --headless --agent adam --max-ep 16 \
  --speed 20 --log-zu --seed 42 --outdir "$OUT" > "$OUT/cap.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] DONE rc=$rc"
ls -la "$OUT/zu_log.npz" 2>/dev/null || echo "⚠ zu_log.npz 없음 — $OUT/cap.log 확인"
