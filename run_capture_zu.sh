#!/usr/bin/env bash
# ============================================================
# run_capture_zu.sh — 오프라인 gyro-Q sweep용 (z,u) 스트림 캡처 (짧은 고정정책 sweep)
#   nohup ./run_capture_zu.sh > sweep_capture_zu.log 2>&1 &
#
# 목적: results_capture/zu_log.npz 생성 → replay_q_sweep.py 로 Q_gyro 오프라인 재계산.
#   RL 재학습 아님. combined ft1.5 ramp0.0 동결 세팅.
#   grid: 0(정상/급기동 baseline) + 밴드 1.34/1.37/1.40. delays=1(dhover 최소). ep=5.
#   온셋 track-attack 샘플 = 3밴드 × 5ep = 15,  정상 baseline = bias0 전정책 × 5.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
GRID="0,1.34,1.37,1.40"
RAMP="0.0"; EP="5"; FT="1.5"; DELAYS="1"
OUT="results_capture"
mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] START capture-zu combined ft=${FT} ramp=${RAMP}s ep=${EP} grid=${GRID} → ${OUT}"
echo "=========================================================="
setsid ${PY} online_rl_main.py --sweep --headless --speed 10 --log-zu \
    --ramp "${RAMP}" --episodes "${EP}" \
    --sweep-mode combined --ft-ratio "${FT}" --sweep-values "${GRID}" \
    --hover-delays "${DELAYS}" \
    --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!
echo " [pid ${pid}] 캡처 시작. 30초 내 배너 확인:"
for i in $(seq 1 60); do
  sleep 2
  if grep -qE "\[ZU\]|SWEEP DONE|Traceback" "${OUT}/run.log" 2>/dev/null; then break; fi
  b=$(grep -oE "cells=[0-9]+ .*ft_ratio=[0-9.]+" "${OUT}/run.log" 2>/dev/null | head -1)
  [ -n "$b" ] && { echo " [gate] $b"; }
done
wait ${pid}
echo " [$(date +%H:%M:%S)] DONE capture-zu → ${OUT}/zu_log.npz"
ls -la "${OUT}/zu_log.npz" 2>/dev/null || echo " ⚠ zu_log.npz 없음 — run.log 확인"
