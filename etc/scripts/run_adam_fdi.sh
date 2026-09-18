#!/usr/bin/env bash
# ============================================================
# run_adam_fdi.sh — Adam baseline 학습 (2026-08-19 확정 FDI 틸트 config)
#   nohup bash run_adam_fdi.sh > train_adam_fdi.log 2>&1 &
# config v2(2026-08-19 재설계): 리워드 상수 0.5/0.5/-1/-1·term-2.5(딜레이폐기), 압축c=3,
#         강풍6-12(약풍0.5-2 75%), R_gyro0.2, δ0.4-0.8 tilt, 무공격0.3, 300ep.
# 플랜트: MOTOR_TAU0.014·THR_MDL_FAC0.78·drag[.5,.3,0] / UKF R_g0.2 Q_g5e-3 (전부 default).
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf

OUT="results_adam_fdi_v4"
[ -d "${OUT}" ] && mv "${OUT}" "${OUT}_old_$(date +%m%d_%H%M%S)"
mkdir -p "${OUT}"

echo "=========================================================="
echo " [$(date '+%m-%d %H:%M:%S')] ADAM FDI v4 학습 START → ${OUT} (300ep, seed0, FN그래디언트+logcap6압축)"
echo "=========================================================="

python3 -u online_rl_main.py --headless --agent adam --speed 10 \
    --seed 0 --outdir "${OUT}" > "${OUT}/run.log" 2>&1
RC=$?

echo "=========================================================="
echo " [$(date '+%m-%d %H:%M:%S')] ADAM FDI 학습 DONE (rc=${RC})"
echo "ADAM_FDI_DONE rc=${RC} $(date '+%m-%d %H:%M:%S')" > "${OUT}/DONE_MARKER"
echo "=========================================================="
# 마지막 요약 몇 줄
tail -30 "${OUT}/run.log" 2>/dev/null
