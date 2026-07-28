#!/usr/bin/env bash
# run_verify_frame.sh — 질량정합 + 재캘리브레이션 + 오일러 프레임 수정 검증용 짧은 캡처
#   확인 항목: (1) [MASS] 총 1.372kg  (2) 호버 전환 시 요 슬루 소멸(기존 중앙 84.5°)
#              (3) 항력/추력 모델 정합 (4) NIS 기준선
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
OUT="results_verify_frame"
[ -d "${OUT}" ] && mv "${OUT}" "${OUT}_old_$(date +%H%M%S)"
mkdir -p "${OUT}"
echo " [$(date +%H:%M:%S)] VERIFY START → ${OUT}"
export ROTOR_LOG="$(pwd)/${OUT}/rotor_log.npz"
exec python3 -u online_rl_main.py --sweep --headless --agent adam --speed 10 \
    --seed 0 --sweep-values 0.0 --sweep-pattern aggressive \
    --sweep-wind-type none --sweep-wind-speed 0.0 \
    --episodes 2 --log-zu --log-sysid \
    --outdir "${OUT}" > "${OUT}/run.log" 2>&1
