#!/usr/bin/env bash
# ============================================================
# run_p0_flightcheck.sh — Phase 0 비행 검증 (2026-07-29)
#   nohup ./run_p0_flightcheck.sh > p0_flightcheck.log 2>&1 &
#
# 목적: 플랜트/모델 정합 + 각도 감사 수정이 실제 비행에서 정상인지 한 번에 확인.
#   bias=0(무공격) × 5패턴 × (dhover 0/3/5/7 + track) = 25 cells × EP ep, disturbance=none.
#   dhover 셀은 bias=0 이어도 **호버 전환을 실제로 수행**하므로 요 슬루 확인에 그대로 쓰인다.
#
# 확인 항목:
#   1) [MOTOR] 배너에 τ_up/τ_down 정상 출력 (비대칭 지연 활성)
#   2) [MASS]  배너 총질량 1.372kg
#   3) 무공격인데 추락하는 셀이 없는가 (모터 지연 주입 후 비행 안정성)
#   4) max_roll/max_pitch 분포 — 자세 클립 1.05(60.2°) 상향의 실제 도달 빈도
#   5) circle 패턴 요 추종 (yaw wrap 수정 후) + 호버 전환 요 슬루 소멸 (구 중앙 84.5°)
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3

# ── 플랜트: 비대칭 모터 지연 중간값 (실기 τ 실측 전 잠정) ──
export MOTOR_TAU_UP="${MOTOR_TAU_UP:-0.03}"
export MOTOR_TAU_DOWN="${MOTOR_TAU_DOWN:-0.07}"

PATS="${PATS:-hover,waypoint,circle,figure8,aggressive}"
EP="${EP:-2}"
OUT="${OUT:-results_p0_flightcheck}"
MAXWAIT="${MAXWAIT:-5400}"      # 90분 상한

mkdir -p "${OUT}"
echo "=========================================================="
echo " [$(date +%H:%M:%S)] P0 FLIGHT CHECK"
echo "   MOTOR_TAU_UP=${MOTOR_TAU_UP}s  MOTOR_TAU_DOWN=${MOTOR_TAU_DOWN}s"
echo "   patterns=${PATS}  bias=0  ep=${EP}  → ${OUT}"
echo "=========================================================="

setsid ${PY} online_rl_main.py --sweep --headless --speed 10 --log-zu \
    --capture-mode deadline \
    --deadline-patterns "${PATS}" \
    --deadline-biases 0 \
    --episodes "${EP}" \
    --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!
echo " [pid ${pid}] 시작. 배너 대기..."

# ── 게이트: 90초 내 [MOTOR]/[MASS]/셀수 배너 확인 ──
for i in $(seq 1 45); do
  sleep 2
  grep -qE "CAPTURE:deadline|Traceback" "${OUT}/run.log" 2>/dev/null && break
done
echo "──────── 게이트 ────────"
grep -m2 -E "\[MOTOR\]" "${OUT}/run.log" 2>/dev/null || echo " ⚠ [MOTOR] 배너 없음 — 지연이 비활성일 수 있음"
grep -m2 -E "\[MASS\]"  "${OUT}/run.log" 2>/dev/null || echo " ⚠ [MASS] 배너 없음"
grep -m1 -E "CAPTURE:deadline.*cells" "${OUT}/run.log" 2>/dev/null || echo " ⚠ 셀 배너 없음"
if grep -q "Traceback" "${OUT}/run.log" 2>/dev/null; then
  echo " ✗ 기동 중 예외 발생 — 중단"; tail -30 "${OUT}/run.log"; kill -- -${pid} 2>/dev/null; exit 1
fi
echo "────────────────────────"

# ── 완료 대기(상한 있음) ──
waited=0
while kill -0 ${pid} 2>/dev/null; do
  sleep 20; waited=$((waited+20))
  if [ ${waited} -ge ${MAXWAIT} ]; then
    echo " ⚠ ${MAXWAIT}s 초과 — 강제 종료"; kill -- -${pid} 2>/dev/null; break
  fi
done
echo " [$(date +%H:%M:%S)] DONE (경과 ${waited}s) → ${OUT}"
ls -la "${OUT}"/*.csv "${OUT}"/zu_log.npz 2>/dev/null
