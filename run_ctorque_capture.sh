#!/usr/bin/env bash
# ============================================================
# run_ctorque_capture.sh — C_torque 축별 재검증용 ROTOR_LOG 캡처 (2026-07-29)
#   nohup ./run_ctorque_capture.sh > ctorque_capture.log 2>&1 &
#
# 배경: fit_static_from_rotor.py 가 낸 C_torque_y=4.017 이 기하학적 최대 피치 토크
#   2.555 N·m 를 157% 초과한다. 또 롤 팔(0.213m)이 피치 팔(0.131m)보다 1.6배 길어
#   롤 권한이 더 커야 하는데 현재 값은 C_x(3.568) < C_y(4.017) 로 역전돼 있다.
#
# 캡처: bias=0(무공격), 모터지연 OFF, 5패턴 전체.
#   패턴별 여기 특성이 달라야 축 분리가 된다:
#     circle    — 요가 접선방향 → 구심가속이 바디 y → **롤 우세, 정상상태**
#     waypoint  — 직선구간, 요가 진행방향 → 가감속이 바디 x → **피치 우세**
#     figure8   — 혼합
#     aggressive— 양축 대진폭 (상한 근처 여기)
#     hover     — 트림 성분(결합항 K) 기준선
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3

OUT="${OUT:-results_ctorque}"
EP="${EP:-2}"
PATS="${PATS:-hover,waypoint,circle,figure8,aggressive}"
MAXWAIT="${MAXWAIT:-5400}"

mkdir -p "${OUT}"
# ── run_sim 은 Popen 이 env 를 상속하므로 export 로 전달된다 ──
export ROTOR_LOG="$(pwd)/${OUT}/rotor_log.npz"
export MOTOR_TAU_UP=0
export MOTOR_TAU_DOWN=0

echo "=========================================================="
echo " [$(date +%H:%M:%S)] C_TORQUE 재검증 캡처"
echo "   ROTOR_LOG=${ROTOR_LOG}   모터지연 OFF"
echo "   patterns=${PATS}  bias=0  ep=${EP}  → ${OUT}"
echo "=========================================================="

setsid ${PY} online_rl_main.py --sweep --headless --speed 10 \
    --capture-mode deadline \
    --deadline-patterns "${PATS}" \
    --deadline-biases 0 \
    --episodes "${EP}" \
    --outdir "${OUT}" \
    > "${OUT}/run.log" 2>&1 &
pid=$!
echo " [pid ${pid}] 시작. 배너 대기..."
for i in $(seq 1 45); do
  sleep 2
  grep -qE "CAPTURE:deadline|Traceback" "${OUT}/run.log" 2>/dev/null && break
done
grep -m1 -E "CAPTURE:deadline.*cells" "${OUT}/run.log" 2>/dev/null || echo " ⚠ 셀 배너 없음"
if grep -q "Traceback" "${OUT}/run.log" 2>/dev/null; then
  echo " ✗ 예외 — 중단"; tail -25 "${OUT}/run.log"; kill -- -${pid} 2>/dev/null; exit 1
fi

waited=0
while kill -0 ${pid} 2>/dev/null; do
  sleep 20; waited=$((waited+20))
  if [ ${waited} -ge ${MAXWAIT} ]; then
    echo " ⚠ ${MAXWAIT}s 초과 — 강제 종료"; kill -- -${pid} 2>/dev/null; break
  fi
done
echo " [$(date +%H:%M:%S)] 캡처 완료 (경과 ${waited}s)"
grep -c "\[ROTOR\]" "${OUT}/sim_process.log" 2>/dev/null | xargs -I{} echo " [ROTOR] 저장 횟수 {}"
ls -la "${OUT}/rotor_log.npz" 2>/dev/null || { echo " ✗ rotor_log.npz 없음"; exit 1; }

echo
echo "════════ 독립 검증 ════════"
~/isaacsim/python.sh verify_ctorque_bounds.py "${OUT}/rotor_log.npz"
echo
echo "════════ 기존 적합 코드 (대조, --write 안 함) ════════"
~/isaacsim/python.sh calibration/fit_static_from_rotor.py "${OUT}/rotor_log.npz" || true
