#!/usr/bin/env bash
# ============================================================
# run_align_verify.sh — 2026-08-12 실기 정합 반영분의 **sim 검증** (2026-08-12)
#
#   nohup ./run_align_verify.sh > verify.log 2>&1 &
#
# 무엇을 확인하나 (셋 다 "적용됐지만 미검증" 상태)
#   ① COM_ALIGN  : sim 호버 트림 τ_norm 이 실기 [-0.0073, -0.0554, +0.0042] 와 맞는가
#                   ← 부호를 뒤집기 쉬운 항. 한 번 틀렸던 이력 있음. **최우선**
#   ② ROTOR_CM   : 로터로그의 실제 요 토크 / 명령 → C_torque_z 가 1.378 인가
#   ③ NIS 바닥    : 평시 NIS 가 실기(gyro 0.6~1.2 / vel 0.18~0.41) 수준인가
#
# ⚠ run_ctorque_capture.sh 와 다른 점: **MOTOR_TAU 를 끄지 않는다.**
#   ①③ 은 실운용 설정(0.014)에서 봐야 의미가 있다. ② 는 지연 때문에 약간 감쇠되지만
#   호버/기동 저주파 성분으로 충분히 판정된다.
# ============================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3

OUT="${OUT:-results_verify_0812}"
EP="${EP:-1}"
PATS="${PATS:-hover,aggressive}"     # hover=트림·NIS 기준선, aggressive=요 여기(C_torque_z)
MAXWAIT="${MAXWAIT:-3600}"

mkdir -p "${OUT}"
export ROTOR_LOG="$(pwd)/${OUT}/rotor_log.npz"
# MOTOR_TAU 는 run_sim 기본값(0.014) 그대로 둔다 — 실운용 설정에서 검증

echo "=========================================================="
echo " [$(date +%H:%M:%S)] 정합 검증 캡처"
echo "   ROTOR_LOG=${ROTOR_LOG}   MOTOR_TAU=기본(0.014)"
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
for i in $(seq 1 90); do
  sleep 2
  grep -qE "CAPTURE:deadline|Traceback" "${OUT}/run.log" 2>/dev/null && break
done
grep -m1 -E "CAPTURE:deadline.*cells" "${OUT}/run.log" 2>/dev/null || echo " ⚠ 셀 배너 없음"
if grep -q "Traceback" "${OUT}/run.log" 2>/dev/null; then
  echo " ✗ 예외 — 중단"; tail -30 "${OUT}/run.log"; kill -- -${pid} 2>/dev/null; exit 1
fi

# 플랜트 주입 확인 로그 (있으면 즉시 보여준다)
grep -m1 "\[MASS\]" "${OUT}/sim_process.log" 2>/dev/null || true
grep -m1 "\[COM\]"  "${OUT}/sim_process.log" 2>/dev/null || true
grep -m1 "\[IRIS\]" "${OUT}/sim_process.log" 2>/dev/null || true

waited=0
while kill -0 ${pid} 2>/dev/null; do
  sleep 20; waited=$((waited+20))
  if [ ${waited} -ge ${MAXWAIT} ]; then
    echo " ⚠ ${MAXWAIT}s 초과 — 강제 종료"; kill -- -${pid} 2>/dev/null; break
  fi
done
echo " [$(date +%H:%M:%S)] 캡처 완료 (경과 ${waited}s)"
ls -la "${OUT}/rotor_log.npz" 2>/dev/null || echo " ⚠ rotor_log.npz 없음"
echo
echo "════════ 판정 ════════"
~/isaacsim/python.sh verify_align_0812.py "${OUT}" || true
