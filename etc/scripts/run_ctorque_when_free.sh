#!/usr/bin/env bash
# ============================================================
# run_ctorque_when_free.sh — GPU 가 비면 C_torque 검증 캡처를 시작한다 (2026-07-29)
#   nohup ./run_ctorque_when_free.sh > ctorque_queue.log 2>&1 &
#
# ⚠ 대기 조건에 pgrep 를 쓰지 않는다. 패턴 문자열이 이 스크립트 자신의 명령줄에도
#   들어 있어 pgrep -f 가 자기 자신을 매칭한다(실제로 2회 사고). nvidia-smi 는
#   프로세스 목록이 아니라 GPU 컨텍스트를 보므로 자기 매칭이 원리적으로 불가능하다.
# ============================================================
set -u
cd /home/acsl/projects/Issacsim-rhukf

IDLE_NEED="${IDLE_NEED:-420}"      # 연속 유휴 요구 시간(초)
MAXWAIT_Q="${MAXWAIT_Q:-43200}"    # 큐 대기 상한 12시간

busy() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
   | grep -v MATLAB \
   | awk -F', ' '{gsub(/ MiB/,"",$3); if ($3+0 > 1000) c++} END {exit !(c>0)}'
}

echo " [$(date '+%H:%M:%S')] 큐 시작 — GPU 유휴 ${IDLE_NEED}s 연속 대기"
start=$(date +%s); last_busy=$(date +%s)
while true; do
  now=$(date +%s)
  if busy; then last_busy=$now; fi
  if [ $((now - last_busy)) -ge "${IDLE_NEED}" ]; then
    echo " [$(date '+%H:%M:%S')] GPU 유휴 확인 — 캡처 시작"
    break
  fi
  if [ $((now - start)) -ge "${MAXWAIT_Q}" ]; then
    echo " [$(date '+%H:%M:%S')] 큐 대기 상한 초과 — 중단"; exit 1
  fi
  sleep 20
done

exec ./run_ctorque_capture.sh
