#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_isaac_manual.sh — Isaac Sim 조종기 수동비행 + 오프보드 시퀀스 검증
#
#   bash px4_field/launch_isaac_manual.sh <f1..f8|f3b|f4b> [speed] [extra args...]
#
#   실비행 전 2단계 리허설. "조종기로 이륙 → 오프보드 스위치 ON →
#   스크립트가 시퀀스 비행 → 스위치 OFF 인계" 전 과정을 가상에서 재현한다.
#
#   ★ 비행 번호는 2026-08-07 확정 F1~F8 체계다 (RUN.txt 와 동일).
#        f1  호버(+수직스윕은 수동)      f1_hover.py
#        f2  오토튠                      autotune_go.py       ← 오프보드 아님
#        f3  doublet                     f2_doublet.py        (구 F2)
#        f3b yaw_spin                    f5_pattern.py        요축 보조
#        f4  drag 등속왕복               f3_drag.py           (구 F3)
#        f4b accel_line                  f5_pattern.py        drag 속도스윕 보조
#        f5  waypoint     f6 circle      f7 figure8      f8 aggressive
#                                        f5_pattern.py --pattern ...
#      ⚠ 구 번호(f1/f2/f3 = hover/doublet/drag)로 쓰던 습관과 f3/f4 가 어긋난다.
#        구 호출을 위해 별칭 f2old(doublet) / f3old(drag) 를 남겨둔다.
#
#   [speed]     Isaac 물리 속도 (기본 1.0)
#               ★ 수동 조종이므로 1.0 고정 권장. RTF>1 이면 사람 반응이
#                 시뮬레이션 시간 기준으로 느려져 조종성이 나빠진다.
#   [extra]     시퀀스 스크립트로 그대로 전달 (예: --dur 30)
#
# 예:
#   bash px4_field/launch_isaac_manual.sh f1
#   bash px4_field/launch_isaac_manual.sh f3 1.0 --n 3 --thrust 0.33
#   bash px4_field/launch_isaac_manual.sh f6 1.0 --laps 1        # circle 1바퀴
#   HEADLESS=1 bash px4_field/launch_isaac_manual.sh f1          # 화면 없는 환경
#
# 조종기 설정은 px4_field/MANUAL_FLIGHT_SETUP.md,
# 8비행 전체 절차는 px4_field/ISAAC_F1_F8.txt 를 볼 것.
# ─────────────────────────────────────────────────────────────────────────────
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

SEQ="${1:?시퀀스 필요: f1 f2 f3 f3b f4 f4b f5 f6 f7 f8 f9 f13}"
PATARG=()
SCRIPT_DIR="$HERE"      # 기본은 px4_field. f9 만 저장소 루트의 rc_attack_trigger.py 를 쓴다
case "$SEQ" in
    f1)          SCRIPT=f1_hover.py   ;;
    f2)          SCRIPT=autotune_go.py ;;
    f3|f2old)    SCRIPT=f2_doublet.py ;;
    f4|f3old)    SCRIPT=f3_drag.py    ;;
    f3b)         SCRIPT=f5_pattern.py; PATARG=(--pattern yaw_spin)   ;;
    f4b)         SCRIPT=f5_pattern.py; PATARG=(--pattern accel_line) ;;
    f5)          SCRIPT=f5_pattern.py; PATARG=(--pattern waypoint)   ;;
    f6)          SCRIPT=f5_pattern.py; PATARG=(--pattern circle)     ;;
    f7)          SCRIPT=f5_pattern.py; PATARG=(--pattern figure8)    ;;
    f8)          SCRIPT=f5_pattern.py; PATARG=(--pattern aggressive) ;;
    f9)          # ★ 공격 주입(RC 트리거). 수동조종 비행 중 /attack_config 발행 → run_sim wrench.
                 SCRIPT=rc_attack_trigger.py; SCRIPT_DIR="$ROOT" ;;
    f13)         # ★ 실시간 탐지 정책 리허설(2026-09-21): shadow UKF+NN+failsafe hover. 조종기: 이륙 → 오프보드 ON → 패턴 중
                 #   다른 창 `python3 etc/analysis/rc_attack_trigger.py --stdin` (t 0.3 = 토크 30%) 로 공격 → 정책이 hover.
                 #   무인 자동검증: HEADLESS=1 ... f13 1.0 --attack dds --sitl-auto 2.5
                 SCRIPT=f13_policy.py; PATARG=(--model models/swirl_v2_s42.npz --pattern circle --attack rc --fs-url udpin:0.0.0.0:14540 --no-fetch) ;;
    *)  echo "✗ 알 수 없는 시퀀스 '$SEQ' (f1 f2 f3 f3b f4 f4b f5 f6 f7 f8 f9)"; exit 1 ;;
esac
if [ $# -ge 2 ] && [[ "${2}" != --* ]]; then SPEED="$2"; shift 2
else SPEED="1.0"; shift 1; fi

HEADLESS="${HEADLESS:-0}"
[ "$HEADLESS" = "0" ] && HFLAG="--no-headless" || HFLAG="--headless"

# ── 0) 잔재 정리 확인 ────────────────────────────────────────────────────
#  런처의 준비판정이 /gt/odometry 존재라서, 이전 세션이 살아 있으면 새 Isaac
#  부팅이 끝나기도 전에 옛 엔진 상태를 읽고 오동작한다.
if ros2 topic list 2>/dev/null | grep -q "^/gt/odometry$"; then
    echo "✗ /gt/odometry 가 이미 존재 — 이전 엔진이 아직 살아 있습니다:"
    pgrep -af "run_sim.py|online_rl_main.py" || true
    echo "  정리:  pkill -f run_sim.py; pkill -9 -f bin/px4   (토픽 사라진 뒤 재실행)"
    exit 1
fi
# 고아 px4: run_sim 을 죽여도 Pegasus 가 띄운 px4 는 살아남아 포트를 물고 있다
# → 새 PX4 가 못 뜨거나 다른 인스턴스로 떠서 QGC 가 Disconnected 로 남는다.
if pgrep -f bin/px4 >/dev/null 2>&1; then
    echo "✗ 잔재 px4 프로세스 — QGC 연결 실패의 주범입니다:"
    pgrep -af bin/px4
    echo "  정리:  pkill -9 -f bin/px4   (3초 후 재실행)"
    exit 1
fi

# ── 1) uXRCE agent ───────────────────────────────────────────────────────
#  없으면 /fmu/out/* 이 통째로 침묵한다. 스크립트가 nav_state 를 못 읽어
#  오프보드 진입을 영영 감지하지 못한다.
if ! pgrep -x MicroXRCEAgent >/dev/null; then
    echo "[isaac] MicroXRCEAgent 기동"
    nohup MicroXRCEAgent udp4 -p 8888 > "$HOME/.xrce_agent.log" 2>&1 &
    sleep 1
fi

LOGDIR=$(mktemp -d /tmp/isaac_manual_XXXXXX)
echo "[isaac] 시퀀스=$SEQ  speed=${SPEED}x  headless=$HEADLESS"
echo "[isaac] 엔진 로그: $LOGDIR/engine.log"

# ── 2) Isaac 엔진 (백그라운드) ───────────────────────────────────────────
cd "$ROOT"
"$HOME/isaacsim/python.sh" run_sim.py $HFLAG --speed "$SPEED" \
    > "$LOGDIR/engine.log" 2>&1 &
ENGINE_PID=$!
cleanup() {
    echo; echo "[isaac] 정리 중..."
    kill "$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
    pkill -9 -f bin/px4 2>/dev/null
    exit "${1:-0}"
}
trap 'cleanup 130' INT TERM

# ── 3) 부팅 대기 ─────────────────────────────────────────────────────────
echo -n "[isaac] Isaac+PX4 부팅 대기 (최대 ~5분)"
READY=0
for _ in $(seq 1 150); do
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo; echo "[isaac] ✗ 엔진 조기 종료 — tail $LOGDIR/engine.log:"
        tail -20 "$LOGDIR/engine.log"; exit 1
    fi
    if ros2 topic list 2>/dev/null | grep -q "^/gt/odometry$"; then READY=1; break; fi
    echo -n "."; sleep 2
done
echo
[ "$READY" -ne 1 ] && { echo "[isaac] ✗ 300s 내 /gt/odometry 미검출:"; tail -20 "$LOGDIR/engine.log"; cleanup 1; }

echo "[isaac] ✓ 엔진 준비. PX4 가 내보내는 토픽:"
ros2 topic list 2>/dev/null | grep "^/fmu/out/" | sed 's/^/          /'

echo "[isaac] ✓ 제어기 제한값 주입 확인 ([PX4LIM] 로그):"
grep -m1 "\[PX4LIM\]" "$LOGDIR/engine.log" | sed 's/^/          /' \
    || echo "          ⚠ [PX4LIM] 이 안 보입니다 — run_sim.py 의 _PX4_LIMITS 확인"

cat <<'EOF'

  ── QGroundControl 에서 ────────────────────────────────────────────
    1. PX4 SITL 은 UDP 로 자동 연결됨 (Disconnected 면 잔재 px4 확인)
    2. Vehicle Setup → Joystick → Enable joystick input → Calibrate
    3. ★ Joystick → Button Assignment 에서 버튼 하나를 Offboard 로 매핑
       (이게 실기 조종기의 "오프보드 스위치" 역할이다)
    4. Position 모드로 이륙 → 고도 2~3 m 안정 (F2 오토튠은 5 m 이상)
    5. ★ 오프보드 버튼 ON  → 아래 창에서 "OFFBOARD 진입" 확인
       ※ F2(오토튠)만 예외다 — Position 모드 그대로 두고 스틱에서 손을 뗀다
    6. 시퀀스 완주 → Ctrl-C → 오프보드 버튼 OFF → 착륙

    자세한 설정: px4_field/MANUAL_FLIGHT_SETUP.md
    8비행 절차:  px4_field/ISAAC_F1_F8.txt
  ────────────────────────────────────────────────────────────────────

  별도 창에서 감시하면 좋은 것:
    ros2 topic hz   /fmu/in/trajectory_setpoint      # setpoint 나가는가
    ros2 topic echo /fmu/out/actuator_motors         # 모터 4채널 (yaml 추가 후)
    python3 px4_field/watch_offboard.py --hz 10      # 명령 vs 실제 추종

EOF

# ── 4) 시퀀스 (포그라운드) ───────────────────────────────────────────────
#  --bench 를 붙이지 않는다: SITL 은 GPS/위치 추정이 유효하므로 실비행과
#  동일한 경로(위치·속도 setpoint)를 그대로 검증한다. 벤치에서 검증 못 하는
#  F4(drag) 속도 경로가 여기서 처음으로 확인된다.
if [ "$SEQ" = "f9" ]; then
    cat <<'EOF'

  ── F9 공격 주입 (RC 트리거) ────────────────────────────────────────
    수동조종(Position 모드)으로 안정 호버 후, 조종기 스위치+노브로 공격을 켠다.
    QGC 조이스틱 축을 aux 로 매핑했으면 그대로, 아니면 --stdin 으로 조종:
        (다른 터미널에서)  python3 rc_attack_trigger.py --stdin
        t 0.3  → 토크 30%   h 0.2 → 추력 20%   off → 끔   q → 종료
    NIS 는 별 창에서:  python3 px4_field/watch_offboard.py --hz 10
  ────────────────────────────────────────────────────────────────────
EOF
fi
cd "$SCRIPT_DIR"
python3 "$SCRIPT" "${PATARG[@]+"${PATARG[@]}"}" "$@"
cleanup 0
