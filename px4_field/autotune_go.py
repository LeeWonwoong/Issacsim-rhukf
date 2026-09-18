#!/usr/bin/env python3
"""autotune_go.py — PX4 오토튠을 MAVLink 로 시작한다 (2026-08-06 개정)

★★ 2026-08-06 개정 이유 — 구현이 통째로 바뀌었다 ★★
  구 버전은 ROS2 `/fmu/in/vehicle_command` 로 MAVLink 명령 212
  (VEHICLE_CMD_DO_AUTOTUNE_ENABLE)를 쐈다. **v1.17 이하에서는 그 경로가 없다.**

      버전            오토튠 모듈이 212 를 직접 받나   시작 트리거
      v1.15.4~v1.17.0        ✗                    MC_AT_START 파라미터
      main(1.18-alpha)       ✓                    명령 212 직접

  v1.16.2 에서 212 를 처리하는 곳은 오토튠 모듈이 아니라 **MAVLink 수신기**다:
      mavlink_receiver.cpp:612  MAV_CMD_DO_AUTOTUNE_ENABLE
                                → param_find("MC_AT_START") 를 1 로 세팅
  그리고 모듈은 그 파라미터만 본다:
      mc_autotune_attitude_control.cpp:274  if (_param_mc_at_start.get())

  → uORB/DDS 로 쏜 VehicleCommand 는 mavlink_receiver 를 거치지 않으므로
    **아무도 받지 않는다.** 2026-08-06 SITL 리허설이 통과했던 건 그때 SITL 이
    1.18 이었기 때문이고, 1.15.4 실기에서도 원래 동작하지 않았을 것이다.

  그래서 이제 **MAVLink 로 MC_AT_START=1 을 직접 쓴다.** QGC 의 Autotune 버튼이
  결국 하는 일과 동일하며 v1.15/1.16/1.17 전부에서 동작한다.

연결 (자동 선택)
      실기 : /dev/ttyACM0   ← 젯슨 USB-C ↔ FC USB.
                              rcS:501-505 가 USB 연결 시 mavlink 를 자동 기동한다.
      SITL : udpin:0.0.0.0:14540
                              px4-rc.mavlink:26 이 offboard 링크를 14540 으로 보낸다.
      --conn 으로 직접 지정 가능.

사전 조건 (지상에서 QGC 로 1회)
      MC_AT_EN    = 1     ← rc.mc_apps:23 이 **부팅 시** 검사한다. 설정 후 재부팅
      MC_AT_APPLY = 0     ← ★ 새 게인을 적용하지 않는다(로깅 전용).
                            1 이면 착륙 후 게인이 실제로 바뀌어 F1/F2/F3 비교가 깨진다
      MC_AT_SYSID_AMP = 0.7 (기본)
      이 스크립트가 시작 전에 셋 다 읽어서 확인해준다.

비행 조건
      Position 모드, 고도 5 m 이상, 반경 10 m 여유, 바람 잔잔. 호버 안정 후 실행.

실행 중 조종사 주의 (PX4 가 즉시 중단하는 조건)
      · 롤/피치 스틱 0.05 이상 → 중단      ← 이게 비상탈출이다
      · 비행모드 변경           → 중단
      · 한 축이 20초 안에 수렴 못 함 → 중단
      MC_AT_APPLY=0 이라 중단돼도 기체 특성은 변하지 않는다.

진행
      롤 흔듦 → 2초 정지 → 피치 → 2초 → 요 → 2초 → 끝.  총 25~45초.
      ※ SITL 실측(v1.16.2, 2026-08-06): ROLL 5.1s → pause 2.1s → PITCH 5.1s
        → pause 2.1s → YAW 14.6s → pause 2.1s.  총 약 37s. 요가 가장 오래 걸린다.
      ★ PX4 는 끝나면 MC_AT_START 를 스스로 0 으로 되돌린다
        (mc_autotune_attitude_control.cpp:583-584).
        이 스크립트는 그 0 복귀를 폴링해서 **완료/중단 시점을 알려준다.**

★ "부팅당 1회" 제약은 v1.16.2 에는 **없다** (2026-08-06 SITL 실측으로 정정)
      한 부팅 안에서 3회 연속 완주를 확인했다.
      1.18 에서는 트리거가 명령 212 이고 stopAutotune() 이 자기 웨이크업 소스
      (vehicle_torque_setpoint 콜백)를 해제해 두 번째가 안 먹었다.
      1.16.2 는 트리거가 **파라미터**라 param_update 콜백이 계속 살아 있다.
      → 현장에서 실패해도 같은 팩에서 재시도 가능하다.

결과 판독
      ulog 의 `autotune_attitude_control_status` (기본 로깅 프로파일, 10 Hz)
          coeff[5] = [a1, a2, b0, b1, b2],  dt_model,  state
          ★ state 는 2=ROLL 4=PITCH 6=YAW (3/5/7 은 각 축 뒤의 2초 pause).
            2026-08-06 이전 fit_autotune.py 가 3/6/9 로 읽어 축이 통째로 어긋났었다.
      정합성 검사 :  a1 + a2 ≈ −1      (토크→각속도는 적분기)
          G = (b0+b1+b2) / (dt_model · (1 − a2))    [rad/s² per 정규화 단위]
      → 워크스테이션에서 fit_autotune.py 로 뽑는다.

사용
      python3 autotune_go.py                     # 자동 연결, 3초 카운트다운
      python3 autotune_go.py --now               # 즉시
      python3 autotune_go.py --conn udpin:0.0.0.0:14540    # SITL 강제
      python3 autotune_go.py --conn /dev/ttyACM0           # 실기 강제
      python3 autotune_go.py --check             # 사전조건만 확인하고 끝(발사 안 함)
"""
import argparse
import os
import struct
import sys
import time

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit('✗ pymavlink 가 없다.   pip install pymavlink')

# ── PX4 정수 파라미터의 MAVLink 인코딩 (2026-08-06 실측으로 확인) ──
#  PX4 는 INT32 파라미터를 PARAM_VALUE.param_value(float 필드)에 **비트 그대로**
#  실어 보낸다(바이트 단위 인코딩). 그래서 float 로 그냥 읽으면
#      MC_AT_EN = 1  →  1.4013e-45   (= 0x00000001 을 float 로 해석한 값)
#  처럼 보인다. 읽을 때도 쓸 때도 반드시 비트 재해석을 해야 한다.
#  ⚠ 쓰기를 안 고치면 MC_AT_START=1 이 1.0f 의 비트(0x3F800000)로 나가
#    PX4 가 1065353216 으로 받는다 → 오토튠이 시작되지 않는다.
_INT_TYPES = (
    mavutil.mavlink.MAV_PARAM_TYPE_UINT8,  mavutil.mavlink.MAV_PARAM_TYPE_INT8,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT16, mavutil.mavlink.MAV_PARAM_TYPE_INT16,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT32, mavutil.mavlink.MAV_PARAM_TYPE_INT32,
)


def _decode(param_value, param_type):
    """PARAM_VALUE → 사람이 읽는 값."""
    if param_type in _INT_TYPES:
        return float(struct.unpack('<i', struct.pack('<f', param_value))[0])
    return param_value


def _encode_int(value):
    """정수 → PARAM_SET.param_value 에 실을 float(비트 재해석)."""
    return struct.unpack('<f', struct.pack('<i', int(value)))[0]

PARAMS_TO_CHECK = [
    # (이름, 기대값, 필수인가, 설명)
    ('MC_AT_EN',        1,   True,  '0 이면 오토튠 모듈이 아예 안 뜬다 (rc.mc_apps:23). 설정 후 재부팅 필요'),
    ('MC_AT_APPLY',     0,   True,  '1 이면 착륙 후 게인이 실제로 바뀐다 → F1/F2/F3 비교가 깨진다'),
    ('MC_AT_SYSID_AMP', 0.7, False, '여기 신호 진폭. 요축이 약하면 1.0~1.5'),
]


def autodetect_conn():
    for dev in ('/dev/ttyACM0', '/dev/ttyACM1'):
        if os.path.exists(dev):
            return dev
    return 'udpin:0.0.0.0:14540'


def get_param(m, name, timeout=3.0):
    """파라미터 하나를 읽는다. 정수형은 비트 재해석해서 돌려준다. 못 읽으면 None."""
    m.mav.param_request_read_send(m.target_system, m.target_component,
                                  name.encode(), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg and msg.param_id.strip('\x00') == name:
            return _decode(msg.param_value, msg.param_type)
    return None


def set_param_int(m, name, value):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                         _encode_int(value), mavutil.mavlink.MAV_PARAM_TYPE_INT32)


def set_param_float(m, name, value):
    """실수 파라미터(예: MC_AT_SYSID_AMP)를 쓴다. 쓴 값이 되읽혀 확인되면 True."""
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                         float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg and msg.param_id.strip('\x00') == name:
            return abs(msg.param_value - float(value)) < 1e-3
    return False


def check_sticks(m, seconds=2.0):
    """조종기 롤/피치의 최대 이탈량을 본다. (roll, pitch) 또는 못 받으면 None.

    PX4 는 |roll|>0.05 또는 |pitch|>0.05 면 오토튠을 즉시 중단한다
    (mc_autotune_attitude_control.cpp:443-452). MANUAL_CONTROL 은 x=pitch,
    y=roll 이며 -1000~1000 스케일이다.
    """
    r_max = p_max = 0.0
    got = False
    t0 = time.time()
    while time.time() - t0 < seconds:
        msg = m.recv_match(type='MANUAL_CONTROL', blocking=True, timeout=0.5)
        if msg is None:
            continue
        got = True
        r_max = max(r_max, abs(msg.y / 1000.0))
        p_max = max(p_max, abs(msg.x / 1000.0))
    return (r_max, p_max) if got else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conn', default=None,
                    help='MAVLink 연결 문자열. 생략하면 /dev/ttyACM0 → SITL(udpin:14540) 순으로 자동')
    ap.add_argument('--baud', type=int, default=57600,
                    help='시리얼 보레이트 (USB CDC 는 무시한다)')
    ap.add_argument('--now', action='store_true', help='카운트다운 없이 즉시')
    ap.add_argument('--check', action='store_true', help='사전조건만 확인하고 끝')
    ap.add_argument('--force', action='store_true',
                    help='스틱 중립 경고를 무시하고 발사')
    ap.add_argument('--timeout', type=float, default=90.0,
                    help='완료 대기 상한 [s]')
    ap.add_argument('--amp', type=float, default=None,
                    help='MC_AT_SYSID_AMP 여기 진폭을 발사 전에 설정. 요축이 약해 미수렴이면 1.0~1.5. '
                         '미지정이면 기존 값 유지. (예: --amp 1.3)')
    ap.add_argument('--yaw', action='store_true',
                    help='요축 집중 프리셋: --amp 1.3 --repeat 2 와 동치. 요 오토튠이 반복 미수렴할 때.')
    ap.add_argument('--repeat', type=int, default=1,
                    help='한 비행에서 오토튠을 N회 반복(각 ~40s). 요 수렴 확률↑·평균화. '
                         '⚠ PX4 가 부팅당 1회로 막으면 2회차가 즉시 중단될 수 있다 → ulog 로 판정.')
    a = ap.parse_args()
    if a.yaw:                       # 프리셋: 요축 집중
        if a.amp is None:
            a.amp = 1.3
        a.repeat = max(a.repeat, 2)

    conn = a.conn or autodetect_conn()
    kind = '실기(시리얼)' if conn.startswith('/dev/') else 'SITL(UDP)'
    print(f'\n  연결 : {conn}   [{kind}]')

    try:
        m = (mavutil.mavlink_connection(conn, baud=a.baud)
             if conn.startswith('/dev/') else mavutil.mavlink_connection(conn))
    except Exception as e:
        sys.exit(f'✗ 연결 실패: {e}')

    print('  heartbeat 대기...', flush=True)
    hb = m.wait_heartbeat(timeout=15)
    if hb is None:
        sys.exit('✗ 15초 안에 heartbeat 없음.\n'
                 '   실기: USB 케이블 / dialout 권한 확인\n'
                 '   SITL: PX4 가 떠 있는지, 포트가 14540 인지 확인')
    armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    # ── comp=0 방어 ──────────────────────────────────────────
    #  PX4 오토파일럿은 항상 comp=1 로 heartbeat 를 낸다. wait_heartbeat 가
    #  comp=0 을 물었다는 건 링크에 다른 MAVLink 참가자(GCS/router/남은 스크립트)가
    #  있어 그쪽 heartbeat 를 먼저 잡았다는 뜻이다. 그 상태로 param 을 comp=0 에
    #  요청하면 응답이 안 오거나(버전에 따라) 그 참가자가 PARAM_VALUE 를 가로채
    #  "읽기 실패"가 난다. target 을 오토파일럿(comp=1)으로 강제한다.
    #  ⚠ 그래도 실패하면 근본 원인은 접속 경합이다 → `lsof /dev/ttyACM0` 로
    #    ttyACM0 를 물고 있는 다른 프로세스(QGC 등)를 죽일 것.
    if m.target_component == 0:
        print('  ⚠ heartbeat comp=0 — 오토파일럿(comp=1)으로 강제. '
              '링크에 다른 MAVLink 프로세스가 있는지 lsof 로 확인 권장.')
        m.target_component = 1
    print(f'  기체   : sys={m.target_system} comp={m.target_component}  '
          f'{"ARMED" if armed else "DISARMED"}')

    # ── 사전조건 확인 ────────────────────────────────────────
    print('\n  사전조건')
    blocking = []
    for name, want, required, why in PARAMS_TO_CHECK:
        got = get_param(m, name)
        if got is None:
            print(f'    ?  {name:16s} 읽기 실패')
            if required:
                blocking.append(f'{name} 읽기 실패')
            continue
        ok = abs(got - want) < 1e-3
        mark = 'OK' if ok else '✗ '
        print(f'    {mark} {name:16s} = {got:g}   (기대 {want:g})')
        if not ok:
            print(f'         → {why}')
            if required:
                blocking.append(f'{name}={got:g} (기대 {want:g})')

    if blocking:
        print('\n  ✗ 사전조건 미충족 — QGC 로 고친 뒤 다시 실행:')
        for b in blocking:
            print(f'      {b}')
        print('    ※ MC_AT_EN 을 바꿨으면 **재부팅**해야 모듈이 뜬다.')
        sys.exit(1)

    start_now = get_param(m, 'MC_AT_START')
    if start_now is not None and start_now > 0.5:
        sys.exit('\n  ✗ MC_AT_START 가 이미 1 이다 — 오토튠이 진행 중이거나 이전 시도가 남아 있다.\n'
                 '     끝나기를 기다리거나 FC 를 재부팅할 것.')

    if a.check:
        print('\n  사전조건 통과. (--check 이므로 발사하지 않음)')
        return

    if not armed:
        print('\n  ⚠ 기체가 DISARMED 다. 비행 중(Position 모드, 5m 이상)에 실행할 것.')

    # ── ★ 스틱 중립 확인 ────────────────────────────────────
    #  mc_autotune_attitude_control.cpp:443-452
    #    |manual_control_setpoint.roll| > 0.05  또는 pitch > 0.05  → 즉시 state::fail
    #  즉 조종기가 정확히 중립이 아니면 시작하자마자 중단된다. 미리 본다.
    stick = check_sticks(m, seconds=2.0)
    if stick is None:
        print('\n  ? MANUAL_CONTROL 을 못 받았다 — 스틱 중립 여부를 확인할 수 없다.')
        print('    (조종기 입력이 MAVLink 로 안 오는 구성일 수 있다. 그대로 진행한다)')
    else:
        r, p = stick
        bad = max(abs(r), abs(p)) > 0.05
        print(f'\n  스틱 중립  roll={r:+.3f}  pitch={p:+.3f}   '
              f'{"✗ 한계 0.05 초과" if bad else "OK"}')
        if bad:
            print('    → 이 상태로 발사하면 PX4 가 **즉시 중단**한다'
                  ' (mc_autotune_attitude_control.cpp:443-452).')
            print('    · 조종기 트림을 중립으로 맞추거나 QGC 조이스틱을 재캘리브레이션')
            print('    · 게임패드면 데드존 설정을 키울 것')
            if not a.force:
                sys.exit('    중단한다. 고치고 다시 실행할 것. (무시하려면 --force)')

    # ── ★ 여기 진폭 설정 (요축 미수렴 대응) ──────────────────
    #  MC_AT_SYSID_AMP 는 전 축 공통이지만, 높이면 약한 요축의 SNR 이 올라 수렴한다.
    #  (roll/pitch 는 원래 잘 되므로 과여기해도 MC_AT_APPLY=0 이라 무해)
    if a.amp is not None:
        ok = set_param_float(m, 'MC_AT_SYSID_AMP', a.amp)
        print(f'\n  MC_AT_SYSID_AMP = {a.amp:g}  {"✓ 설정됨" if ok else "⚠ 확인 실패(그대로 진행)"}'
              f'   (요축 집중 여기)')

    print('\n  ★ 발사 후: 롤 → 2초 → 피치 → 2초 → 요.  스틱 건드리면 즉시 중단됨.')
    if a.repeat > 1:
        print(f'  ★ 반복 {a.repeat}회 — 각 실행 후 다음 발사. 요 수렴 확률↑ (ulog 에 실행별로 남는다).')

    def _fire_once():
        """MC_AT_START=1 발사 → 세팅확인. (True=발사됨 / 'noresp' / 'aborted')."""
        seen = None
        for attempt in range(3):
            set_param_int(m, 'MC_AT_START', 1)
            t0 = time.time()
            while time.time() - t0 < 1.5:
                msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
                if msg and msg.param_id.strip('\x00') == 'MC_AT_START':
                    seen = _decode(msg.param_value, msg.param_type)
                    break
            if seen is not None:
                break
            print(f'    (PARAM_VALUE 응답 없음 — 재시도 {attempt + 1}/3)')
        if seen is None:
            return 'noresp'
        if seen < 0.5:
            return 'aborted'
        return True

    def _wait_done():
        """PX4 가 MC_AT_START 를 0 으로 되돌릴 때까지 폴링 = 완료/중단 시점."""
        t0 = time.time(); last = -1.0
        while time.time() - t0 < a.timeout:
            v = get_param(m, 'MC_AT_START', timeout=1.5)
            el = time.time() - t0
            if v is not None and v < 0.5:
                print(f'  ✓ MC_AT_START 0 복귀 — 종료 ({el:.1f}s)')
                return True
            if el - last >= 5.0:
                print(f'    ... 진행 중 {el:.0f}s'); last = el
            time.sleep(1.0)
        print(f'  ⚠ {a.timeout:.0f}s 안에 종료 신호 없음.')
        return False

    n_ok = 0
    for run_i in range(a.repeat):
        if a.repeat > 1:
            print(f'\n  ── 실행 {run_i + 1}/{a.repeat} ──')
        if run_i > 0:
            # 다음 발사 전 MC_AT_START 가 0 인지 확인하고 잠시 안정화
            for _ in range(10):
                if (get_param(m, 'MC_AT_START', timeout=1.0) or 0.0) < 0.5:
                    break
                time.sleep(1.0)
            time.sleep(3.0)
        elif not a.now:
            for i in (3, 2, 1):
                print(f'  {i}...', flush=True); time.sleep(1)

        res = _fire_once()
        if res == 'noresp':
            print('\n  ✗ PARAM_SET 무응답 — 링크는 살아있는데 쓰기가 안 먹는다.'
                  ' target_component/파라미터 잠금 의심.')
            break
        if res == 'aborted':
            msg = ('  ✗ MC_AT_START 가 세팅 직후 0 으로 되돌아옴 = 시작하자마자 중단.\n'
                   '     원인: 스틱이 중립(±0.05) 벗어남 ← 대부분 / 비행상태 아님 / '
                   '부팅당 1회 제약(반복시 2회차부터)')
            if run_i == 0:
                sys.exit('\n' + msg)
            print('\n' + msg + '\n     (반복 중이므로 여기서 멈춘다 — 부팅당 1회 제약일 가능성)')
            break
        print('  >>> 발사 확인. 오토튠 진행...')
        if _wait_done():
            n_ok += 1

    print(f'\n  ── 완료: {n_ok}/{a.repeat} 회 종료신호 확인 ──')
    print('    완주/수렴 판정은 ulog 로:  python3 fit_autotune.py <log>.ulg')
    print('    · state 가 roll→pitch→yaw 를 다 지나고 COMPLETE 면 완주')
    print('    · yaw maxVar > 50 이면 미수렴 → 다음엔 --amp 를 더 높여 재비행'
          + (f' (이번 {a.amp})' if a.amp else ' (--amp 1.3)'))


if __name__ == '__main__':
    main()
