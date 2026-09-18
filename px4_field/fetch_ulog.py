#!/usr/bin/env python3
"""fetch_ulog.py — 착륙 직후 젯슨에서 ulog 를 바로 받아온다 (2026-08-07 신규)

무엇을 대체하나
  지금까지: 착륙 → SD 를 뽑거나 노트북에 QGC 를 USB 로 붙여 다운로드.
            그런데 **FC 의 USB 포트는 하나뿐이고 젯슨이 이미 물고 있다**
            (오토튠 때문에 상시 연결. RUN.txt [팩2][1] 참조).
            → 케이블을 바꿔 끼워야 했고, 그 사이 젯슨 링크가 끊겼다.
  이제:     그 **같은 MAVLink 링크로 그대로 받는다.** 케이블을 안 건드린다.
            비행 끝 → 이 스크립트 한 줄 → 젯슨에 .ulg 가 떨어진다.

  QGC 가 하는 일과 완전히 같은 프로토콜이다. QGC 도 MAVLink 로 받는다.
  전용 경로가 아니라 표준 로그 다운로드 프로토콜(MAVLink common)이다:
      LOG_REQUEST_LIST  → LOG_ENTRY  (id, 크기, UTC 시각)
      LOG_REQUEST_DATA  → LOG_DATA   (90 바이트 조각)
      LOG_REQUEST_END   → 스트림 종료
  PX4 구현: src/modules/logger/logger.cpp + mavlink/mavlink_log_handler.cpp

★ 언제 실행하나
  **disarm 이후.** PX4 는 arm~disarm 을 한 파일로 쓰고, disarm 전에는 그 파일이
  아직 닫히지 않아 크기가 0 이거나 잘려서 보인다. 착륙 → disarm → 몇 초 뒤 실행.
  ⚠ 전원을 분리하면 못 받는다. **전원 분리 전에** 받을 것.

★ 얼마나 걸리나
  USB CDC-ACM 링크라 보레이트 제한이 없다. 실측 대역이 로그에 표시된다.
  참고: 90 바이트/메시지라 메시지 오버헤드가 크다. 5 MB 로그가 대략 1~3분.
  급하면 --last 1 로 마지막 것만 받는다(기본값이 그렇다).

사용
    python3 fetch_ulog.py --list                 # 목록만 본다 (받지 않음)
    python3 fetch_ulog.py                        # 마지막 로그 1개
    python3 fetch_ulog.py --name f1_0807         # 이름 붙여 저장
    python3 fetch_ulog.py --last 3               # 최근 3개
    python3 fetch_ulog.py --id 43                # log_43 지정
    python3 fetch_ulog.py --out ~/ulog           # 저장 폴더
    python3 fetch_ulog.py --conn udpin:0.0.0.0:14540   # SITL 강제

연결 (autotune_go.py 와 동일 규칙)
    실기 : /dev/ttyACM0  →  없으면  /dev/ttyACM1  →  없으면 SITL udpin:14540

⚠ 젯슨에서 이 스크립트를 돌리는 동안 autotune_go.py 를 같이 돌리지 말 것.
  둘 다 같은 시리얼 포트를 연다. 하나가 못 연다.
"""
import argparse
import os
import sys
import time

from pymavlink import mavutil

CHUNK = 90                      # LOG_DATA 한 개가 나르는 바이트 수 (프로토콜 고정)
STALL_TIMEOUT = 2.0             # 이 시간 동안 LOG_DATA 가 없으면 구멍 재요청
MAX_ROUNDS = 200                # 재요청 라운드 상한 (무한루프 방지)


def autodetect_conn():
    """autotune_go.autodetect_conn 과 같은 규칙. 실기 시리얼 우선, 없으면 SITL."""
    for dev in ('/dev/ttyACM0', '/dev/ttyACM1'):
        if os.path.exists(dev):
            return dev
    return 'udpin:0.0.0.0:14540'


def connect(conn, baud, timeout=15.0):
    kind = '실기(시리얼)' if conn.startswith('/dev/') else 'SITL(UDP)'
    print(f"[연결] {conn}   ({kind})")
    try:
        m = mavutil.mavlink_connection(conn, baud=baud)
    except Exception as e:                                   # noqa: BLE001
        print(f"✗ 포트를 열지 못했습니다: {e}")
        if 'Permission' in str(e):
            print("   → sudo usermod -aG dialout $USER  후  newgrp dialout")
        if 'Device or resource busy' in str(e) or 'busy' in str(e).lower():
            print("   → 다른 스크립트(autotune_go.py 등)가 포트를 쥐고 있습니다")
        sys.exit(1)
    print("[연결] heartbeat 대기...", end='', flush=True)
    hb = m.wait_heartbeat(timeout=timeout)
    if hb is None:
        print("\n✗ heartbeat 없음. FC 전원/케이블, SITL 이면 포트 14540 확인")
        sys.exit(1)
    # comp=0 방어: PX4 오토파일럿은 항상 comp=1 이다. comp=0 을 물었다는 건
    # 링크에 다른 MAVLink 프로세스가 붙어 그쪽 heartbeat 를 먼저 잡은 것 →
    # LOG_* 응답도 가로채여 다운로드가 멎을 수 있다. target 을 comp=1 로 강제하고
    # 경합을 경고한다. (근본 해결은 lsof /dev/ttyACM0 로 그 프로세스를 죽이는 것)
    if m.target_component == 0:
        print("\n⚠ heartbeat comp=0 — 오토파일럿(comp=1)으로 강제.")
        print("   링크에 다른 MAVLink 프로세스가 있으면 다운로드가 멎는다:")
        print("     sudo lsof /dev/ttyACM0   → autotune_go/QGC 등 죽일 것")
        m.target_component = 1
    print(f" ✓  sys={m.target_system} comp={m.target_component}")
    return m


def beat(m, state={'t': 0.0}):
    """1초에 한 번 GCS heartbeat. 지상국이 살아 있다고 PX4 에 알린다.

    로그 전송 자체는 heartbeat 없이도 돌지만, 다운로드가 수 분 걸리는 동안
    PX4 가 이 링크를 "지상국 없음" 으로 보면 스트림 구성이 바뀔 수 있다.
    QGC 도 내내 heartbeat 를 보낸다. 비용이 없으므로 그냥 보낸다.
    """
    now = time.time()
    if now - state['t'] < 1.0:
        return
    state['t'] = now
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def wait_disarm(m, timeout, settle=5.0):
    """시동이 풀릴 때까지 기다린다. 이미 풀려 있으면 즉시 돌아온다.

    ★ 이게 자동 다운로드의 전제다. PX4 는 `SDLOG_MODE=0` 이라 **arm~disarm 을
      한 파일로 쓰고 disarm 때 닫는다**(실기 PX4_6.params 확인). 비행 시퀀스가
      끝난 시점에는 아직 날고 있으므로, 그때 받으면 크기 0 이거나 잘린 파일이다.
      → 조종사가 착륙·disarm 할 때까지 기다렸다가 받는다.

    HEARTBEAT.base_mode 의 SAFETY_ARMED 비트를 본다. 우리가 보내는 GCS
    heartbeat 와 구별하려고 autopilot 필드가 INVALID 가 아닌 것만 센다.
    """
    ARMED = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    state = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        beat(m)
        msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
        if msg is None:
            continue
        if msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            continue                              # 우리(GCS) 자신의 heartbeat
        armed = bool(msg.base_mode & ARMED)
        if armed != state:
            state = armed
            if armed:
                print(f"  시동 상태 — 착륙·disarm 을 기다립니다 "
                      f"(최대 {timeout:.0f}s, Ctrl-C 로 중단)")
            else:
                print(f"  ✓ disarm 확인. 로그가 닫히도록 {settle:.0f}초 대기")
        if not armed:
            time.sleep(settle)
            return True
    print(f"  ⚠ {timeout:.0f}초 안에 disarm 을 못 봤습니다 — 그대로 진행합니다.")
    print("     (아직 날고 있다면 마지막 로그가 크기 0 으로 보일 수 있습니다)")
    return False


def list_logs(m, timeout=10.0):
    """LOG_REQUEST_LIST → LOG_ENTRY 수집. [(id, size, time_utc), ...] 를 돌려준다.

    PX4 는 요청 하나에 num_logs 개의 LOG_ENTRY 를 연달아 보낸다. 마지막까지
    다 왔는지는 첫 응답의 num_logs 로 판정한다(중간에 유실되면 재요청).
    """
    entries = {}
    total = None
    for attempt in range(3):
        m.mav.log_request_list_send(m.target_system, m.target_component, 0, 0xFFFF)
        t0 = time.time()
        while time.time() - t0 < timeout:
            beat(m)
            msg = m.recv_match(type='LOG_ENTRY', blocking=True, timeout=1.0)
            if msg is None:
                if total is not None and len(entries) == total:
                    break
                continue
            total = msg.num_logs
            if msg.num_logs == 0:
                return []
            # size==0 인 엔트리는 "아직 안 닫힌 파일" 이거나 빈 파일이다.
            entries[msg.id] = (msg.id, msg.size, msg.time_utc)
            t0 = time.time()
            if len(entries) == total:
                break
        if total is not None and len(entries) == total:
            break
        print(f"  (목록 재요청 {attempt + 1}/3 — {len(entries)}/{total} 수신)")
    return [entries[k] for k in sorted(entries)]


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f"{n:.1f} {unit}" if unit != 'B' else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def utc_str(t):
    if not t:
        return '(시각 없음)'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def download(m, log_id, size, path):
    """로그 하나를 받아 path 에 쓴다. 성공하면 True.

    한 번에 파일 전체를 요청하고, 스트림이 멎으면 **첫 구멍부터 다시** 요청한다.
    PX4 는 LOG_REQUEST_DATA 의 ofs 로 seek 하므로 구멍 메우기가 그대로 된다.
    (QGC 도 같은 방식이다 — LogDownloadController.cc)

    ⚠ 조각 수가 5 MB 기준 6만 개다. "첫 구멍 찾기"를 매 메시지마다 전체 스캔하면
      O(n²) 가 되어 다운로드보다 스캔이 느려진다. 그래서 `scan` 포인터를 두고
      **연속 수신된 앞부분을 다시 안 보게** 한다 (포인터는 뒤로 가지 않는다).
    """
    nchunks = (size + CHUNK - 1) // CHUNK
    buf = bytearray(size)
    have = bytearray(nchunks)          # 0/1 플래그
    n_have = 0                         # 받은 조각 수
    got = 0                            # 받은 바이트 수
    scan = 0                           # 이 인덱스 앞은 전부 받았다
    t_start = time.time()
    last_print = 0.0

    def first_gap():
        nonlocal scan
        while scan < nchunks and have[scan]:
            scan += 1
        return None if scan >= nchunks else scan

    for rnd in range(MAX_ROUNDS):
        start_chunk = first_gap()
        if start_chunk is None:
            break
        ofs = start_chunk * CHUNK
        m.mav.log_request_data_send(m.target_system, m.target_component,
                                    log_id, ofs, 0xFFFFFFFF)
        t_last = time.time()
        while time.time() - t_last < STALL_TIMEOUT:
            beat(m)
            msg = m.recv_match(type='LOG_DATA', blocking=True, timeout=0.5)
            if msg is None:
                continue
            if msg.id != log_id:
                continue
            t_last = time.time()
            o, cnt = msg.ofs, msg.count
            if o >= size:
                continue
            cnt = min(cnt, size - o)
            idx = o // CHUNK
            if idx < nchunks and not have[idx]:
                have[idx] = 1
                n_have += 1
                got += cnt
            buf[o:o + cnt] = bytes(msg.data[:cnt])

            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                el = now - t_start
                rate = got / el if el > 0 else 0
                pct = 100.0 * got / size if size else 100.0
                eta = (size - got) / rate if rate > 0 else 0
                print(f"\r  {pct:5.1f}%  {human(got)}/{human(size)}  "
                      f"{human(rate)}/s  남은시간 {eta:4.0f}s   ",
                      end='', flush=True)
            if n_have >= nchunks:
                break
        if n_have >= nchunks:
            break
        print(f"\n  … 스트림 정지. 남은 조각 {nchunks - n_have}개 재요청 "
              f"(라운드 {rnd + 1})")

    m.mav.log_request_end_send(m.target_system, m.target_component)
    print()

    missing = nchunks - n_have
    if missing:
        print(f"  ✗ 조각 {missing}개를 못 받았습니다 ({MAX_ROUNDS} 라운드 소진).")
        print("     링크가 불안정합니다. 다시 실행하거나 케이블을 확인하세요.")
        return False

    with open(path, 'wb') as f:
        f.write(buf)
    el = time.time() - t_start
    print(f"  ✓ 저장 {path}  ({human(size)}, {el:.0f}s, 평균 {human(size / max(el, 1e-9))}/s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conn', default=None,
                    help='MAVLink 연결. 생략하면 /dev/ttyACM0 → SITL(udpin:14540) 자동')
    ap.add_argument('--baud', type=int, default=57600,
                    help='시리얼 보레이트 (USB CDC 는 무시한다)')
    ap.add_argument('--list', action='store_true', help='목록만 보고 끝')
    ap.add_argument('--last', type=int, default=1, help='최근 N 개를 받는다 (기본 1)')
    ap.add_argument('--id', type=int, default=None, help='특정 로그 번호만')
    ap.add_argument('--all', action='store_true', help='전부 받는다 (오래 걸린다)')
    ap.add_argument('--name', default=None,
                    help='파일명 접두어. 예 --name f1_0807 → f1_0807_log43.ulg')
    ap.add_argument('--out', default='ulog', help='저장 폴더 (기본 ./ulog)')
    ap.add_argument('--force', action='store_true', help='이미 있는 파일도 다시 받는다')
    ap.add_argument('--wait-disarm', dest='wait_disarm', type=float, default=0.0,
                    help='시동이 풀릴 때까지 최대 N초 기다렸다가 받는다 '
                         '(로그는 disarm 때 닫힌다). 0 = 기다리지 않음')
    ap.add_argument('--quiet-list', dest='quiet_list', action='store_true',
                    help='로그 목록 표를 찍지 않는다 (자동 호출용)')
    a = ap.parse_args()

    conn = a.conn or autodetect_conn()
    m = connect(conn, a.baud)

    if a.wait_disarm > 0:
        print("[대기] 로그는 disarm 때 닫힌다 — 시동 상태 확인 중")
        wait_disarm(m, a.wait_disarm)

    print("[목록] LOG_REQUEST_LIST ...")
    logs = list_logs(m)
    if not logs:
        print("✗ 로그가 없습니다. SD 카드가 꽂혀 있는지, SDLOG_MODE 를 확인하세요.")
        return 1

    if not a.quiet_list:
        print(f"\n  로그 {len(logs)}개")
        print("     id        크기   기록시각")
        for lid, size, tutc in logs:
            mark = ' ←' if lid == logs[-1][0] else ''
            print(f"    {lid:4d}  {human(size):>10}   {utc_str(tutc)}{mark}")
        print("    (← 가 마지막 로그)")

    if a.list:
        return 0

    # 받을 대상 고르기
    if a.id is not None:
        targets = [e for e in logs if e[0] == a.id]
        if not targets:
            print(f"\n✗ log id {a.id} 가 목록에 없습니다.")
            return 1
    elif a.all:
        targets = logs
    else:
        targets = logs[-max(1, a.last):]

    # 크기 0 은 아직 안 닫힌 파일이다 — 받아봐야 쓸모없다
    skipped = [e for e in targets if e[1] == 0]
    targets = [e for e in targets if e[1] > 0]
    for lid, _, _ in skipped:
        print(f"\n  ⚠ log {lid} 은 크기 0 입니다 — 아직 안 닫혔거나 빈 파일.")
        print("     disarm 후 몇 초 기다렸다가 다시 실행하세요.")
    if not targets:
        return 1

    outdir = os.path.expanduser(a.out)
    os.makedirs(outdir, exist_ok=True)
    total = sum(e[1] for e in targets)
    print(f"\n[받기] {len(targets)}개, 합계 {human(total)}  →  {outdir}/")

    ok = 0
    saved = []
    for lid, size, tutc in targets:
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(tutc)) if tutc else 'nodate'
        base = f"{a.name}_" if a.name else ''
        path = os.path.join(outdir, f"{base}log{lid}_{stamp}.ulg")
        if os.path.exists(path) and not a.force:
            print(f"\n  log {lid}: 이미 있음 → 건너뜀 ({path})  [--force 로 덮어쓰기]")
            ok += 1
            saved.append(path)
            continue
        print(f"\n  log {lid}  {human(size)}")
        if download(m, lid, size, path):
            ok += 1
            saved.append(path)

    print(f"\n[완료] {ok}/{len(targets)}")
    # 자동 호출부(offboard_common)가 파일 경로를 집어갈 수 있게 기계가 읽는 줄로도 찍는다
    for p in saved:
        print(f"SAVED\t{os.path.abspath(p)}")
    if ok and not a.quiet_list:
        print("\n  바로 판독하려면:")
        print("    python3 px4_field/check_ulog.py <파일>.ulg --type hover --mass 1.340")
        print("    ~/isaacsim/python.sh px4_field/fit_autotune.py  <파일>.ulg   (F2 오토튠)")
        print("    ~/isaacsim/python.sh px4_field/fit_from_ulog.py <파일>.ulg --mass 1.340")
    return 0 if ok == len(targets) else 1


if __name__ == '__main__':
    sys.exit(main())
