#!/usr/bin/env python3
"""check_ulog.py — 실기 로그 현장 검증 (2026-07-29)

★ 현장에서, 짐 싸기 전에 돌린다. FAIL 이면 그 자리에서 다시 뜬다.

사용:
    python3 check_ulog.py <log>.ulg --type ground     # 지상 30초 (사전 점검)
    python3 check_ulog.py <log>.ulg --type hover      # F1 정지 호버 60초
    python3 check_ulog.py <log>.ulg --type doublet    # F2 자세 여기
    python3 check_ulog.py <log>.ulg --type drag       # F3 직선 왕복
    python3 check_ulog.py <log>.ulg --mass 1.372      # 질량 주면 C_thrust 즉석 추정

필요: pip install pyulog
"""
import argparse
import sys

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    sys.exit("[!] pyulog 가 없습니다.  pip install pyulog")

REQUIRED = [
    'vehicle_thrust_setpoint',
    'vehicle_torque_setpoint',
    'sensor_combined',
    'vehicle_attitude',
    'vehicle_local_position',
]
# GPS 는 PX4 버전마다 이름이 다르다
GPS_ALIASES = ['sensor_gps', 'vehicle_gps_position']
OPTIONAL = ['vehicle_angular_velocity', 'battery_status', 'vehicle_status',
            'esc_status', 'vehicle_imu_status']

OK, BAD, WARN = '  [OK]  ', '  [FAIL]', '  [warn]'


def get(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def rate(d):
    t = d.data['timestamp'] * 1e-6
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0


def fld(d, *cands):
    """필드명이 버전마다 달라 후보 중 존재하는 것을 고른다."""
    for c in cands:
        if c in d.data:
            return np.asarray(d.data[c], dtype=float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ulg')
    ap.add_argument('--type', default='hover',
                    choices=['ground', 'hover', 'doublet', 'drag'])
    ap.add_argument('--mass', type=float, default=None, help='AUW [kg] — 주면 C_thrust 추정')
    args = ap.parse_args()

    try:
        ulog = ULog(args.ulg)
    except Exception as e:
        sys.exit(f"[!] 로그를 열 수 없습니다: {e}")

    dur = (ulog.last_timestamp - ulog.start_timestamp) * 1e-6
    print("=" * 72)
    print(f"{args.ulg}   유형={args.type}   길이 {dur:.1f}s")
    print("=" * 72)

    fails = []

    # ── 1) 필수 메시지 ───────────────────────────────────────
    print("\n[1] 필수 메시지")
    for name in REQUIRED:
        d = get(ulog, name)
        if d is None:
            print(f"{BAD} {name:28s} 없음")
            fails.append(f"{name} 누락")
        else:
            print(f"{OK} {name:28s} {len(d.data['timestamp']):7d} rows  {rate(d):7.1f} Hz")
    gps = None
    for a in GPS_ALIASES:
        gps = get(ulog, a)
        if gps is not None:
            print(f"{OK} {a:28s} {len(gps.data['timestamp']):7d} rows  {rate(gps):7.1f} Hz")
            break
    if gps is None:
        print(f"{BAD} {'/'.join(GPS_ALIASES):28s} 없음")
        fails.append("GPS 메시지 누락")

    print("\n[2] 선택 메시지")
    for name in OPTIONAL:
        d = get(ulog, name)
        if d is None:
            print(f"{WARN} {name:28s} 없음 (필수 아님)")
        else:
            print(f"{OK} {name:28s} {len(d.data['timestamp']):7d} rows  {rate(d):7.1f} Hz")

    sc = get(ulog, 'sensor_combined')
    if sc is not None and rate(sc) < 150:
        print(f"{WARN} sensor_combined {rate(sc):.0f} Hz — High rate 로깅이 꺼져 있을 수 있음 "
              f"(진동 PSD 분석에 200Hz+ 권장)")

    if args.type == 'ground':
        verdict(fails, "지상 점검")
        return

    # ── 3) 유형별 품질 게이트 ────────────────────────────────
    lp = get(ulog, 'vehicle_local_position')
    ts = get(ulog, 'vehicle_thrust_setpoint')
    tq = get(ulog, 'vehicle_torque_setpoint')

    if lp is None or ts is None or tq is None:
        verdict(fails, args.type)
        return

    t_lp = lp.data['timestamp'] * 1e-6
    vx = fld(lp, 'vx'); vy = fld(lp, 'vy'); vz = fld(lp, 'vz')
    z = fld(lp, 'z')
    vh = np.hypot(vx, vy)
    alt = -z if z is not None else None

    t_ts = ts.data['timestamp'] * 1e-6
    thr = fld(ts, 'xyz[2]')
    t_tq = tq.data['timestamp'] * 1e-6
    tqx = fld(tq, 'xyz[0]'); tqy = fld(tq, 'xyz[1]'); tqz = fld(tq, 'xyz[2]')

    print(f"\n[3] 품질 게이트 — {args.type}")

    if args.type == 'hover':
        # 비행 중(고도>2m) & 저속(|vh|<0.3) 인 최장 연속 구간
        good = (vh < 0.3) & (alt > 2.0) if alt is not None else (vh < 0.3)
        seg = longest_run(good, t_lp)
        print(f"       정지 구간(|vh|<0.3 m/s, 고도>2m) 최장 = {seg:.1f}s")
        if seg < 45:
            print(f"{BAD} 45초 미만 — 바람에 밀렸거나 스틱 입력이 있었을 가능성. 다시 뜰 것")
            fails.append("정지 호버 구간 부족")
        else:
            print(f"{OK} 정지 호버 구간 충분")
        if alt is not None:
            print(f"       고도 중앙 {np.median(alt[good]) if good.any() else float('nan'):.2f} m "
                  f"(지면효과 회피 위해 3m 이상 권장)")

        # 즉석 추정치
        m = mask_from(t_ts, t_lp, good)
        if m.any():
            u_h = np.abs(thr[m]).mean()
            print(f"\n       ─ 즉석 추정 ─")
            print(f"       호버 스로틀 u_hover = {u_h:.4f}")
            if args.mass:
                print(f"       C_thrust = m·g/u = {args.mass*9.81/u_h:.3f} N   "
                      f"(sim Iris = 25.58)")
            else:
                print(f"       C_thrust = m·9.81/{u_h:.4f}  ← --mass 주면 계산")
        mq = mask_from(t_tq, t_lp, good)
        if mq.any() and m.any():
            u_h = np.abs(thr[m]).mean()
            tr = [tqx[mq].mean(), tqy[mq].mean(), tqz[mq].mean()]
            print(f"       호버 트림 토크(정규화) = "
                  f"[{tr[0]:+.5f}, {tr[1]:+.5f}, {tr[2]:+.5f}]")
            print(f"       → 결합항 K(정규화) = "
                  f"[{tr[0]/u_h:+.5f}, {tr[1]/u_h:+.5f}, {tr[2]/u_h:+.5f}] per unit thrust")
            print(f"         (sim 은 피치축이 지배적이었다. 한 축이 유독 크면 정상)")
        if sc is not None:
            gx = fld(sc, 'gyro_rad[0]'); gy = fld(sc, 'gyro_rad[1]'); gz_ = fld(sc, 'gyro_rad[2]')
            t_sc = sc.data['timestamp'] * 1e-6
            ms = mask_from(t_sc, t_lp, good)
            if ms.any():
                print(f"       자이로 노이즈 σ = [{gx[ms].std():.4f}, {gy[ms].std():.4f}, "
                      f"{gz_[ms].std():.4f}] rad/s   (sim 실증 0.031)")
        if gps is not None:
            gv = [fld(gps, f'vel_{a}_m_s') for a in ('n', 'e', 'd')]
            if all(v is not None for v in gv):
                t_g = gps.data['timestamp'] * 1e-6
                mg = mask_from(t_g, t_lp, good)
                if mg.any():
                    print(f"       ★ GPS 속도 노이즈 σ = "
                          f"[{gv[0][mg].std():.4f}, {gv[1][mg].std():.4f}, {gv[2][mg].std():.4f}] m/s")
                    print(f"         (정지 호버는 참속도≈0 → GPS 속도가 곧 노이즈. RTK 불필요)")

    elif args.type == 'doublet':
        for nm, v in (('롤 x', tqx), ('피치 y', tqy), ('요 z', tqz)):
            thr_lvl = max(0.05, 0.3 * np.abs(v).max())
            n_exc = count_excursions(v, thr_lvl)
            span = np.abs(v).max()
            ok = n_exc >= 5 and span > 0.08
            print(f"{OK if ok else BAD} {nm}: |cmd|최대 {span:.3f}, "
                  f"여기 {n_exc}회 (임계 {thr_lvl:.3f})")
            if not ok:
                fails.append(f"{nm} 여기 부족")
        # 축 분리: 동시 여기 비율
        big = (np.abs(tqx) > 0.3*np.abs(tqx).max()) & (np.abs(tqy) > 0.3*np.abs(tqy).max())
        frac = big.mean()
        print(f"{OK if frac < 0.15 else WARN} 롤·피치 동시 여기 비율 {frac*100:.1f}% "
              f"(<15% 여야 축 분리가 된다. 섞어 치면 회귀가 붕괴한다)")
        if frac >= 0.15:
            print("         → 축을 나눠서(롤 먼저 8회, 그다음 피치 8회) 다시 뜰 것")

    elif args.type == 'drag':
        fast = vh > 1.5
        seg = longest_run(fast, t_lp)
        print(f"       |vh|>1.5 m/s 최장 연속 = {seg:.1f}s   (등속 구간 3초 이상 필요)")
        if seg < 3.0:
            print(f"{BAD} 등속 구간 부족 — 미션 직선 구간을 늘리거나 순항속도를 올릴 것")
            fails.append("등속 구간 부족")
        else:
            print(f"{OK} 등속 구간 확보")
        # 방향 다양성 (항력 축 분리)
        if fast.any():
            ang = np.arctan2(vy[fast], vx[fast])
            hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi))
            nb = (hist > 0.02 * fast.sum()).sum()
            print(f"{OK if nb >= 3 else WARN} 진행방향 분포 = {nb}/8 구획 "
                  f"(전후·좌우 양쪽이 있어야 drag_x/drag_y 분리)")
            if nb < 3:
                print("         → 전후 왕복만 했다면 좌우 왕복도 추가할 것")
            print(f"       속도 중앙 {np.median(vh[fast]):.2f} m/s, 최대 {vh.max():.2f} m/s")

    verdict(fails, args.type)


def mask_from(t_target, t_ref, good_ref):
    """t_ref 의 good 구간을 t_target 시간축으로 옮긴다."""
    if not good_ref.any():
        return np.zeros(len(t_target), dtype=bool)
    t0, t1 = t_ref[good_ref][0], t_ref[good_ref][-1]
    return (t_target >= t0) & (t_target <= t1)


def longest_run(mask, t):
    """mask 가 True 인 최장 연속 구간의 시간 길이 [s]."""
    best = cur = 0.0
    start = None
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = t[i]
            cur = t[i] - start
            best = max(best, cur)
        else:
            start = None
    return best


def count_excursions(v, thr):
    """|v| 가 thr 를 넘었다 내려오는 횟수."""
    above = np.abs(v) > thr
    return int(np.sum(above[1:] & ~above[:-1]))


def verdict(fails, label):
    print("\n" + "=" * 72)
    if fails:
        print(f"  ✗ FAIL ({label}) — {len(fails)}건: " + ", ".join(fails))
        print("    → 짐 싸기 전에 다시 뜰 것")
    else:
        print(f"  ✓ PASS ({label}) — 사용 가능")
    print("=" * 72)


if __name__ == '__main__':
    main()
