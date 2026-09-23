#!/usr/bin/env python3
"""nis_from_ulog.py — 실기 ulog 로 sim 과 **동일한** shadow UKF 를 돌려 NIS 기준선을 뽑는다.

왜
  sim 에서 뽑은 밴드·탐지 임계가 실기로 전이되려면, 평시 NIS 수준이 양쪽에서 같아야 한다.
  그러려면 실기 로그에 **같은 필터(같은 Q·R·같은 압축)** 를 물려서 재봐야 한다.
  Q·R 은 센서에 맞춘 값이 아니라 탐지 설계값(고집)이므로 절대 바꾸지 않는다.

sim 과 맞춘 것 (online_rl_main._run_ukf_step 과 1:1)
  · z_9d = [GPS pos NED(3), GPS vel NED(3), gyro(3)]
  · u    = to_physical_u(thrust_sp.xyz, torque_sp.xyz, calibration.json)
  · 50 Hz predict + gyro update,  GPS 는 새 샘플이 온 스텝만 fresh (멀티레이트 게이트)
  · NIS  = compute_nis_scaled(res[3:6]|res[6:9]|res[0:3], Pzz 대응블록, nz=3)

실기 고유 처리
  · GPS 는 **원시 vehicle_gps_position** 을 쓴다. vehicle_local_position(EKF 출력)을 쓰면
    이미 평활된 값이라 NIS 가 비현실적으로 낮게 나온다.  lat/lon → 로컬 NED 는
    vehicle_local_position 의 ref_lat/ref_lon/ref_alt 로 등거리 근사 투영.
  · 자세 쿼터니언은 PX4 가 이미 NED/FRD 라 표준 ZYX 로 바로 오일러각.
    (online_rl_main 의 ENU→NED 보정은 Isaac GT 전용이므로 여기선 쓰지 않는다.)

사용
    ~/isaacsim/python.sh px4_field/nis_from_ulog.py <ulg> [--offboard-only] [--label 이름]
    ~/isaacsim/python.sh px4_field/nis_from_ulog.py px4_field/ulog_0812/*.ulg --offboard-only
"""
import argparse
import os
import sys

import numpy as np
try:
    from pyulog import ULog
except ImportError:            # ★09-22: f13 가 quat_to_euler_ned 만 쓰므로 pyulog 없이도 임포트되게(ulog 판독 때만 필요)
    ULog = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration, to_physical_u  # noqa: E402

R_EARTH = 6371000.0


def get(u, name):
    for d in u.data_list:
        if d.name == name:
            return d
    return None


def quat_to_euler_ned(q):
    """PX4 쿼터니언(NED/FRD) → (roll, pitch, yaw). 보정 불필요."""
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.column_stack([roll, pitch, yaw])


def run_one(path, calib, offboard_only=True, label=None):
    if ULog is None:
        raise SystemExit('pyulog 가 없습니다: pip3 install pyulog')
    u = ULog(path, ['vehicle_gps_position', 'vehicle_local_position', 'vehicle_attitude',
                    'sensor_combined', 'vehicle_torque_setpoint', 'vehicle_thrust_setpoint',
                    'vehicle_status', 'vehicle_land_detected'])
    gps, lp, att = get(u, 'vehicle_gps_position'), get(u, 'vehicle_local_position'), get(u, 'vehicle_attitude')
    sc, tq, th = get(u, 'sensor_combined'), get(u, 'vehicle_torque_setpoint'), get(u, 'vehicle_thrust_setpoint')
    vs, ld = get(u, 'vehicle_status'), get(u, 'vehicle_land_detected')
    if any(v is None for v in (gps, lp, att, sc, tq, th, ld)):
        return None

    # ── 구간 선택 ──
    t_lp = lp.data['timestamp'] * 1e-6
    air = np.interp(t_lp, ld.data['timestamp'] * 1e-6, 1.0 - ld.data['landed'].astype(float)) > 0.5
    sel = air
    if offboard_only and vs is not None:
        ns = np.interp(t_lp, vs.data['timestamp'] * 1e-6, vs.data['nav_state'].astype(float))
        sel = air & (np.abs(ns - 14) < 0.5)
    if sel.sum() < 50:
        return None
    t0, t1 = t_lp[sel].min(), t_lp[sel].max()

    # ── 기준점 (로컬 NED 원점) ──
    ref_lat = np.nanmedian(lp.data['ref_lat'][sel]) if 'ref_lat' in lp.data else np.nan
    ref_lon = np.nanmedian(lp.data['ref_lon'][sel]) if 'ref_lon' in lp.data else np.nan
    ref_alt = np.nanmedian(lp.data['ref_alt'][sel]) if 'ref_alt' in lp.data else 0.0
    if not np.isfinite(ref_lat):
        return None

    t_g = gps.data['timestamp'] * 1e-6
    # PX4 버전에 따라 필드명이 다르다: 구 lat/lon(1e-7 정수) vs 신 latitude_deg(실수)
    if 'latitude_deg' in gps.data:
        lat, lon = gps.data['latitude_deg'], gps.data['longitude_deg']
        alt = gps.data['altitude_msl_m']
    else:
        lat = gps.data['lat'] * 1e-7
        lon = gps.data['lon'] * 1e-7
        alt = gps.data['alt'] * 1e-3
    gx = np.radians(lat - ref_lat) * R_EARTH                                  # N
    gy = np.radians(lon - ref_lon) * R_EARTH * np.cos(np.radians(ref_lat))    # E
    gz = -(alt - ref_alt)                                                     # D
    gvn, gve, gvd = (gps.data['vel_n_m_s'], gps.data['vel_e_m_s'], gps.data['vel_d_m_s'])

    t_a = att.data['timestamp'] * 1e-6
    eul = quat_to_euler_ned(np.column_stack([att.data[f'q[{i}]'] for i in range(4)]))
    t_s = sc.data['timestamp'] * 1e-6
    gyro = np.column_stack([sc.data[f'gyro_rad[{i}]'] for i in range(3)])
    t_t = tq.data['timestamp'] * 1e-6
    tq_xyz = np.column_stack([tq.data[f'xyz[{i}]'] for i in range(3)])
    t_h = th.data['timestamp'] * 1e-6
    th_xyz = np.column_stack([th.data[f'xyz[{i}]'] for i in range(3)])

    # ── 50 Hz 그리드 ──
    dt = 0.02
    grid = np.arange(t0, t1, dt)
    if grid.size < 50:
        return None
    ip = lambda tx, v: np.interp(grid, tx, v)
    # ★ GPS 는 ZOH(마지막 수신값 유지) — sim 의 obs_gps_* 와 같은 취급.
    #   선형보간하면 stale 스텝에 없는 정보가 들어가 잔차가 인위적으로 줄어든다.
    ig = np.clip(np.searchsorted(t_g, grid, side='right') - 1, 0, len(t_g) - 1)
    Z = np.column_stack([gx[ig], gy[ig], gz[ig], gvn[ig], gve[ig], gvd[ig],
                         ip(t_s, gyro[:, 0]), ip(t_s, gyro[:, 1]), ip(t_s, gyro[:, 2])])
    U_th = np.column_stack([ip(t_h, th_xyz[:, i]) for i in range(3)])
    U_tq = np.column_stack([ip(t_t, tq_xyz[:, i]) for i in range(3)])
    U = to_physical_u(U_th, U_tq, calib)
    E0 = np.array([ip(t_a, eul[:, i])[0] for i in range(3)])

    # GPS 신선도: 원시 GPS 샘플이 새로 들어온 스텝만 True
    fresh = np.concatenate([[True], np.diff(ig) > 0])

    # ── 필터 구동 (sim 과 동일 설정) ──
    ukf = DynamicsUKF(dt=dt, calib=calib)
    ukf.x[0:3] = Z[0, 0:3]
    ukf.x[3:6] = E0
    ukf.x[6:9] = Z[0, 3:6]
    ukf.x[9:12] = Z[0, 6:9]
    out = {k: [] for k in ('v_raw', 'v', 'g_raw', 'g', 'p_raw', 'p')}
    for k in range(grid.size):
        res, Pzz = ukf.step(Z[k], U[k], gps_fresh=bool(fresh[k]))
        vr, vs_ = compute_nis_scaled(res[3:6], Pzz[3:6, 3:6], 3.0)
        gr, gs_ = compute_nis_scaled(res[6:9], Pzz[6:9, 6:9], 3.0)
        pr, ps_ = compute_nis_scaled(res[0:3], Pzz[0:3, 0:3], 3.0)
        for key, val in zip(('v_raw', 'v', 'g_raw', 'g', 'p_raw', 'p'),
                            (vr, vs_, gr, gs_, pr, ps_)):
            out[key].append(val)
    # ★ 정책이 보는 것은 GPS 가 새로 온 스텝뿐(gps_updated 게이트, 10Hz).
    #   stale 스텝은 R 이 1e6 배 부풀어 NIS≈0 이라 섞으면 통계가 왜곡된다.
    f = np.asarray(fresh, dtype=bool)
    return {'label': label or os.path.basename(path), 'n': int(f.sum()), 'dur': grid.size * dt,
            **{k: np.asarray(v)[f] for k, v in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ulg', nargs='+')
    ap.add_argument('--offboard-only', action='store_true', help='오프보드 구간만 (기본: 공중 전체)')
    ap.add_argument('--label', default=None)
    ap.add_argument('--calib', default=None)
    a = ap.parse_args()
    calib = load_calibration(a.calib) if a.calib else load_calibration()

    print(f"{'비행':18s} {'초':>5s} | {'vel NIS  p50/p95':>20s} | {'gyro NIS p50/p95':>20s} "
          f"| {'압축 vel':>12s} {'압축 gyr':>12s}")
    print('-' * 100)
    rows = []
    for p in a.ulg:
        try:
            r = run_one(p, calib, a.offboard_only, a.label)
        except Exception as e:                                    # noqa: BLE001
            print(f'{os.path.basename(p):18s}  실패: {e}')
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"{r['label']:18s} {r['dur']:5.0f} | "
              f"{np.median(r['v_raw']):9.2f}/{np.percentile(r['v_raw'],95):<9.2f} | "
              f"{np.median(r['g_raw']):9.2f}/{np.percentile(r['g_raw'],95):<9.2f} | "
              f"{np.median(r['v']):5.3f}/{np.percentile(r['v'],95):<5.3f} "
              f"{np.median(r['g']):5.3f}/{np.percentile(r['g'],95):<5.3f}")
    if rows:
        vr = np.concatenate([r['v_raw'] for r in rows])
        gr = np.concatenate([r['g_raw'] for r in rows])
        print('-' * 100)
        print(f"{'전체':18s} {sum(r['dur'] for r in rows):5.0f} | "
              f"{np.median(vr):9.2f}/{np.percentile(vr,95):<9.2f} | "
              f"{np.median(gr):9.2f}/{np.percentile(gr,95):<9.2f}")


if __name__ == '__main__':
    main()
