#!/usr/bin/env python3
"""fit_from_ulog.py — 실기 ulog 에서 UKF 모델 계수 추출 (2026-07-31)

사용:
    python3 fit_from_ulog.py <log>.ulg --mass 1.372
    python3 fit_from_ulog.py <log>.ulg --mass 1.372 --fc 5.0

추출 항목
    C_thrust        호버 평형   C = m·g / u_hover
    k_norm (결합항)  호버 트림   k = trim_torque_norm / thrust_norm
    G_i             자세 여기   ω̇_i = G_i · cmd_torque_i     (= C_torque_i / I_i)
    drag            등속 구간   정상상태 힘 평형
    센서 노이즈      정지 호버   자이로 σ, GPS 속도 σ

★ 핵심: UKF 모델은 C_torque 와 I 를 개별로 쓰지 않는다.
   ω̇ = τ/I = (C_torque·cmd)/I = cmd × G      →  G 만 재면 된다.
   그래서 암 길이·프로펠러·모터 사양이 필요 없다.

★ 결합항이 ω̇ 에 주는 영향도 정규화 형태로 닫힌다:
   ω̇_coupling = K·u_thrust/I = (trim_norm/thrust_norm) × G = k_norm × G
   sim 의 N·m/N 형태로 바꿀 때:  K_sim = k_norm × C_torque_sim / C_thrust_sim
"""
import argparse
import sys

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    sys.exit("[!] pyulog 가 없습니다.  pip install pyulog")

try:
    from scipy.signal import butter, filtfilt
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

G0 = 9.80665
AX = ('roll(x)', 'pitch(y)', 'yaw(z)')


# ──────────────────────────────────────────────────────────────
#  ulog 접근 헬퍼
# ──────────────────────────────────────────────────────────────
def topic(ulog, *names):
    for n in names:
        for d in ulog.data_list:
            if d.name == n:
                return d
    return None


def t_of(d):
    return np.asarray(d.data['timestamp'], dtype=float) * 1e-6


def col(d, *cands):
    for c in cands:
        if c in d.data:
            return np.asarray(d.data[c], dtype=float)
    return None


def vec3(d, base):
    """xyz[0..2] 또는 base_x/y/z 형태를 (N,3) 으로."""
    out = []
    for i, s in enumerate('xyz'):
        v = col(d, f'{base}[{i}]', f'{base}_{s}', s)
        if v is None:
            return None
        out.append(v)
    return np.column_stack(out)


def resample(t_src, y_src, t_dst):
    """t_dst 시간축으로 선형보간 (ZOH 아님 — 명령은 계단이지만 저역통과 후 쓰므로 무해)."""
    if y_src.ndim == 1:
        return np.interp(t_dst, t_src, y_src)
    return np.column_stack([np.interp(t_dst, t_src, y_src[:, j])
                            for j in range(y_src.shape[1])])


def lowpass(y, dt, fc):
    """영위상 저역통과. 명령·응답 **양쪽에 동일 적용**해야 위상차가 안 생긴다."""
    if not HAVE_SCIPY or fc <= 0:
        return y
    fs = 1.0 / dt
    if fc >= 0.45 * fs:
        return y
    b, a = butter(2, fc / (0.5 * fs))
    axis = 0 if y.ndim > 1 else -1
    return filtfilt(b, a, y, axis=axis)


def quat_to_R(q):
    """(N,4) [w,x,y,z] → (N,3,3) 바디→NED 회전행렬. PX4 쿼터니언은 이미 NED/FRD."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.empty((len(w), 3, 3))
    R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-z*w); R[:, 0, 2] = 2*(x*z+y*w)
    R[:, 1, 0] = 2*(x*y+z*w);   R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-x*w)
    R[:, 2, 0] = 2*(x*z-y*w);   R[:, 2, 1] = 2*(y*z+x*w);   R[:, 2, 2] = 1-2*(x*x+y*y)
    return R


def runs(mask, t, min_len):
    """mask 가 True 인 연속 구간 [(i0,i1), ...] 중 길이 min_len[s] 이상."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if t[i-1] - t[s] >= min_len:
                out.append((s, i))
            s = None
    if s is not None and t[-1] - t[s] >= min_len:
        out.append((s, len(mask)))
    return out


def lstsq_report(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    ss = ((y - y.mean())**2).sum()
    r2 = 1 - (resid**2).sum()/ss if ss > 0 else float('nan')
    return b, r2


# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ulg')
    ap.add_argument('--mass', type=float, required=True, help='AUW [kg] — 날리는 그대로')
    ap.add_argument('--fc', type=float, default=5.0, help='영위상 저역통과 [Hz] (0=끔)')
    ap.add_argument('--hover-vmax', type=float, default=0.35, help='정지 판정 수평속도 [m/s]')
    ap.add_argument('--hover-altmin', type=float, default=1.5, help='정지 판정 최소고도 [m]')
    ap.add_argument('--cruise-vmin', type=float, default=1.5, help='등속 판정 최소속도 [m/s]')
    args = ap.parse_args()

    ulog = ULog(args.ulg)
    m = args.mass
    print("=" * 78)
    print(f"실기 계수 추출 — {args.ulg}   AUW = {m} kg   저역통과 fc = {args.fc} Hz")
    print("=" * 78)
    if not HAVE_SCIPY:
        print("  ⚠ scipy 없음 → 저역통과 생략. 결과 신뢰도 낮음")

    d_thr = topic(ulog, 'vehicle_thrust_setpoint')
    d_tq = topic(ulog, 'vehicle_torque_setpoint')
    d_lp = topic(ulog, 'vehicle_local_position')
    d_att = topic(ulog, 'vehicle_attitude')
    d_sc = topic(ulog, 'sensor_combined')
    d_gps = topic(ulog, 'sensor_gps', 'vehicle_gps_position')
    for nm, d in (('thrust_setpoint', d_thr), ('torque_setpoint', d_tq),
                  ('local_position', d_lp), ('attitude', d_att), ('sensor_combined', d_sc)):
        if d is None:
            sys.exit(f"[!] 필수 메시지 없음: {nm}")

    t_lp, t_thr, t_tq, t_sc = t_of(d_lp), t_of(d_thr), t_of(d_tq), t_of(d_sc)
    thr = vec3(d_thr, 'xyz')
    tq = vec3(d_tq, 'xyz')
    gyro = vec3(d_sc, 'gyro_rad')
    vx, vy, vz = col(d_lp, 'vx'), col(d_lp, 'vy'), col(d_lp, 'vz')
    zz = col(d_lp, 'z')
    alt = -zz
    vh = np.hypot(vx, vy)

    # ── 1) 정지 호버 구간 ─────────────────────────────────
    print("\n" + "-" * 78)
    print("[1] 정지 호버 — C_thrust, 결합항, 센서 노이즈")
    print("-" * 78)
    hov = (vh < args.hover_vmax) & (alt > args.hover_altmin)
    segs = runs(hov, t_lp, 5.0)
    if not segs:
        print(f"  구간 없음 (|vh|<{args.hover_vmax}, 고도>{args.hover_altmin}m, 5초 이상)")
        print(f"  참고: 이 로그의 고도 최대 {alt.max():.2f} m, |vh| 중앙 {np.median(vh):.2f} m/s")
        C_thrust = None
        k_norm = None
    else:
        tot = sum(t_lp[b-1]-t_lp[a] for a, b in segs)
        print(f"  구간 {len(segs)}개, 총 {tot:.1f}s "
              f"(최장 {max(t_lp[b-1]-t_lp[a] for a,b in segs):.1f}s)")
        t0, t1 = t_lp[segs[0][0]], t_lp[segs[-1][1]-1]
        mth = (t_thr >= t0) & (t_thr <= t1)
        mtq = (t_tq >= t0) & (t_tq <= t1)
        u_h = np.abs(thr[mth, 2]).mean()
        C_thrust = m * G0 / u_h
        print(f"\n  호버 스로틀 u_hover = {u_h:.4f}")
        print(f"  ★ C_thrust = m·g/u = {C_thrust:.3f} N          (sim Iris = 25.58)")
        trim = tq[mtq].mean(axis=0)
        k_norm = trim / u_h
        print(f"\n  호버 트림 토크(정규화) = [{trim[0]:+.5f}, {trim[1]:+.5f}, {trim[2]:+.5f}]")
        print(f"  ★ 결합항 k_norm       = [{k_norm[0]:+.5f}, {k_norm[1]:+.5f}, {k_norm[2]:+.5f}]"
              f"   (= trim / u_hover)")
        print(f"     → ω̇ 기여 = k_norm × G  (G 는 아래 [2] 에서)")
        print(f"     → sim 형식 K = k_norm × C_torque_sim / C_thrust_sim")
        msc = (t_sc >= t0) & (t_sc <= t1)
        if msc.sum() > 10:
            s = gyro[msc].std(axis=0)
            print(f"\n  자이로 노이즈 σ = [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}] rad/s"
                  f"   (sim 실증 0.031)")
        if d_gps is not None:
            t_g = t_of(d_gps)
            gv = [col(d_gps, f'vel_{a}_m_s') for a in ('n', 'e', 'd')]
            if all(v is not None for v in gv):
                mg = (t_g >= t0) & (t_g <= t1)
                if mg.sum() > 5:
                    s = [v[mg].std() for v in gv]
                    print(f"  GPS 속도 노이즈 σ = [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}] m/s"
                          f"   (표본 {mg.sum()}개)")
                    if mg.sum() < 30:
                        print(f"    ⚠ 표본 부족 — sensor_gps 로깅 레이트를 올리면 정확해짐")

    # ── 2) G = C_torque / I ───────────────────────────────
    print("\n" + "-" * 78)
    print("[2] 자세 여기 — G_i = C_torque_i / I_i   (ω̇_i = G_i · cmd_i)")
    print("-" * 78)
    dt = np.median(np.diff(t_sc))
    w = lowpass(gyro, dt, args.fc)
    wdot = np.gradient(w, dt, axis=0)
    cmd = resample(t_tq, tq, t_sc)
    cmd = lowpass(cmd, dt, args.fc)
    uz = resample(t_thr, np.abs(thr[:, 2]), t_sc)
    uz = lowpass(uz, dt, args.fc)

    # 여기가 있는 구간만 (명령이 충분히 흔들리는 곳)
    exc = np.abs(cmd).max(axis=1) > max(0.03, 0.15 * np.abs(cmd).max())
    n_exc = exc.sum()
    print(f"  여기 샘플 {n_exc}/{len(cmd)}  (dt={dt*1000:.2f}ms, {1/dt:.0f}Hz)")
    if n_exc < 200:
        print("  ⚠ 여기 샘플 부족 — doublet 비행이 아니면 신뢰할 수 없다")

    # 교차항 + 추력항 포함 (축간 누설·결합항 흡수)
    X = np.column_stack([cmd[exc, 0], cmd[exc, 1], cmd[exc, 2], uz[exc]])
    print(f"  설계행렬 조건수 = {np.linalg.cond(X):.1f}   (>30 이면 축 분리 불량)")
    Gd, Gc = [], []
    print(f"\n  {'축':>10s} {'G(대각만)':>11s} {'G(교차포함)':>12s} {'R²(대각)':>9s} {'R²(교차)':>9s}")
    for j in range(3):
        y = wdot[exc, j]
        bd, r2d = lstsq_report(np.column_stack([cmd[exc, j], uz[exc]]), y)
        bc, r2c = lstsq_report(X, y)
        Gd.append(bd[0]); Gc.append(bc[j])
        print(f"  {AX[j]:>10s} {bd[0]:11.2f} {bc[j]:12.2f} {r2d:9.3f} {r2c:9.3f}")
    print(f"\n  ※ R² 가 0.5 미만이면 여기가 부족하거나 폐루프 편향이다.")
    print(f"     doublet(개루프 여기) 비행이 아니면 이 값을 쓰지 말 것.")

    if C_thrust is not None and k_norm is not None:
        print(f"\n  결합항이 ω̇ 에 주는 기여 = k_norm × G:")
        for j in range(3):
            print(f"    {AX[j]:>10s}  {k_norm[j]:+.5f} × {Gc[j]:7.2f} = "
                  f"{k_norm[j]*Gc[j]:+7.3f} rad/s²")

    # ── 3) drag ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print("[3] 등속 구간 — drag")
    print("-" * 78)
    if C_thrust is None:
        print("  C_thrust 미확정(호버 구간 없음) → drag 산출 불가")
    elif d_att is None:
        print("  vehicle_attitude 없음 → drag 산출 불가")
    else:
        fast = vh > args.cruise_vmin
        segs2 = runs(fast, t_lp, 2.0)
        if not segs2:
            print(f"  등속 구간 없음 (|vh|>{args.cruise_vmin} m/s, 2초 이상)")
            print(f"  참고: |vh| 최대 {vh.max():.2f} m/s")
        else:
            tot = sum(t_lp[b-1]-t_lp[a] for a, b in segs2)
            print(f"  구간 {len(segs2)}개, 총 {tot:.1f}s")
            t_a = t_of(d_att)
            q = vec3(d_att, 'q')
            if q is None:
                q4 = np.column_stack([col(d_att, f'q[{i}]') for i in range(4)])
            else:
                q4 = np.column_stack([col(d_att, f'q[{i}]') for i in range(4)])
            idx = np.concatenate([np.arange(a, b) for a, b in segs2])
            tt = t_lp[idx]
            R = quat_to_R(resample(t_a, q4, tt))
            T = C_thrust * resample(t_thr, np.abs(thr[:, 2]), tt)      # [N]
            v_ned = np.column_stack([vx[idx], vy[idx], vz[idx]])
            # 정상상태:  0 = R·F_thrust_body + F_drag_ned + m·g_ned
            F_thr_ned = np.einsum('nij,nj->ni', R, np.column_stack([
                np.zeros_like(T), np.zeros_like(T), -T]))
            F_drag_ned = -F_thr_ned - np.column_stack([
                np.zeros_like(T), np.zeros_like(T), np.full_like(T, m*G0)])
            F_drag_b = np.einsum('nji,nj->ni', R, F_drag_ned)          # Rᵀ·F
            v_b = np.einsum('nji,nj->ni', R, v_ned)
            print(f"\n  {'축':>8s} {'drag':>9s} {'R²':>8s} {'|v_body| 중앙':>14s} {'n':>7s}")
            for j, nm in enumerate(('x(전후)', 'y(좌우)', 'z(상하)')):
                sel = np.abs(v_b[:, j]) > 0.8
                if sel.sum() < 30:
                    print(f"  {nm:>8s} {'—':>9s} {'—':>8s} {'표본부족':>14s} {sel.sum():7d}")
                    continue
                b, r2 = lstsq_report(v_b[sel, j:j+1], -F_drag_b[sel, j])
                print(f"  {nm:>8s} {b[0]:9.3f} {r2:8.3f} "
                      f"{np.median(np.abs(v_b[sel,j])):14.2f} {sel.sum():7d}")
            print(f"\n  ※ sim Pegasus 설정값 = [0.50, 0.30, 0.00]")
            print(f"     전후·좌우 양쪽 구간이 있어야 x/y 분리가 된다.")

    # ── 4) 요약 ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("요약 — sim 정합에 쓸 값")
    print("=" * 78)
    print(f"  C_thrust  = {f'{C_thrust:.3f} N' if C_thrust else '미확정 (호버 구간 필요)'}")
    print(f"  k_norm    = {np.round(k_norm,5).tolist() if k_norm is not None else '미확정'}")
    print(f"  G         = {[round(v,2) for v in Gc]}   (R² 확인 필수)")
    print(f"\n  다음 단계: I_sim = C_torque_iris / G_real  로 sim 관성 스케일")
    print(f"            (로터 기하는 그대로 두고 관성만 바꾼다)")
    print("=" * 78)


if __name__ == '__main__':
    main()
