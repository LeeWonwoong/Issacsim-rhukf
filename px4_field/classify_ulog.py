#!/usr/bin/env python3
"""classify_ulog.py — 오늘 받은 ulog 무더기를 F1~F9 로 자동 라벨링 (2026-08-10)

문제
  하루에 F1~F9 를 섞어서, 어떤 건 한 번 어떤 건 두세 번 날렸다. fetch_ulog --list
  는 id·크기·시각만 보여줘 **어느 로그가 어느 비행인지** 알 수 없다.

푸는 법 (두 축)
  ① 시그니처로 확정되는 것
       F2 오토튠 : autotune_attitude_control_status 메시지가 있다 → 100% 확정
       F9 공격   : manual_control_setpoint.aux1(=VRA) 이 비행 중 0.9 초과
                   또는 actuator_attack.active → 확정
       모드 분리 : nav_state 로 Position(F1/F3/F4) vs Offboard(F5~F8) 를 가른다
  ② 사람 눈이 제일 정확한 것 — 궤적 모양
       F5 waypoint / F6 circle / F7 figure8 / F8 aggressive 는
       XY 궤적을 보면 사람이 0.5초 만에 안다. --plot 이 격자 그림 하나로 뽑는다.

  자동 추정(guess)은 참고용이다. 패턴 4종은 --plot 그림으로 최종 확인할 것.

사용
    ~/isaacsim/python.sh px4_field/classify_ulog.py px4_field/ulog/            # 폴더 전체
    ~/isaacsim/python.sh px4_field/classify_ulog.py px4_field/ulog/*.ulg       # 글롭
    ~/isaacsim/python.sh px4_field/classify_ulog.py px4_field/ulog/ --plot     # +궤적 격자 PNG
    ~/isaacsim/python.sh px4_field/classify_ulog.py log59.ulg                  # 한 개

필요: pyulog (표), matplotlib (--plot)
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    sys.exit("[!] pyulog 가 없습니다.  pip install pyulog")

# PX4 nav_state
NAV_MANUAL, NAV_ALTCTL, NAV_POSCTL, NAV_OFFBOARD = 0, 1, 2, 14


def get(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def fld(d, *cands):
    for c in cands:
        if d is not None and c in d.data:
            return np.asarray(d.data[c], dtype=float)
    return None


def rate(d):
    t = d.data['timestamp'] * 1e-6
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0


def dom_freq(sig, t):
    """detrend 후 지배 주파수 [Hz]. 표본이 적으면 0."""
    if len(sig) < 16:
        return 0.0
    s = sig - np.polyval(np.polyfit(t, sig, 1), t)      # 선형 추세 제거
    n = len(s)
    dt = (t[-1] - t[0]) / (n - 1)
    if dt <= 0:
        return 0.0
    f = np.fft.rfftfreq(n, dt)
    p = np.abs(np.fft.rfft(s * np.hanning(n)))
    p[0] = 0.0                                            # DC 제거
    k = int(np.argmax(p))
    return float(f[k]) if k < len(f) else 0.0


def classify_shape(x, y, z, vh):
    """오프보드 XY 궤적을 patttern 4종 중 하나로 추정. (라벨, 근거문자열)."""
    cx, cy = x - np.mean(x), y - np.mean(y)
    span = float(max(cx.max() - cx.min(), cy.max() - cy.min()))
    r = np.hypot(cx, cy)
    vpk = float(np.nanmax(vh)) if len(vh) else 0.0
    vrange = float(np.nanmax(z) - np.nanmin(z)) if z is not None and len(z) else 0.0

    if span < 1.0:                                        # 거의 안 움직임
        return 'hover', f'XY span {span:.1f}m (제자리)'

    # 원 판정: 반경이 거의 일정
    circ = 1.0 - (np.std(r) / (np.mean(r) + 1e-9))
    fx = dom_freq(cx, np.arange(len(cx)))
    fy = dom_freq(cy, np.arange(len(cy)))
    ratio = fy / fx if fx > 1e-6 else 0.0

    ev = f'span {span:.1f}m, 원형도 {circ:.2f}, fy/fx {ratio:.2f}, v_peak {vpk:.1f}, Δalt {vrange:.1f}'

    # figure8: y 가 x 의 약 2배 주파수 (또는 그 역)
    if 1.6 <= ratio <= 2.5 or 0.4 <= ratio <= 0.62:
        return 'figure8', ev
    if circ > 0.75:
        return 'circle', ev
    # aggressive: 상하 기동이 크거나 속도가 빠름
    if vrange > 1.5 or vpk > 1.7:
        return 'aggressive', ev
    return 'waypoint', ev                                 # 폴리곤/직선 구간형


def count_doublets(v):
    if v is None or len(v) < 3:
        return 0
    thr = max(0.05, 0.3 * np.abs(v).max())
    above = np.abs(v) > thr
    return int(np.sum(above[1:] & ~above[:-1]))


def analyze(path):
    """로그 하나 → dict(라벨·근거·특징)."""
    try:
        ulog = ULog(path)
    except Exception as e:                                # noqa: BLE001
        return {'file': os.path.basename(path), 'guess': f'열기실패({e})', 'dur': 0}

    dur = (ulog.last_timestamp - ulog.start_timestamp) * 1e-6
    r = {'file': os.path.basename(path), 'dur': dur, 'x': None, 'y': None}

    # ATK 파라미터 (로그에 초기 파라미터 덤프가 들어있다) — 공격 판정의 근거
    ip = getattr(ulog, 'initial_parameters', {}) or {}
    r['atk_en'] = ip.get('ATK_EN')
    r['atk_tqmax'] = ip.get('ATK_TQ_MAX')
    r['atk_thmax'] = ip.get('ATK_TH_MAX')
    r['atk_axis'] = ip.get('ATK_TQ_AXIS')

    vs = get(ulog, 'vehicle_status')
    armed_ever = False
    ns = t_vs = None
    if vs is not None:
        a = fld(vs, 'arming_state')
        armed_ever = bool(a is not None and (a >= 2).any())
        ns = fld(vs, 'nav_state')
        t_vs = vs.data['timestamp'] * 1e-6

    lp = get(ulog, 'vehicle_local_position')
    flight = None
    if lp is not None:
        t_lp = lp.data['timestamp'] * 1e-6
        x, y, z = fld(lp, 'x'), fld(lp, 'y'), fld(lp, 'z')
        vx, vy = fld(lp, 'vx'), fld(lp, 'vy')
        alt = -z if z is not None else None
        vh = np.hypot(vx, vy) if vx is not None else np.zeros_like(t_lp)
        flight = (alt > 0.5) if alt is not None else np.ones(len(t_lp), bool)
        r.update(x=x, y=y, z=alt, vh=vh, t=t_lp, flight=flight)

    # 모드 비율 (비행 구간 기준) + 오프보드 마스크(궤적 판정을 이 구간으로 한정)
    offb_frac = pos_frac = 0.0
    offb_mask = None
    if ns is not None and lp is not None:
        ns_lp = np.round(np.interp(t_lp, t_vs, ns))
        offb_mask = ns_lp == NAV_OFFBOARD
        if flight is not None and flight.any():
            fl = flight
            offb_frac = float(np.mean(offb_mask[fl]))
            pos_frac = float(np.mean(np.isin(ns_lp[fl], [NAV_POSCTL, NAV_ALTCTL])))
    r['offb_frac'] = offb_frac
    r['pos_frac'] = pos_frac
    r['offb_mask'] = offb_mask

    # 공격(F9): 실제 주입 = ATK_EN==1 ∧ (VRA=aux1>0.9 ∨ actuator_attack.active)
    #   ATK_EN=0 이면 VRA 를 올려도 주입되지 않는다 → 공격 아님(그냥 노브가 올라간 것).
    mc = get(ulog, 'manual_control_setpoint')
    aux1 = fld(mc, 'aux1', 'aux[0]')
    aux1max = float(aux1.max()) if (aux1 is not None and len(aux1)) else None
    vra_up = bool(armed_ever and aux1max is not None and aux1max > 0.9)
    aa = get(ulog, 'actuator_attack')
    aa_active = False
    if aa is not None:
        act = fld(aa, 'active')
        aa_active = bool(act is not None and (act > 0.5).any())
    r['vra_up'], r['aux1max'] = vra_up, aux1max
    en = r['atk_en']
    if en is None:                                       # 구펌웨어 등 — 파라미터 없음
        attack = vra_up or aa_active
        atk_ev = f'VRAmax {aux1max:.2f} (ATK_EN미상)' if vra_up else ''
    elif float(en) >= 0.5:                               # ATK_EN=1
        attack = vra_up or aa_active
        atk_ev = f'ATK_EN=1 ∧ VRA {aux1max:.2f}' if attack else ''
    else:                                                # ATK_EN=0 → 주입 안 됨
        attack = False
        atk_ev = f'VRA {aux1max:.2f}지만 ATK_EN=0(미주입)' if vra_up else ''

    has_autotune = get(ulog, 'autotune_attitude_control_status') is not None

    tq = get(ulog, 'vehicle_torque_setpoint')
    dbx = count_doublets(fld(tq, 'xyz[0]'))
    dby = count_doublets(fld(tq, 'xyz[1]'))

    # 비행 특징
    span = vpk = vrange = alt_max = 0.0
    flew = bool(flight is not None and flight.any() and flight.sum() > 20)
    if flew:
        xf, yf = r['x'][flight], r['y'][flight]
        span = float(max(xf.max() - xf.min(), yf.max() - yf.min()))
        vpk = float(np.nanmax(r['vh'][flight]))
        if r['z'] is not None:
            vrange = float(np.nanmax(r['z'][flight]) - np.nanmin(r['z'][flight]))
            alt_max = float(np.nanmax(r['z'][flight]))
    # 고도 발산 징후: 비정상적으로 높은 최고고도 or EKF 리셋 다발
    nres = 0
    for c in ('xy_reset_counter', 'z_reset_counter', 'heading_reset_counter'):
        v = fld(lp, c)
        if v is not None:
            nres += int((np.diff(v) != 0).sum())
    r['alt_max'], r['ekf_reset'] = alt_max, nres
    r['diverged'] = bool(flew and (alt_max > 8.0 or nres >= 3))

    # ── 결정 트리 ────────────────────────────────────────────
    if not armed_ever:
        r['guess'], r['why'] = 'ground/미시동', '시동 기록 없음'
    elif not flew or dur < 12:
        r['guess'], r['why'] = 'aborted/짧음', f'비행표본 부족 (dur {dur:.0f}s)'
    elif has_autotune:
        r['guess'], r['why'] = 'F2 autotune', 'autotune_attitude_control_status 존재'
    elif attack:
        r['guess'], r['why'] = 'F9 attack', atk_ev
    elif offb_frac > 0.3:
        # 궤적 판정은 오프보드 구간만 (이착륙·전이 구간이 원형도를 오염시킨다)
        seg = (flight & offb_mask) if offb_mask is not None else flight
        if seg.sum() < 20:
            seg = flight
        shape, ev = classify_shape(r['x'][seg], r['y'][seg],
                                   r['z'][seg] if r['z'] is not None else None,
                                   r['vh'][seg])
        fmap = {'circle': 'F6 circle', 'figure8': 'F7 figure8',
                'waypoint': 'F5 waypoint', 'aggressive': 'F8 aggressive',
                'hover': 'offboard hover(움직임적음)'}
        r['guess'], r['why'] = fmap[shape], f'offboard {offb_frac*100:.0f}%; {ev}'
    else:
        # Position/manual: F1 vertical / F3 doublet / F4 drag
        osc = max(dbx, dby) >= 100                        # 크로싱 수백회 = 진동, doublet 아님
        if osc:
            r['guess'] = 'position: 진동/불안정?'
            r['why'] = f'토크크로싱 롤{dbx}피치{dby} (수백회=진동, 깨끗한 doublet 아님)'
        elif 8 <= dbx + dby and span < 3.0:
            r['guess'], r['why'] = 'F3 doublet', f'토크 여기 롤{dbx}+피치{dby}회'
        elif vrange > 1.2 and span < 2.5:
            r['guess'], r['why'] = 'F1 hover/vertical', f'수직 Δalt {vrange:.1f}m, XY span {span:.1f}m'
        elif span > 3.0 and vpk > 0.8:
            r['guess'], r['why'] = 'F4 drag', f'직선 왕복 span {span:.1f}m, v_peak {vpk:.1f}'
        elif span < 1.5:
            r['guess'], r['why'] = 'F1 hover', f'제자리 span {span:.1f}m'
        else:
            r['guess'], r['why'] = 'position(불명)', f'span {span:.1f}m v_peak {vpk:.1f}'

    if r.get('diverged'):
        r['why'] += f'  ⚠발산?(alt_max {alt_max:.1f}m, EKF리셋 {nres})'
    r['span'], r['vpk'], r['vrange'] = span, vpk, vrange
    return r


def montage(results, outpath):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 없음 — 그림 건너뜀)")
        return
    plottable = [r for r in results if r.get('x') is not None
                 and r.get('flight') is not None and r['flight'].any()]
    if not plottable:
        print("  (그릴 궤적이 없음)")
        return
    n = len(plottable)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, plottable):
        fl = r['flight']
        x, y = r['x'][fl], r['y'][fl]
        ax.scatter(y, x, c=np.arange(len(x)), cmap='viridis', s=2)   # y=동, x=북
        ax.set_aspect('equal', 'box')
        gid = r['file'].split('_')[0]                    # log59
        ax.set_title(f"{gid}  {r['dur']:.0f}s\n{r['guess']}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('XY trajectory (color=time).  circle/figure8/waypoint/aggressive: confirm by eye',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=110)
    print(f"\n  ★ 궤적 격자 저장: {outpath}")
    print("     → 이 그림으로 F5~F8 패턴을 눈으로 최종 확인하세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='ulg 파일들 또는 폴더')
    ap.add_argument('--plot', action='store_true', help='XY 궤적 격자 PNG 저장')
    ap.add_argument('--out', default='ulog_montage.png', help='격자 PNG 경로')
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.ulg')))
        elif os.path.isfile(p):
            files.append(p)
        else:
            files += sorted(glob.glob(p))
    if not files:
        sys.exit("✗ .ulg 파일을 못 찾았습니다.")

    print("=" * 96)
    print(f"{'파일':30s} {'길이':>6s}  {'모드':10s} {'추정':22s} 근거")
    print("=" * 96)
    results = []
    for f in files:
        r = analyze(f)
        results.append(r)
        mode = ''
        if r.get('offb_frac', 0) > 0.3:
            mode = f"Offb {r['offb_frac']*100:.0f}%"
        elif r.get('pos_frac', 0) > 0.3:
            mode = f"Pos {r['pos_frac']*100:.0f}%"
        en = r.get('atk_en')
        atk = 'EN1' if (en is not None and float(en) >= 0.5) else ('EN0' if en is not None else 'EN?')
        if r.get('vra_up'):
            atk += '+VRA'
        print(f"{r['file'][:26]:26s} {r['dur']:5.0f}s  {mode:9s} {atk:8s} "
              f"{r.get('guess',''):20s} {r.get('why','')}")

    print("=" * 96)
    print("확정도:  F2(오토튠)·F9(공격) = 시그니처로 확실.  F5~F8 패턴 = 아래 그림으로 확인.")
    print("         '한 번/두세 번' 은 같은 추정 라벨이 몇 개 나오는지로 대조하세요.")

    if a.plot:
        montage(results, a.out)


if __name__ == '__main__':
    main()
