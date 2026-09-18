"""compare_p0_tau.py — 모터 지연 유무의 요 진동 비교 (2026-07-29)

사용: ~/isaacsim/python.sh compare_p0_tau.py results_p0_tau0 results_p0_flightcheck

목적: P0 비행검증에서 발견된 지속적 요 진동(±20~30°, 요레이트 ±3 rad/s)의 원인이
      (a) 새로 주입한 모터 지연 τ_down=70ms 인지
      (b) 2026-07-28 관성 변경(Izz 감소 + PX4 게인 불변)인지 구분한다.
      τ=0 대조군에서도 같은 진동이 나오면 (b), 사라지면 (a).
"""
import os, sys
import csv
import numpy as np

RAD = np.pi / 180.0
dirs = sys.argv[1:] or ['results_p0_tau0', 'results_p0_flightcheck']


def seglist(outdir):
    """zu_log 를 reset 마커로 세그먼트 분할 + summary 행과 순서 매핑."""
    d = np.load(os.path.join(outdir, 'zu_log.npz'), allow_pickle=True)
    A = d['data']
    cols = [s.strip() for s in str(d['cols']).replace(',', ' ').split()]
    c = {n: i for i, n in enumerate(cols)}
    bnd = np.where(A[:, c['reset']] == 1)[0]
    starts = list(bnd) if (len(bnd) and bnd[0] == 0) else [0] + list(bnd)
    segs = [(s, starts[k + 1] if k + 1 < len(starts) else len(A))
            for k, s in enumerate(starts)]
    sm = list(csv.DictReader(open(os.path.join(outdir, 'sweep_summary.csv'))))
    return A, c, segs, sm


def eff_dt(segs, sm):
    """zu 저장 dt 는 명목값(0.02)이라 실제 간격을 에피소드 길이에서 역산."""
    n = min(len(segs), len(sm))
    tot_samp = sum(e - s for s, e in segs[:n])
    tot_sec = sum(float(sm[k]['steps']) * 0.1 for k in range(n))
    return tot_sec / max(tot_samp, 1)


def metrics(outdir):
    A, c, segs, sm = seglist(outdir)
    dt = eff_dt(segs, sm)
    n = min(len(segs), len(sm))
    out = {}
    for k in range(n):
        s, e = segs[k]
        r = sm[k]
        if e - s < 64:
            continue
        gz = A[s:e, c['z8_gyrz']]
        gx = A[s:e, c['z6_gyrx']]
        gy = A[s:e, c['z7_gyry']]
        psi = A[s:e, c['euler_psi']]
        pu = np.unwrap(psi)
        # 저주파 추세 제거(패턴 자체의 요 회전) 후 진동 성분만
        t = np.arange(len(pu))
        trend = np.polyval(np.polyfit(t, pu, 2), t)
        osc = pu - trend
        # 지배 주파수 (gyro_z FFT, DC 제외)
        w = gz - gz.mean()
        sp = np.abs(np.fft.rfft(w * np.hanning(len(w))))
        fr = np.fft.rfftfreq(len(w), dt)
        fpk = fr[1 + np.argmax(sp[1:])] if len(sp) > 2 else 0.0
        rec = out.setdefault(r['pattern'], {'osc': [], 'gz': [], 'gxy': [], 'f': []})
        rec['osc'].append(np.std(osc))
        rec['gz'].append(np.percentile(np.abs(gz), 95))
        rec['gxy'].append(np.percentile(np.abs(np.hypot(gx, gy)), 95))
        rec['f'].append(fpk)
    return dt, out


print("=" * 84)
print("모터 지연 유무 — 요 진동 비교")
print("=" * 84)
res = {}
for d0 in dirs:
    if not os.path.exists(os.path.join(d0, 'zu_log.npz')):
        print(f"  ⚠ {d0}: zu_log.npz 없음 — 건너뜀")
        continue
    dt, m = metrics(d0)
    res[d0] = m
    print(f"\n[{d0}]  실효 로깅 간격 dt = {dt*1000:.1f} ms ({1/dt:.1f} Hz)")
    print(f"  {'패턴':>11s} {'요진동 std':>11s} {'|gyro_z| 95pct':>15s} {'|gyro_xy| 95pct':>16s} {'지배주파수':>11s}")
    for p in sorted(m):
        v = m[p]
        print(f"  {p:>11s} {np.mean(v['osc'])/RAD:9.2f}°  {np.mean(v['gz']):13.3f}   "
              f"{np.mean(v['gxy']):14.3f}   {np.mean(v['f']):9.2f} Hz")

if len(res) == 2:
    a, b = dirs[0], dirs[1]
    print("\n" + "=" * 84)
    print(f"판정 — {a}(대조) vs {b}")
    print("=" * 84)
    print(f"  {'패턴':>11s} {'요진동 std':>22s} {'|gyro_z| 95pct':>22s}")
    print(f"  {'':>11s} {a.split('_')[-1]:>10s} → {b.split('_')[-1]:<9s} {a.split('_')[-1]:>10s} → {b.split('_')[-1]:<9s}")
    ratios = []
    for p in sorted(set(res[a]) & set(res[b])):
        o0, o1 = np.mean(res[a][p]['osc']), np.mean(res[b][p]['osc'])
        g0, g1 = np.mean(res[a][p]['gz']), np.mean(res[b][p]['gz'])
        ratios.append(o1 / max(o0, 1e-9))
        print(f"  {p:>11s} {o0/RAD:9.2f}° → {o1/RAD:8.2f}°  {g0:10.3f} → {g1:9.3f}")
    rmean = float(np.mean(ratios))
    print(f"\n  요진동 배율 평균 = {rmean:.2f}×")
    if rmean > 2.0:
        print("  → ★ 모터 지연이 원인. τ_down=70ms 가 요 루프를 불안정화했다.")
        print("     조치: τ_down 축소 또는 PX4 요 게인 재튜닝 필요. 밴드 측정 전에 해결할 것.")
    elif rmean < 1.5:
        print("  → ★ 모터 지연은 원인이 아니다. τ=0 에서도 같은 진동 = 07-28 관성 변경 쪽 의심.")
        print("     조치: Izz 감소(0.0629→0.0534) 대비 PX4 요 게인이 과대. 게인 재튜닝 검토.")
    else:
        print("  → 판정 애매(1.5~2.0×). 두 원인이 겹칠 수 있음 — τ 중간값으로 추가 확인 권장.")
print("=" * 84)
