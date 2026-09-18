#!/usr/bin/env python3
"""verify_align_0812.py — 2026-08-12 정합 반영분의 sim 검증 판정.

    ~/isaacsim/python.sh verify_align_0812.py <outdir>

세 가지를 실기 실측치와 대조한다.
  ① COM_ALIGN   호버 트림 τ_norm  vs 실기 [-0.00731, -0.05539, +0.00417]
  ② ROTOR_CM    C_torque_z        vs 목표 1.378  (로터로그의 실제 요 토크 / 명령)
  ③ NIS 바닥     평시 NIS          vs 실기 gyro 0.6~1.2 / vel 0.18~0.41
"""
import csv
import os
import sys

import numpy as np

REAL_TRIM = np.array([-0.00731, -0.05539, +0.00417])     # 실기 11소티 중앙 (τ_norm)
REAL_NIS_G = (0.60, 1.17)                                # 실기 평시 gyro NIS p50 범위
REAL_NIS_V = (0.18, 0.41)                                # 실기 평시 vel  NIS p50 범위
TARGET_CTZ = 1.378


def hdr(t):
    print('\n' + '=' * 74)
    print(f' {t}')
    print('=' * 74)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'results_verify_0812'
    ok_all = True

    # ── ①② 로터로그 ─────────────────────────────────────────────
    rp = os.path.join(out, 'rotor_log.npz')
    if not os.path.exists(rp):
        print(f'[!] {rp} 없음 — ①② 판정 불가')
    else:
        z = np.load(rp)
        d, cols = z['data'], str(z['cols']).split(',')
        ix = {c: i for i, c in enumerate(cols)}
        thr = np.abs(d[:, ix['cmd_thr_z']])
        tq = d[:, [ix['cmd_tq_x'], ix['cmd_tq_y'], ix['cmd_tq_z']]]
        w = d[:, [ix['w0'], ix['w1'], ix['w2'], ix['w3']]]
        spd = np.linalg.norm(d[:, [ix['vE'], ix['vN'], ix['vU']]], axis=1)
        # ★ 로터로그 각속도는 Isaac 바디 **FLU**, cmd_torque 는 PX4 **FRD** → y,z 부호가 반대다.
        #   안 뒤집으면 ω̇z~τ_z 회귀 기울기가 음수로 나온다(2026-08-12 실측 −49.8).
        wz_frd = -d[:, ix['wz']]
        wz = np.abs(wz_frd)

        hdr('① COM_ALIGN — 호버 트림 τ_norm (부호가 핵심)')
        m = (thr > 0.15) & (spd < 0.25)                   # 떠 있고 거의 정지
        if m.sum() < 200:
            print(f'  표본 부족 ({m.sum()}) — 호버 구간이 없다')
            ok_all = False
        else:
            sim = tq[m].mean(axis=0)
            print(f"  {'':10s} {'roll':>10s} {'pitch':>10s} {'yaw':>10s}")
            print(f"  {'sim':10s} {sim[0]:+10.5f} {sim[1]:+10.5f} {sim[2]:+10.5f}   (n={m.sum()})")
            print(f"  {'실기':10s} {REAL_TRIM[0]:+10.5f} {REAL_TRIM[1]:+10.5f} {REAL_TRIM[2]:+10.5f}")
            sgn = np.sign(sim) == np.sign(REAL_TRIM)
            rel = np.abs(sim - REAL_TRIM) / np.maximum(np.abs(REAL_TRIM), 1e-4)
            print(f"  {'부호':10s} {'  일치' if sgn[0] else '  ✗반대':>10s} "
                  f"{'  일치' if sgn[1] else '  ✗반대':>10s} {'  일치' if sgn[2] else '  ✗반대':>10s}")
            print(f"  {'상대오차':10s} {rel[0]*100:9.0f}% {rel[1]*100:9.0f}% {rel[2]*100:9.0f}%")
            # 피치가 지배항이라 그것으로 판정
            good = sgn[1] and rel[1] < 0.35
            ok_all &= good
            print(f"\n  → {'✅ 통과 (피치 부호 일치 + 35% 이내)' if good else '✗ 실패 — COM 부호/크기 재조정 필요'}")
            if not sgn[1]:
                print('     ⚠ 피치 부호가 반대다. _apply_com_align 의 (K_pitch, −K_roll) 부호를 뒤집어 볼 것.')

        hdr('② ROTOR_CM — C_torque_z (요 권한)')
        # 실제 적용 요 토크 = Σ c_m ω² dir  ∝  명령 τ_z × C_torque_z
        # 회전방향은 쿼드 X 배치의 (+,+,-,-) 를 가정하지 않고, 명령과의 회귀로 게인만 뽑는다.
        me = (thr > 0.15) & (wz > 0.15)                   # 요가 실제로 움직이는 구간
        if me.sum() < 200:
            print(f'  요 여기 표본 부족 ({me.sum()}) — aggressive 패턴이 돌았는지 확인')
        else:
            # 요 각가속도 ω̇z 로부터: ω̇z = C_torque_z·τ_z/Izz  →  C_torque_z = slope·Izz
            import json
            Izz = json.load(open('calibration/calibration.json'))['drone']['Izz']
            t = d[:, ix['t']]
            wz_s = wz_frd
            dt = np.diff(t)
            good = (dt > 1e-6) & me[:-1] & me[1:]
            if good.sum() < 100:
                print('  각가속도 표본 부족')
            else:
                dwz = np.diff(wz_s)[good] / dt[good]
                tz = tq[:-1, 2][good]
                A = tz[:, None]
                slope = float(np.linalg.lstsq(A, dwz, rcond=None)[0][0])
                r2 = 1 - ((dwz - A @ [slope]) ** 2).sum() / ((dwz - dwz.mean()) ** 2).sum()
                ctz = slope * Izz
                print(f'  회귀 ω̇z ~ τ_z :  기울기 {slope:8.1f} rad/s²/unit   R² {r2:5.3f}   (n={good.sum()})')
                print(f'  C_torque_z = 기울기 × Izz({Izz:.6f}) = {ctz:.3f}   목표 {TARGET_CTZ}')
                rel = abs(ctz - TARGET_CTZ) / TARGET_CTZ
                g = rel < 0.30 and r2 > 0.2
                ok_all &= g
                print(f"\n  → {'✅ 통과' if g else '⚠ 재조정'} (오차 {rel*100:.0f}%)   "
                      f"ROTOR_CM 배율 제안 ×{TARGET_CTZ/max(ctz,1e-6):.3f}")
                if r2 <= 0.2:
                    print('     ⚠ R² 가 낮다 — 폐루프 편향. 오토튠으로 재확인하는 편이 낫다.')

    # ── ③ NIS 바닥 ───────────────────────────────────────────────
    hdr('③ 평시 NIS 바닥 — 실기와 같은 수준인가')
    dp = os.path.join(out, 'sweep_detail.csv')
    if not os.path.exists(dp):
        print(f'[!] {dp} 없음')
        return 1
    per = {}
    with open(dp, newline='') as f:
        for r in csv.DictReader(f):
            # ★ bias=0 캡처는 attack_active 가 1 로 찍혀도 δ=0 이라 물리적으로 평시다.
            #   attack_active 만으로 거르면 전 행이 날아간다(2026-08-12 실측).
            try:
                if float(r.get('bias', 0)) != 0.0 and r['attack_active'] not in ('0', '0.0', 'False', ''):
                    continue
            except ValueError:
                pass
            k = r['pattern']
            per.setdefault(k, {'v': [], 'g': []})
            try:
                per[k]['v'].append(float(r['nis_v_raw']))
                per[k]['g'].append(float(r['nis_g_raw']))
            except (ValueError, KeyError):
                pass
    if not per:
        print('  평시(무공격) 표본 없음')
        return 1
    print(f"  {'패턴':14s} {'n':>7s} {'vel p50/p95':>16s} {'gyro p50/p95':>16s}")
    for k, v in per.items():
        a, b = np.array(v['v']), np.array(v['g'])
        print(f"  {k:14s} {len(a):7d} {np.median(a):7.2f}/{np.percentile(a,95):<8.2f} "
              f"{np.median(b):7.2f}/{np.percentile(b,95):<8.2f}")
    allv = np.concatenate([np.array(v['v']) for v in per.values()])
    allg = np.concatenate([np.array(v['g']) for v in per.values()])
    mv, mg = np.median(allv), np.median(allg)
    print(f"\n  sim  전체 p50 : vel {mv:.2f}   gyro {mg:.2f}")
    print(f"  실기 평시 p50 : vel {REAL_NIS_V[0]:.2f}~{REAL_NIS_V[1]:.2f}   "
          f"gyro {REAL_NIS_G[0]:.2f}~{REAL_NIS_G[1]:.2f}")
    okg = REAL_NIS_G[0] / 3 <= mg <= REAL_NIS_G[1] * 3
    okv = REAL_NIS_V[0] / 3 <= mv <= REAL_NIS_V[1] * 3
    ok_all &= (okg and okv)
    print(f"\n  → {'✅ 같은 자릿수 (3배 이내)' if okg and okv else '⚠ 자릿수가 다르다'}")
    print(f"     gyro {'OK' if okg else 'NG'} · vel {'OK' if okv else 'NG'}")

    hdr('종합')
    print(f"  {'✅ 세 항목 모두 통과 — 정합 확정 가능' if ok_all else '⚠ 미통과 항목 있음 — 위 지시대로 재조정 후 재비행'}")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
