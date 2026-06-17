"""
sweep_compare.py — combined/torque/thrust 3개 sweep 결과를 한 표로 비교
====================================================================
사용:
    python sweep_compare.py                       # results_{combined,torque,thrust}
    python sweep_compare.py dirA dirB dirC ...     # 직접 지정

각 모드별: 결과성 밴드(track추락∧hover생존) + 공격 NIS 분리(d', vel/gyr) 요약.
"""
import sys, os, csv
from collections import defaultdict
import numpy as np


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bval(r):
    return float(r.get('bias', r.get('alpha', 0.0)))


def analyze_dir(d):
    sp = os.path.join(d, 'sweep_summary.csv')
    dp = os.path.join(d, 'sweep_detail.csv')
    if not os.path.exists(sp):
        return None
    summ = load(sp)
    detail = load(dp) if os.path.exists(dp) else []
    mode = summ[0].get('mode', os.path.basename(d)) if summ else '?'
    unit = 'N' if mode == 'thrust' else 'Nm'
    biases = sorted(set(bval(r) for r in summ))

    inj = {}
    for r in summ:
        if 'tq_xy' in r and 'th_n' in r:
            inj[bval(r)] = (float(r['tq_xy']), float(r['th_n']))

    surv = defaultdict(list)
    creason = defaultdict(lambda: defaultdict(int))
    for r in summ:
        surv[(bval(r), r['policy'])].append(int(r['survived']))
        if not int(r['survived']):
            creason[(bval(r), r['policy'])][r['crash_reason']] += 1

    # 밴드: track 추락(<0.5) ∧ hover 생존(>0.5)
    band = [b for b in biases if b > 0
            and np.mean(surv.get((b, 'track'), [1])) < 0.5
            and np.mean(surv.get((b, 'hover'), [0])) > 0.5]
    # track 첫 추락 / hover 첫 추락 임계
    def first_crash(pol):
        for b in biases:
            if b > 0 and np.mean(surv.get((b, pol), [1])) < 0.5:
                return b
        return None
    b_track = first_crash('track')
    b_hover = first_crash('hover')

    # NIS: baseline(b=0,track) vs 각 bias
    def pick(pred):
        v, g = [], []
        for r in detail:
            if pred(r):
                v.append(float(r['nis_v_raw'])); g.append(float(r['nis_g_raw']))
        return np.array(v), np.array(g)
    bv, bg = pick(lambda r: bval(r) == 0.0 and r['policy'] == 'track')
    base_gm, base_gs = (bg.mean(), bg.std()) if len(bg) else (0., 1.)
    base_vm, base_vs = (bv.mean(), bv.std()) if len(bv) else (0., 1.)

    # 대표 bias = 밴드 있으면 밴드 하한, 없으면 최대 생존 bias
    rep = band[0] if band else (max([b for b in biases if b > 0
                                     and np.mean(surv.get((b, 'hover'), [0])) > 0.5], default=None))
    sep = None
    if rep is not None:
        av, ag = pick(lambda r, rep=rep: bval(r) == rep and r['policy'] == 'hover'
                      and r['attack_active'] == '1')
        if len(ag):
            sep = {
                'bias': rep,
                'vel_dprime': (av.mean() - base_vm) / (base_vs + 1e-9),
                'gyr_dprime': (ag.mean() - base_gm) / (base_gs + 1e-9),
                'vel_mean': av.mean(), 'gyr_mean': ag.mean(),
            }

    return dict(mode=mode, unit=unit, biases=biases, band=band, inj=inj,
                b_track=b_track, b_hover=b_hover, creason=creason,
                base_gm=base_gm, base_vm=base_vm, sep=sep, surv=surv)


def main():
    dirs = sys.argv[1:] if len(sys.argv) > 1 else \
        ['results_combined', 'results_torque', 'results_thrust']
    res = [(d, analyze_dir(d)) for d in dirs]

    print("\n" + "=" * 78)
    print("  SWEEP 모드 비교")
    print("=" * 78)
    hdr = f"{'mode':>9} {'unit':>4} | {'b_track':>8} {'b_hover':>8} | {'밴드(track↓∧hover↑)':>22}"
    print(hdr); print("-" * 78)
    for d, a in res:
        if a is None:
            print(f"{d:>9}  (sweep_summary.csv 없음)")
            continue
        bt = f"{a['b_track']:.3f}" if a['b_track'] is not None else "—(없음)"
        bh = f"{a['b_hover']:.3f}" if a['b_hover'] is not None else "—(없음)"
        if a['band']:
            lo, hi = min(a['band']), max(a['band'])
            bandstr = f"[{lo:.3f},{hi:.3f}]{a['unit']}"
        else:
            bandstr = "없음"
        print(f"{a['mode']:>9} {a['unit']:>4} | {bt:>8} {bh:>8} | {bandstr:>22}")

    print("\n" + "-" * 78)
    print("  공격 검출 분리도 (대표 bias, hover, attack-on; baseline=b0/track)")
    print("-" * 78)
    print(f"{'mode':>9} | {'rep bias':>9} | {'vel d′':>8} {'gyr d′':>9} | "
          f"{'vel mean':>9} {'gyr mean':>9}")
    for d, a in res:
        if a is None or not a['sep']:
            continue
        s = a['sep']
        print(f"{a['mode']:>9} | {s['bias']:9.3f} | {s['vel_dprime']:8.1f} "
              f"{s['gyr_dprime']:9.1f} | {s['vel_mean']:9.3f} {s['gyr_mean']:9.3f}")

    # 밴드점 실제 주입값
    print("\n" + "-" * 78)
    print("  밴드 지점 실제 주입 바이어스")
    print("-" * 78)
    for d, a in res:
        if a is None:
            continue
        if a['band']:
            b = a['band'][0]
            if b in a['inj']:
                tq, th = a['inj'][b]
                print(f"  [{a['mode']}] b={b:.3f}{a['unit']} → 토크 {tq:.3f}Nm + 추력 {th:.3f}N")
            crs = dict(a['creason'].get((b, 'track'), {}))
            print(f"            track 추락모드: {crs}")
        else:
            print(f"  [{a['mode']}] 밴드 없음 "
                  f"(b_track≈b_hover → 정책무관 또는 track 안추락)")
    print()


if __name__ == '__main__':
    main()
