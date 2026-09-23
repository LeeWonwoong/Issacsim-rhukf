#!/usr/bin/env python3
"""check_timing.py — SIMCLOCK_UKF·LEARNER_PROC Isaac 검증 런 판독 (numpy 만 씀 — 시스템 python3 로 실행 가능).

사용:
  python3 check_timing.py R1=<adam_dir> R2=<swirl_ui1_dir> R3=<swirl_ui4_dir> R4=<adam_speed15_dir> [--ep 11-40] [--act-lat 0.06]
  (이름=폴더 를 원하는 만큼. 짝 비교는 R1 을 기준으로 R2·R3(학습기 무관), R4(speed 불변성) 를 한다.)

런 하나마다(steps/ep*.npz 의 TIMING_LOG 열, 이름 기반):
  · 타이밍 불변식: n_pred==5 · sum_dt==0.1 · dt_gps==100000 µs · gps_gap==0 · gt_gap==0 (전 행 비율)
  · 노드 지연: node_lag_sim p50/p99/max, >0.02 s 비율 · deferred(결정 전에 온 격자) 비율
  · 행동 적용: act_applied_sim_lat 분포, overrun 비율(전체·에피 최대), act_eff_lat p50/p99/max
  · 학습기: act_wait_ms·learn_ms p50/p99/max
  · 입력 짝 어긋남 추정(sim ms) = (gt_rx_lag_ms − imu_age_ms)·speed  p5/p50/p95  (+ u 도 같이)
  · 정책 무관 gyro spike(>10) 비율 [prefix: step≥16, 첫 hover·첫 공격 이전] · clean-track spike 비율
    [현재·직전 10 스텝 모두 track·무공격, step≥16]
  · train_stdout.log: [LEARN ERROR]·💀·[SIMCLOCK]·[ACT_LAT] 경고·학습기 불일치 수, simclock 요약 줄
짝(R1 기준): clean-track·prefix spike OR(95% CI, Fisher p) · 온셋 노출(onset+0..2 gyro NIS 평균비, 부트스트랩 CI)
  · R4 vs R1(speed 불변성): 평시 NIS(v·g) KS 검정 p · clean spike 차(%p, 95% CI) · gt_err 중앙값 차
판정 줄(PASS/FAIL/INFO)은 TIMING_REPORT §5 합격 기준을 따른다.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

SPIKE = 10.0
W_CLEAN = 10


# ── 통계 (scipy 없이) ─────────────────────────────────────────────────
def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_p(a, b, c, d):
    """2×2 [[a,b],[c,d]] 양측 Fisher exact p (초기하 확률 ≤ 관측 확률의 합)."""
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    lp = lambda x: _lchoose(c1, x) + _lchoose(n - c1, r1 - x) - _lchoose(n, r1)
    p0 = lp(a); tot = 0.0
    for x in range(lo, hi + 1):
        v = lp(x)
        if v <= p0 + 1e-7:
            tot += math.exp(v)
    return min(1.0, tot)


def odds_ratio(s1, n1, s0, n0):
    a, b, c, d = s1, n1 - s1, s0, n0 - s0
    orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    se = math.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return orr, (orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se)), fisher_p(a, b, c, d)


def ks_2samp(x, y):
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float))
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan'), float('nan')
    allv = np.concatenate([x, y])
    D = float(np.max(np.abs(np.searchsorted(x, allv, 'right') / len(x) - np.searchsorted(y, allv, 'right') / len(y))))
    en = math.sqrt(len(x) * len(y) / (len(x) + len(y)))
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return D, float(min(max(p, 0.0), 1.0))


def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float('nan')


# ── 로드 ──────────────────────────────────────────────────────────────
def run_speed(d):
    for f in (os.path.join(d, 'config.yaml'),):
        if os.path.exists(f):
            try:
                import yaml
                return float(yaml.safe_load(open(f))['env']['isaac']['speed'])
            except Exception:
                pass
            t = open(f).read()
            m = re.search(r'\n\s+speed:\s*([\d.]+)', t)
            if m:
                return float(m.group(1))
    t = open(os.path.join(d, 'run.log')).read() if os.path.exists(os.path.join(d, 'run.log')) else ''
    m = re.search(r'env\.isaac\.speed=([\d.]+)', t)
    return float(m.group(1)) if m else float('nan')


def load_run(d, ep_lo, ep_hi):
    E = {}
    for f in sorted(glob.glob(os.path.join(d, 'steps', 'ep*.npz'))):
        ep = int(re.findall(r'ep(\d+)', os.path.basename(f))[0])
        if not (ep_lo <= ep <= ep_hi):
            continue
        z = np.load(f, allow_pickle=False)
        cols = [str(c) for c in z['cols']]; rows = z['rows']
        E[ep] = dict(C={c: rows[:, i] for i, c in enumerate(cols)}, cols=cols, reason=str(z['reason']) if 'reason' in z else '',
                     pattern=str(z['pattern']) if 'pattern' in z else '')
    return E


def prefix_mask(C, skip=16):
    st = C['step'].astype(int); pa = C['prev_action'].astype(int); af = C['atk_flag'].astype(int); de = C['delta_eff']
    ok = st >= skip; bad = (pa == 1) | (af == 1) | (np.abs(de) > 0)
    if bad.any():
        ok[int(np.argmax(bad)):] = False
    return ok


def clean_mask(C):
    pa = C['prev_action'].astype(int); af = C['atk_flag'].astype(int); de = np.abs(C['delta_eff']) > 0; st = C['step'].astype(int)
    bad = (pa == 1) | (af == 1) | de
    out = np.zeros(len(st), bool)
    cut = np.r_[0, np.where(np.diff(st) != 1)[0] + 1, len(st)]
    for a, b in zip(cut[:-1], cut[1:]):
        bb = bad[a:b]
        cb = np.convolve(bb.astype(int), np.ones(W_CLEAN + 1, int), mode='full')[:len(bb)]
        out[a:b] = (cb == 0) & (st[a:b] >= 16)
    return out


def log_counts(d):
    f = os.path.join(d, 'train_stdout.log')
    if not os.path.exists(f):
        return {}
    t = open(f, errors='ignore').read()
    return {'LEARN ERROR': t.count('[LEARN ERROR]'), 'fatal(💀)': t.count('💀'), '[SIMCLOCK] warn': t.count('[SIMCLOCK]'),
            '[ACT_LAT] overrun warn': t.count('[ACT_LAT]'), 'overrun>1% 에피': t.count('[TIMING] overrun'),
            '갱신 예측 불일치': t.count('갱신 예측 불일치'), 'Traceback': t.count('Traceback'),
            'learner mismatch>0': len([m for m in re.findall(r'learner mismatch=(\d+)', t) if int(m) > 0]),
            'RUN_DONE': (open(os.path.join(d, 'RUN_DONE')).read().strip() if os.path.exists(os.path.join(d, 'RUN_DONE')) else '없음')}


# ── 런 판독 ───────────────────────────────────────────────────────────
def analyze(name, d, ep_lo, ep_hi, act_lat):
    E = load_run(d, ep_lo, ep_hi)
    if not E:
        print(f'## {name}: {d} — steps npz 없음(ep{ep_lo}-{ep_hi})'); return None
    speed = run_speed(d)
    need = ['n_pred', 'dt_gps_us', 'overrun', 'act_applied_sim_lat', 'act_eff_lat']
    have_t = all(c in E[next(iter(E))]['cols'] for c in need)
    cat = lambda k: np.concatenate([e['C'][k] for e in E.values() if k in e['C']]) if any(k in e['C'] for e in E.values()) else np.array([])
    R = dict(name=name, dir=d, speed=speed, eps=sorted(E), E=E, verdict=[])
    print(f'\n## {name}  {d}\n   speed={speed}  에피 {len(E)} (ep{min(E)}–{max(E)})  행 {sum(len(e["C"]["step"]) for e in E.values())}')
    if not have_t:
        print('   ⚠ TIMING_LOG 열 없음 — 타이밍 판독 생략 (TIMING_LOG=1 로 돌린 런인가?)')
    else:
        npred = cat('n_pred'); sdt = cat('sum_dt_pred'); dtg = cat('dt_gps_us'); gg = cat('gps_gap'); tg = cat('gt_gap')
        lag = cat('node_lag_sim'); ov = cat('overrun'); lat = cat('act_applied_sim_lat'); eff = cat('act_eff_lat')
        dfr = cat('deferred'); aw = cat('act_wait_ms'); lm = cat('learn_ms')
        f5 = float(np.mean(npred == 5)); fdt = float(np.mean(np.isclose(sdt, 0.1))); fg = float(np.mean(dtg == 100000))
        print(f'   n_pred==5 {100*f5:.3f}%  (분포 {dict(zip(*np.unique(npred, return_counts=True)))})  Σdt==0.1 {100*fdt:.3f}%')
        print(f'   dt_gps==100000µs {100*fg:.3f}%  (분포 {dict(zip(*np.unique(dtg[np.isfinite(dtg)], return_counts=True)))})  '
              f'gps_gap {int(np.nansum(gg))}  gt_gap {int(np.nansum(tg))}')
        print(f'   node_lag_sim p50/p99/max = {pct(lag,50):.3f}/{pct(lag,99):.3f}/{np.nanmax(lag):.3f} s  (>0.02 s {100*np.mean(lag > 0.02):.2f}%)  '
              f'deferred {100*np.nanmean(dfr):.2f}%')
        m = np.isfinite(ov)
        ovf = float(np.nanmean(ov)) if m.any() else float('nan')
        ep_ov = [float(np.nanmean(e['C']['overrun'])) for e in E.values() if np.isfinite(e['C']['overrun']).any()]
        u, c = np.unique(np.round(lat[np.isfinite(lat)], 3), return_counts=True)
        print(f'   act_applied_sim_lat 분포 {dict(zip(u.tolist(), c.tolist()))}  overrun {100*ovf:.2f}% (에피 최대 {100*max(ep_ov or [0]):.1f}%, '
              f'>1% 에피 {sum(x > 0.01 for x in ep_ov)}/{len(ep_ov)})')
        print(f'   act_eff_lat p50/p99/max = {pct(eff,50):.3f}/{pct(eff,99):.3f}/{np.nanmax(eff):.3f} s   '
              f'act_wait_ms p50/p99/max = {pct(aw,50):.1f}/{pct(aw,99):.1f}/{np.nanmax(aw):.1f}   '
              f'learn_ms(갱신) p50/p99/max = {pct(lm[lm > 0],50):.1f}/{pct(lm[lm > 0],99):.1f}/{np.nanmax(lm) if np.isfinite(lm).any() else float("nan"):.1f}')
        if 'gt_rx_lag_ms' in E[next(iter(E))]['cols']:
            rx = cat('gt_rx_lag_ms'); ia = cat('imu_age_ms'); ua = cat('u_age_ms')
            sk_i = (rx - ia) * speed; sk_u = (rx - ua) * speed
            print(f'   GT 전달지연 ms p50/p99 = {pct(rx,50):.1f}/{pct(rx,99):.1f}  │ 짝 어긋남 추정(sim ms, +면 IMU/u 가 GT 보다 새것) '
                  f'IMU p5/p50/p95 = {pct(sk_i,5):.1f}/{pct(sk_i,50):.1f}/{pct(sk_i,95):.1f}  u = {pct(sk_u,5):.1f}/{pct(sk_u,50):.1f}/{pct(sk_u,95):.1f}')
            R['skew95'] = max(abs(pct(sk_i, 5)), abs(pct(sk_i, 95)))
        ok_lat = np.isfinite(lat) & (ov == 0)
        lat_fixed = bool(ok_lat.any() and np.allclose(lat[ok_lat], act_lat if act_lat > 0 else 0.02)) if act_lat is not None else None
        R['verdict'] += [
            ('n_pred==5 · Σdt 0.1 (100%)', f5 == 1.0 and fdt == 1.0),
            ('dt_gps 100% 100000 µs · gps_gap 0 · gt_gap 0', fg == 1.0 and np.nansum(gg) == 0 and np.nansum(tg) == 0),
            ('node_lag_sim p99 ≤ 0.02 s', pct(lag, 99) <= 0.02 + 1e-9),
            ('overrun ≤ 1% (런 전체)', ovf <= 0.01),
        ]
        if lat_fixed is not None:
            R['verdict'].append((f'적용 지연 = ACT_LAT {act_lat} (overrun 아닌 행)', lat_fixed))
            R['verdict'].append((f'act_eff_lat p99 ≤ ACT_LAT+0.02', pct(eff, 99) <= (act_lat if act_lat > 0 else 0.02) + 0.02 + 1e-9))
        if 'skew95' in R:
            R['verdict'].append(('INFO 짝 어긋남 |p5|,|p95| ≤ 4 ms sim (IMU 1 샘플) — 넘으면 C 단계(스탬프 정렬) 검토', R['skew95'] <= 4.0))
    # spike
    sp = dict(prefix=[0, 0], clean=[0, 0])
    for e in E.values():
        C = e['C']; g = C['nis_g_raw']
        for key, fn in (('prefix', prefix_mask), ('clean', clean_mask)):
            mk = fn(C); sp[key][0] += int((g[mk] > SPIKE).sum()); sp[key][1] += int(mk.sum())
    R['sp'] = sp
    print(f'   정책 무관(prefix) spike>10 {sp["prefix"][0]}/{sp["prefix"][1]} = {100*sp["prefix"][0]/max(sp["prefix"][1],1):.3f}%   '
          f'clean-track {sp["clean"][0]}/{sp["clean"][1]} = {100*sp["clean"][0]/max(sp["clean"][1],1):.3f}%')
    lc = log_counts(d)
    if lc:
        print('   로그: ' + '  '.join(f'{k}={v}' for k, v in lc.items()))
        R['verdict'].append(('로그: LEARN ERROR·💀·Traceback·예측 불일치 0 · RUN_DONE=0',
                             lc['LEARN ERROR'] == 0 and lc['fatal(💀)'] == 0 and lc['Traceback'] == 0 and lc['갱신 예측 불일치'] == 0
                             and lc['RUN_DONE'] == '0'))
    for k, v in R['verdict']:
        tag = ('INFO-OK' if v else 'INFO-주의') if k.startswith('INFO') else ('PASS' if v else 'FAIL')
        print(f'   [{tag}] {k}')
    return R


def onset_exposure(A, B, k_max=2, n_boot=2000, seed=0):
    """같은 에피(같은 시나리오)의 onset+0..k gyro NIS 평균비 A/B (부트스트랩 CI). onset = delta_eff 가 처음 0 이 아닌 행."""
    out = []
    for k in range(k_max + 1):
        xa, xb = [], []
        for ep in sorted(set(A['E']) & set(B['E'])):
            ca, cb = A['E'][ep]['C'], B['E'][ep]['C']
            ia = np.flatnonzero(np.abs(ca['delta_eff']) > 0); ib = np.flatnonzero(np.abs(cb['delta_eff']) > 0)
            if len(ia) and len(ib) and ia[0] + k < len(ca['step']) and ib[0] + k < len(cb['step']):
                xa.append(ca['nis_g_raw'][ia[0] + k]); xb.append(cb['nis_g_raw'][ib[0] + k])
        if len(xa) < 3:
            out.append((k, len(xa), float('nan'), (float('nan'), float('nan')))); continue
        xa, xb = np.array(xa), np.array(xb); rng = np.random.default_rng(seed)
        bs = [np.mean(xa[i]) / max(np.mean(xb[i]), 1e-12) for i in (rng.integers(0, len(xa), len(xa)) for _ in range(n_boot))]
        out.append((k, len(xa), float(np.mean(xa) / max(np.mean(xb), 1e-12)), (pct(bs, 2.5), pct(bs, 97.5))))
    return out


def compare(A, B, label):
    common = sorted(set(A['E']) & set(B['E']))
    print(f'\n### {label}: {A["name"]} vs {B["name"]} (공통 에피 {len(common)})')
    res = []
    for key, fn in (('clean', clean_mask), ('prefix', prefix_mask)):
        s = {}
        for R in (A, B):
            n = k = 0
            for ep in common:
                C = R['E'][ep]['C']; mk = fn(C); k += int((C['nis_g_raw'][mk] > SPIKE).sum()); n += int(mk.sum())
            s[R['name']] = (k, n)
        (ka, na), (kb, nb) = s[A['name']], s[B['name']]
        orr, ci, p = odds_ratio(ka, na, kb, nb)
        ok = ci[0] <= 1.0 <= ci[1]
        print(f'   {key:6s} spike  {A["name"]} {100*ka/max(na,1):.3f}% vs {B["name"]} {100*kb/max(nb,1):.3f}%  '
              f'OR {orr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  p={p:.2g}  → {"PASS(CI∋1)" if ok else "FAIL(CI∌1)"}')
        res.append((f'{label} {key} OR CI∋1', ok))
    for k, n, r, ci in onset_exposure(A, B):
        print(f'   onset+{k} gyro NIS 평균비 {A["name"]}/{B["name"]} = {r:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] (n={n})')
    return res


def speed_invariance(A, B):
    """B(다른 speed) vs A: 평시 NIS KS · clean spike 차 CI · gt_err 중앙값 차."""
    common = sorted(set(A['E']) & set(B['E']))
    print(f'\n### speed 불변성: {A["name"]}(speed {A["speed"]}) vs {B["name"]}(speed {B["speed"]}) (공통 에피 {len(common)})')
    res = []
    for ch in ('nis_v_raw', 'nis_g_raw'):
        xa = np.concatenate([A['E'][e]['C'][ch][prefix_mask(A['E'][e]['C'])] for e in common])
        xb = np.concatenate([B['E'][e]['C'][ch][prefix_mask(B['E'][e]['C'])] for e in common])
        D, p = ks_2samp(xa, xb)
        print(f'   평시(prefix) {ch}: 중앙 {np.median(xa):.3f} vs {np.median(xb):.3f}  KS D={D:.3f} p={p:.3g} → {"PASS" if p > 0.05 else "FAIL"}')
        res.append((f'speed 불변 KS {ch} p>0.05', p > 0.05))
    ka = na = kb = nb = 0
    for e in common:
        for R, acc in ((A, 'a'), (B, 'b')):
            C = R['E'][e]['C']; mk = clean_mask(C); k = int((C['nis_g_raw'][mk] > SPIKE).sum()); n = int(mk.sum())
            if acc == 'a': ka += k; na += n
            else: kb += k; nb += n
    pa, pb = ka / max(na, 1), kb / max(nb, 1)
    se = math.sqrt(pa * (1 - pa) / max(na, 1) + pb * (1 - pb) / max(nb, 1))
    lo, hi = (pb - pa) - 1.96 * se, (pb - pa) + 1.96 * se
    ok = lo <= 0 <= hi
    print(f'   clean spike 차 {100*(pb-pa):+.3f}%p [{100*lo:+.3f}, {100*hi:+.3f}] → {"PASS(CI∋0)" if ok else "FAIL"}')
    res.append(('speed 불변 clean spike 차 CI∋0', ok))
    if all('gt_err' in A['E'][e]['C'] and 'gt_err' in B['E'][e]['C'] for e in common):
        ga = np.concatenate([A['E'][e]['C']['gt_err'][A['E'][e]['C']['prev_action'] == 0] for e in common])
        gb = np.concatenate([B['E'][e]['C']['gt_err'][B['E'][e]['C']['prev_action'] == 0] for e in common])
        ga, gb = ga[ga >= 0], gb[gb >= 0]
        d = float(np.median(gb) - np.median(ga))
        print(f'   gt_err(track 행) 중앙값 {np.median(ga):.3f} vs {np.median(gb):.3f} m (차 {d:+.3f}) → {"PASS" if abs(d) < 0.05 else "FAIL"}')
        res.append(('speed 불변 gt_err 중앙값 차 < 0.05 m', abs(d) < 0.05))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='이름=폴더 (R1=... R2=...)')
    ap.add_argument('--ep', default='11-40', help='분석 에피 범위 (기본 11-40)')
    ap.add_argument('--act-lat', type=float, default=None, help='기대 ACT_LAT_SIM (주면 적용 지연 고정 판정)')
    ap.add_argument('--ref', default='R1', help='짝 비교 기준 이름 (기본 R1)')
    ap.add_argument('--speed-pair', default='R4', help='speed 불변성 비교 대상 이름 (기본 R4)')
    ap.add_argument('--json', default='', help='판정 요약 json 저장 경로')
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.ep.split('-'))
    runs = {}
    for s in a.runs:
        n, d = s.split('=', 1) if '=' in s else (os.path.basename(s.rstrip('/')), s)
        R = analyze(n, d, lo, hi, a.act_lat)
        if R is not None:
            runs[n] = R
    verdict = {n: R['verdict'] for n, R in runs.items()}
    if a.ref in runs:
        A = runs[a.ref]
        for n, B in runs.items():
            if n == a.ref:
                continue
            if n == a.speed_pair:
                verdict[f'{n}~{a.ref}'] = speed_invariance(A, B)
            else:
                verdict[f'{n}~{a.ref}'] = compare(B, A, '학습기 무관')
    print('\n## 요약')
    n_fail = 0
    for k, vs in verdict.items():
        for name, ok in vs:
            if name.startswith('INFO'):
                continue
            n_fail += int(not ok)
            print(f'   {"PASS" if ok else "FAIL"}  {k}: {name}')
    print(f'\n   FAIL {n_fail} 건')
    if a.json:
        json.dump({k: [(n, bool(v)) for n, v in vs] for k, vs in verdict.items()}, open(a.json, 'w'), ensure_ascii=False, indent=1)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
