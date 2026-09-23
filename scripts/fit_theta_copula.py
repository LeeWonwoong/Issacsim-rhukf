#!/usr/bin/env python3
"""scripts/fit_theta_copula.py — surrogate θ 채널 코퓰러 적합 + Isaac 홀드아웃 관문 (★09-24, P2S p3/p4/p6 이식)

surrogate θ 채널(sim/surrogate.py, env.surrogate.theta)은 2단 k-NN 으로 θ 의 조건부 경험분포를 고르고,
그 분위를 AR 코퓰러 z 로 뽑는다:  z = 에피 효과(s_m²) + AR(ρ1)·w1 + AR(ρ2)·w2 + 너겟(1−합).
이 스크립트가 그 5개 값을 **풀과 Isaac 홀드아웃에서 다시 적합**하고, 적합값을 풀 npz 에 저장한다
(load_pool 이 읽어 설정 명시값이 없으면 우선 사용 — 없으면 J32a 기본값 + 경고).

  fit   python3 scripts/fit_theta_copula.py fit  <풀.npz> --hold <런폴더> [<런폴더> ...] [--fit-runs 0 2] [--write]
        홀드아웃 행마다 구현과 같은 2단 조건화(1단: _feat + 직전 6 행동 비트×abw 로 K1 이웃, 2단: Isaac 의 실제 NIS 창이
        가까운 M 행)로 PIT z = Φ⁻¹(분위) 를 만들고, --fit-runs 런의 에피 내 ACF(lag 1–40) 에 모형을 최소제곱 적합한다.
        --write 면 풀 npz 에 theta_copula(5)·theta_fit_meta(JSON: 홀드아웃·목표/모형 ACF·오차·K1/M/abw) 를 덧쓴다.
  gate  python3 scripts/fit_theta_copula.py gate <풀.npz> --hold <런폴더> [...] [--copula s m w1 r1 w2 r2] [--json 출력]
        홀드아웃 에피를 surrogate 로 강제 1:1 재생(같은 공격 계획·풍속·패턴·행동열)해 θ 를 뽑고 Isaac 과 비교:
        클래스 평균(평시/공격 × track/hover) · 에피 내 ACF(평균제거) ρ1/ρ3/ρ10 · corr(θ, log NIS_g) 전체/평시/공격 ·
        frz3 λ4 성형 타깃 G3 = γ³Φ(t+3) − Φ(t) sd (ep≥100; 전체/평시track/평시hover/공격track/공격hover) · 공격 라벨 일치율.
        P2S 판정(J32a, v2clean 풀): θ 평균 0.119/0.152/0.172/0.184 · ρ1/ρ3/ρ10 0.70/0.46/0.14 · corr 0.334/0.328/0.126 ·
        G3 2.83/2.70/2.67/3.16/4.63  —  Isaac 0.121/0.155/0.173/0.194 · 0.80/0.55/0.15 · 0.328/0.311/0.131 · 2.59/2.44/3.89/3.40/4.31.

규칙: 홀드아웃 에피가 풀 출처에 들어 있으면 거부한다(--allow-overlap 로 무시). 홀드아웃 행은 HARD 리셋 앞붙음을 잘라낸다
(P2S p1 과 같은 규칙; 풀 빌더는 그런 에피를 통째로 버린다). 풀은 knn_v4(θ 열) 여야 한다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 scripts/fit_theta_copula.py ...
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))

LAGS = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40])
PATS = ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve']


# ───────────────────────────── 홀드아웃 적재 ─────────────────────────────
def episode_rows_trim(f):
    """steps/capture 에피 npz → (X 15열, step, z). HARD 리셋 앞붙음(스텝 역행) 이전은 잘라낸다. 못 쓰면 None."""
    import build_pool_knn as B
    z = np.load(f, allow_pickle=True)
    cols = [str(c) for c in z['cols']]; R = z['rows'].astype(float)
    st = R[:, cols.index('step')]
    back = np.flatnonzero(np.diff(st) < 0)
    if len(back):
        R = R[back[-1] + 1:]
    zz = {k: z[k] for k in z.files}; zz['rows'] = R; zz['cols'] = np.array(cols)

    class _Z(dict):
        files = list(zz.keys())
    X = B.episode_rows(_Z(zz))
    if X is None or X.shape[1] < 15:
        return None
    return X, R[:, cols.index('step')].astype(int), z


def load_holdout(dirs):
    """런 폴더 목록 → dict(X, ep, run, step, epnum, src, runs). 폴더마다 steps/ 와 capture/ 를 본다."""
    groups = [sorted(glob.glob(os.path.join(d, 'steps', 'ep*.npz'))) + sorted(glob.glob(os.path.join(d, 'capture', 'ep*.npz')))
              for d in dirs]
    return load_holdout_files(groups, runs=list(dirs))


def load_holdout_files(groups, runs=None):
    """에피 파일 목록의 목록(런별) → 홀드아웃 dict. X 는 build_pool_knn 15열(…, theta, roll, pitch)."""
    X, E, RU, ST, EN, SRC = [], [], [], [], [], []
    k = 0
    for ri, files in enumerate(groups):
        for f in files:
            r = episode_rows_trim(f)
            if r is None:
                continue
            x, st, _ = r
            X.append(x[:, :15]); ST.append(st); E.append(np.full(len(x), k)); RU.append(np.full(len(x), ri)); SRC.append(f)
            m = re.search(r'ep(\d+)\.npz$', f); EN.append(np.full(len(x), int(m.group(1)) if m else k)); k += 1
    if not X:
        raise SystemExit(f'홀드아웃 에피가 없다: {runs}')
    return dict(X=np.concatenate(X), ep=np.concatenate(E), run=np.concatenate(RU), step=np.concatenate(ST),
                epnum=np.concatenate(EN), src=SRC, runs=list(runs or range(len(groups))))


def check_overlap(pool_path, H, allow=False):
    P = np.load(pool_path)
    src = set(os.path.realpath(str(s) if str(s).startswith('/') else os.path.join(ROOT, str(s))) for s in P['sources'])
    ov = [f for f in H['src'] if os.path.realpath(f) in src]
    if ov and not allow:
        raise SystemExit(f'홀드아웃 에피 {len(ov)}개가 풀 출처에 들어 있다(예: {ov[0]}) — 홀드아웃을 뺀 풀을 쓰거나 --allow-overlap')
    return len(ov)


# ───────────────────────────── 구현과 같은 2단 조건화 ─────────────────────────────
def make_env(pool, copula=None, extra=()):
    from cfgload import load_experiment
    from sim.surrogate import SurrogateEnv
    ov = [f'env.surrogate.pool={pool}', 'env.surrogate.theta=true', 'run.device=cpu', *extra]
    if copula is not None:
        ov.append('env.surrogate.theta_copula=[' + ','.join(str(float(c)) for c in copula) + ']')
    cwd = os.getcwd(); os.chdir(ROOT)
    try:
        exp = load_experiment(['configs/newenv_v5.yaml', 'configs/overlays/surrogate_knn_v3clean.yaml'], ov)
    finally:
        os.chdir(cwd)
    env = SurrogateEnv(exp.surrogate, exp.scenario.attack, exp.scenario.wind, ep_steps=300, seed=4242)
    return env, exp


def pit_z(env, H, workers=4, chunk=5000):
    """홀드아웃 행별 PIT z 와 2단 조건평균(구현 _theta_step 과 같은 트리·창·척도, 캐시 없이 정확 질의)."""
    from scipy.stats import norm
    from sim.surrogate import _pool_abits, _pool_windows
    c, T = env.cfg, env._TH
    HX, Hep = H['X'], H['ep']
    FH = env._feat(HX[:, 0], HX[:, 1], HX[:, 2], HX[:, 3], HX[:, 4], HX[:, 5], HX[:, 6], HX[:, 7], HX[:, 8],
                   HX[:, 11] if env._has_pat else None)
    if c.theta_abits_w > 0:
        FH = np.c_[FH, _pool_abits(HX[:, 5], Hep) * float(c.theta_abits_w)]
    WH = _pool_windows(HX[:, 9], HX[:, 10], Hep)
    K1 = min(int(c.theta_k1), len(T['TH'])); M = int(c.theta_m)
    z = np.empty(len(HX)); mu = np.empty(len(HX))
    y = HX[:, 12]
    for s in range(0, len(HX), chunk):
        sl = slice(s, s + chunk)
        _, nn = T['tree'].query(FH[sl], k=K1, workers=workers)
        d = (((T['WP'][nn] - WH[sl][:, None, :]) / T['WS']) ** 2).sum(2)
        sel = np.take_along_axis(nn, np.argpartition(d, M, axis=1)[:, :M], 1) if M < K1 else nn
        tv = T['TH'][sel]
        lo = (tv < y[sl, None]).sum(1); eq = (tv == y[sl, None]).sum(1)
        z[sl] = norm.ppf((lo + 0.5 * eq + 0.5) / (sel.shape[1] + 1)); mu[sl] = tv.mean(1)
    return z, mu


def acf_ep(z, ep, lags=LAGS):
    """전역 중심 ACF, 같은 에피 안의 쌍만."""
    mu = z.mean(); v = ((z - mu) ** 2).mean(); out = []
    for L in lags:
        same = ep[L:] == ep[:-L]
        out.append(float(((z[:-L][same] - mu) * (z[L:][same] - mu)).mean() / v))
    return np.array(out)


def copula_acf(p, lags=LAGS):
    sm2, w1, r1, w2, r2 = p
    return sm2 + w1 * r1 ** lags + w2 * r2 ** lags


def fit_copula(target, lags=LAGS):
    from scipy.optimize import least_squares
    res = least_squares(lambda p: np.r_[copula_acf(p, lags) - target, 10 * max(0.0, p[0] + p[1] + p[3] - 1.0)],
                        x0=[0.1, 0.4, 0.7, 0.4, 0.95], bounds=([0, 0, 0, 0, 0], [0.6, 1, 0.99, 1, 0.999]))
    p = res.x.copy()
    tot = p[0] + p[1] + p[3]
    if tot > 1.0:                                       # 벌점으로도 남은 초과분은 비례 축소(분산 합 ≤ 1)
        p[[0, 1, 3]] /= tot
    return tuple(float(x) for x in p), float(np.abs(copula_acf(p, lags) - target).max())


def write_pool(pool, cop, meta):
    d = np.load(pool)
    arrs = {k: d[k] for k in d.files if k not in ('theta_copula', 'theta_fit_meta')}
    arrs['theta_copula'] = np.asarray(cop, float); arrs['theta_fit_meta'] = json.dumps(meta, ensure_ascii=False)
    tmp = pool + '.tmp.npz'
    np.savez(tmp, **arrs)
    os.replace(tmp, pool)


def cmd_fit(a):
    H = load_holdout(a.hold)
    n_ov = check_overlap(a.pool, H, a.allow_overlap)
    env, _ = make_env(a.pool, copula=(0.0, 0.0, 0.0, 0.0, 0.0))   # 적합 단계는 코퓰러 값 무관(트리·창만 쓴다)
    z, mu = pit_z(env, H, workers=a.workers)
    y = H['X'][:, 12]
    fr = list(range(len(a.hold))) if not a.fit_runs else [int(i) for i in a.fit_runs]
    fm = np.isin(H['run'], fr)
    target = acf_ep(z[fm], H['ep'][fm])
    cop, err = fit_copula(target)
    r2 = 1 - ((y - mu) ** 2).mean() / y.var()
    print(f'[fit] 홀드아웃 {len(np.unique(H["ep"]))}에피 {len(y)}행 (풀 겹침 {n_ov}) · 적합 런 {fr} {int(fm.sum())}행')
    print(f'  2단 조건평균 R² {r2:.3f} · PIT z sd {z.std():.3f}')
    print(f'  코퓰러 (s_m², w1, ρ1, w2, ρ2) = ({", ".join(f"{x:.4f}" for x in cop)}) · 너겟 {max(0.0, 1 - cop[0] - cop[1] - cop[3]):.3f} · 최대 오차 {err:.4f}')
    print('  목표 ACF ' + ' '.join(f'{x:.3f}' for x in target))
    print('  모형 ACF ' + ' '.join(f'{x:.3f}' for x in copula_acf(cop)))
    meta = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), hold=[os.path.relpath(h, ROOT) if h.startswith(ROOT) else h for h in a.hold],
                fit_runs=fr, lags=LAGS.tolist(), target_acf=target.tolist(), model_acf=copula_acf(cop).tolist(), max_err=err,
                K1=int(env.cfg.theta_k1), M=int(env.cfg.theta_m), abw=float(env.cfg.theta_abits_w), n_rows=int(fm.sum()),
                r2_cond=float(r2), pit_sd=float(z.std()), method='PIT z ACF LSQ (P2S p6 32a)')
    if a.write:
        write_pool(a.pool, cop, meta)
        print(f'  → {a.pool} 에 theta_copula·theta_fit_meta 저장')
    return cop, meta


# ───────────────────────────── 관문 (강제 1:1 재생 + 지표) ─────────────────────────────
def plan_of(z):
    from env.attack import AttackPlan
    d = np.asarray(z['delta_plan'], float); act = d > 0
    bs = np.full(len(d), -1, int); cur = -1
    for i in range(len(d)):
        if act[i]:
            if cur < 0: cur = i
            bs[i] = cur
        else:
            cur = -1
    return AttackPlan(n=len(d), delta=d, active=act, bstart=bs, events=[], cls='v5')


def replay(env, H, log=print):
    """홀드아웃 에피를 surrogate 로 강제 재생 → 행별 (v, g, θ, 공격 라벨). 행동열은 Isaac 의 prev_action."""
    n = len(H['X']); V = np.zeros(n); G = np.zeros(n); TH = np.zeros(n); A = np.zeros(n)
    for i, e in enumerate(np.unique(H['ep'])):
        m = np.flatnonzero(H['ep'] == e)
        z = np.load(H['src'][e], allow_pickle=True)
        steps = H['step'][m]; amap = dict(zip(steps, H['X'][m, 5].astype(int)))
        env.reset(plan=plan_of(z), ws=float(z['wind_speed']))
        env.pat = PATS.index(str(z['pattern'])) if str(z['pattern']) in PATS else 0
        out = {}
        for t in range(0, int(steps.max()) + 1):
            env.t = t
            v, g, at = env.nis(int(amap.get(t, 0)))
            out[t] = (v, g, env.theta, at)
        R = np.array([out[s] for s in steps], float)
        V[m], G[m], TH[m], A[m] = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        if log and i % 200 == 0:
            log(f'  재생 {i}/{len(np.unique(H["ep"]))}')
    return V, G, TH, A


def freeze(th, a, ep, k=3):
    """P2′ 동결 규칙(env/reward.py 와 같은 정의): prev_action 전환 행부터 k 행은 전환 직전 θ."""
    fz = th.copy()
    sw = np.flatnonzero(np.r_[False, np.diff(a) != 0] & np.r_[False, ep[1:] == ep[:-1]])
    for s in sw:
        for j in range(s, min(len(th), s + k)):
            if ep[j] != ep[s]: break
            fz[j] = fz[s - 1]
    return fz


def acf_ep_demean(th, ep, L):
    out = []
    for e in np.unique(ep):
        x = th[ep == e]
        if len(x) <= L + 5: continue
        x = x - x.mean(); d = (x * x).sum()
        if d > 0: out.append((x[:-L] * x[L:]).sum() / d)
    return float(np.mean(out))


def theta_metrics(th, g, H, gamma=0.97, lam=4.0, theta0=0.1, late_ep=100):
    a = H['X'][:, 5].astype(int); atk = H['X'][:, 0] > 0; ep = H['ep']
    cls = dict(ct=~atk & (a == 0), ch=~atk & (a == 1), at=atk & (a == 0), ah=atk & (a == 1))
    lg = np.log(np.asarray(g) + 1e-6)
    ph = -lam / theta0 * freeze(np.asarray(th, float), a, ep)
    i = np.flatnonzero(np.r_[ep[3:] == ep[:-3], np.zeros(3, bool)])
    g3 = gamma ** 3 * ph[i + 3] - ph[i]; ad = a[np.minimum(i + 1, len(a) - 1)]; at = atk[i]; late = H['epnum'][i] >= late_ep
    if not late.any():
        late = np.ones(len(i), bool)
    g3c = dict(all=np.ones(len(i), bool), ct=~at & (ad == 0), ch=~at & (ad == 1), at=at & (ad == 0), ah=at & (ad == 1))
    return dict(mean={k: float(th[m].mean()) if m.any() else float('nan') for k, m in cls.items()},
                acf=[acf_ep_demean(th, ep, L) for L in (1, 3, 10)],
                corr=[float(np.corrcoef(th, lg)[0, 1]), float(np.corrcoef(th[~atk], lg[~atk])[0, 1]),
                      float(np.corrcoef(th[atk], lg[atk])[0, 1]) if atk.sum() > 2 else float('nan')],
                g3={k: float(np.std(g3[late & m])) if (late & m).any() else float('nan') for k, m in g3c.items()})


def fmt_metrics(tag, r):
    mm = ' '.join(f'{k} {v:.3f}' for k, v in r['mean'].items())
    return (f'{tag:10s} θ평균 {mm} | ACF ρ1/ρ3/ρ10 {r["acf"][0]:.2f}/{r["acf"][1]:.2f}/{r["acf"][2]:.2f} | '
            f'corr 전체/평시/공격 {r["corr"][0]:.3f}/{r["corr"][1]:.3f}/{r["corr"][2]:.3f} | G3frz3 ' +
            ' '.join(f'{k} {v:.2f}' for k, v in r['g3'].items()))


def cmd_gate(a):
    H = load_holdout(a.hold)
    n_ov = check_overlap(a.pool, H, a.allow_overlap)
    env, _ = make_env(a.pool, copula=a.copula)
    print(f'[gate] 홀드아웃 {len(np.unique(H["ep"]))}에피 {len(H["X"])}행 (풀 겹침 {n_ov}) · 코퓰러 {tuple(round(x, 4) for x in env._theta_cop)} ({env.theta_copula_src})')
    V, G, TH, A = replay(env, H)
    RI = theta_metrics(H['X'][:, 12], H['X'][:, 10], H, a.gamma)
    RS = theta_metrics(TH, G, H, a.gamma)
    lab = float(np.mean((A > 0) == (H['X'][:, 0] > 0)))
    print(fmt_metrics('Isaac', RI)); print(fmt_metrics('surrogate', RS)); print(f'  공격 라벨 일치율 {lab:.4f}')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(isaac=RI, surrogate=RS, label_agree=lab, copula=list(env._theta_cop), copula_src=env.theta_copula_src,
                           hold=a.hold, pool=a.pool), f, ensure_ascii=False, indent=1)
    return RI, RS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for nm in ('fit', 'gate'):
        p = sub.add_parser(nm)
        p.add_argument('pool'); p.add_argument('--hold', nargs='+', required=True)
        p.add_argument('--allow-overlap', action='store_true'); p.add_argument('--workers', type=int, default=4)
    sub.choices['fit'].add_argument('--fit-runs', nargs='*', default=None, help='적합 목표 ACF 에 쓸 --hold 인덱스(기본 전부)')
    sub.choices['fit'].add_argument('--write', action='store_true')
    sub.choices['gate'].add_argument('--copula', nargs=5, type=float, default=None)
    sub.choices['gate'].add_argument('--gamma', type=float, default=0.97, help='G3 의 γ (P2S 보고 기준 0.97)')
    sub.choices['gate'].add_argument('--json', default=None)
    a = ap.parse_args(argv)
    return cmd_fit(a) if a.cmd == 'fit' else cmd_gate(a)


if __name__ == '__main__':
    main()
