"""surrogate θ 채널 스모크 (CPU, ~1 분):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_surrogate_theta.py

  기존 풀(pool_knn_v2clean)의 출처 에피소드 일부로 knn_v4 풀을 임시 재생성(scripts/build_pool_knn.build) 한 뒤
  1) θ 를 켜도 NIS·공격 라벨·추락이 비트 동일(θ 전용 난수 스트림)
  2) θ 통계가 P2S(J32a) 보고 수준: 평시 track 평균이 풀 평균 ±25%, 공격 중 θ 가 평시보다 크다, 에피 내 ACF lag1 0.55–0.85
  3) train.run_surrogate: λ=0 이면 θ 켜/끔 학습 이력 동일, λ4 성형이면 reward(r^G)·reward_train·shape_F·theta_mean 기록
  ── 09-24 리뷰 반영 ──
  4) θ 한 스텝을 독립 참조 구현(P2S p6 32a 알고리즘을 시험 안에 다시 씀)과 비트 비교 — 2단 조건화·코퓰러·난수 순서 고정
  5) 1단 이웃 캐시가 인스턴스 소유이고 키에 theta_k1 이 있다: 다른 인스턴스 이력·k1 변경이 θ 를 오염하지 않는다
  6) 코퓰러 출처 우선순위(설정 > 풀 저장값 > J32a+경고), scripts/fit_theta_copula.py fit/쓰기 왕복
원본 풀이 없는 서버(결과 폴더 없음)에서는 건너뛴다.
"""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
from cfgload import load_experiment          # noqa: E402
from sim.surrogate import SurrogateEnv       # noqa: E402

POOL_REL = 'results/claudecodefortest/pools/pool_knn_v2clean.npz'


def _main_root():
    """결과 폴더는 worktree 에 없을 수 있다 → 저장소 본체(git common dir 의 부모)도 본다."""
    cands = [ROOT]
    try:
        gd = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=ROOT).decode().strip()
        cands.append(os.path.dirname(os.path.abspath(os.path.join(ROOT, gd))))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(os.path.join(c, POOL_REL)):
            return c
    return None


MAIN = _main_root()
pytestmark = pytest.mark.skipif(MAIN is None, reason='원본 풀 없음(결과 폴더 없는 서버)')
_TMP = tempfile.mkdtemp(prefix='surr_theta_')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


@pytest.fixture(scope='module')
def pool():
    import build_pool_knn as B
    src = [str(s) for s in np.load(os.path.join(MAIN, POOL_REL))['sources']]
    files = [s if s.startswith('/') else os.path.join(MAIN, s) for s in src][::6]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 100:
        pytest.skip('원본 에피소드 파일 부족')
    out = os.path.join(_TMP, 'pool_v4.npz')
    B.build(out, files)
    z = np.load(out)
    assert str(z['format']) == 'knn_v4' and list(z['names'])[12:] == ['theta', 'roll', 'pitch']
    return out


def _exp(pool, *extra):
    return load_experiment(['configs/newenv_v5.yaml', 'configs/overlays/surrogate_knn_v3clean.yaml'],
                           [f'env.surrogate.pool={pool}', 'run.device=cpu', *extra])


def _roll(env, n_ep, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_ep):
        env.reset(); a = 0; e = []
        for _t in range(300):
            v, g, at = env.nis(a)
            e.append((v, g, float(at), a, np.nan if env.theta is None else env.theta, float(env.crashed)))
            if at: a = 1 if rng.random() < (0.85 if a else 0.5) else 0
            else: a = 1 if rng.random() < (0.6 if a else 0.03) else 0
            if env.step(): break
        out.append(np.array(e, float))
    return out


def test_nis_bit_identical_with_theta(pool):
    e0, e1 = _exp(pool), _exp(pool, 'env.surrogate.theta=true')
    a = _roll(SurrogateEnv(e0.surrogate, e0.scenario.attack, e0.scenario.wind, 300, 777), 6)
    b = _roll(SurrogateEnv(e1.surrogate, e1.scenario.attack, e1.scenario.wind, 300, 777), 6)
    for x, y in zip(a, b):
        assert x.shape == y.shape and np.array_equal(x[:, [0, 1, 2, 3, 5]], y[:, [0, 1, 2, 3, 5]])
        assert np.all(np.isnan(x[:, 4])) and np.all(np.isfinite(y[:, 4])) and np.all(y[:, 4] >= 0)


def test_theta_stats_near_p2s(pool):
    z = np.load(pool); X = z['X']; th_p = X[:, 12]
    m_ct = (X[:, 0] == 0) & (X[:, 5] == 0)
    e1 = _exp(pool, 'env.surrogate.theta=true')
    eps = _roll(SurrogateEnv(e1.surrogate, e1.scenario.attack, e1.scenario.wind, 300, 4242), 30, seed=1)
    R = np.concatenate(eps)
    at, a, th = R[:, 2] > 0, R[:, 3] == 1, R[:, 4]
    ct = th[~at & ~a].mean()
    assert abs(ct / th_p[m_ct].mean() - 1) < 0.25, (ct, th_p[m_ct].mean())      # P2S: 평시 track 0.119 vs Isaac 0.121
    assert th[at].mean() > ct + 0.03                                             # P2S: 공격 0.17–0.18 vs 평시 0.12
    acf = [np.corrcoef(x[:-1, 4] - x[:, 4].mean(), x[1:, 4] - x[:, 4].mean())[0, 1] for x in eps if len(x) > 20]
    assert 0.55 < np.nanmean(acf) < 0.85, np.nanmean(acf)                        # P2S J32a ρ1 0.70 (Isaac 0.80)
    # ★09-24 리뷰 3: > 0.15 는 1단만 쓰는 방식(F0/F1: 0.19–0.22)도 통과했다 → 2단 NIS 창 조건화가 빠지면 잡히게 > 0.27
    assert np.corrcoef(th, np.log(R[:, 1] + 1e-6))[0, 1] > 0.27                  # P2S J32a 0.31–0.33 (Isaac 0.33)


def _train(pool, *extra):
    import train
    out = tempfile.mkdtemp(dir=_TMP)
    exp = _exp(pool, 'agent.type=adam', 'agent.batch=16', 'run.episodes=3', 'run.ep_steps=80', 'log.probe_every=0',
               'log.eval_n=0', f'run.outdir={out}', *extra)
    return train.run_surrogate(exp, log=lambda *a, **k: None)


def test_train_surrogate_shaping_smoke(pool):
    import json
    drop = lambda h: json.dumps([{k: v for k, v in r.items() if k not in ('sec', 'theta_mean')} for r in h], sort_keys=True, default=str)
    h0 = _train(pool)
    h1 = _train(pool, 'env.surrogate.theta=true')
    assert drop(h0) == drop(h1) and 'theta_mean' in h1[0] and 'reward_train' not in h1[0]   # λ=0: θ 는 학습에 무영향
    h4 = _train(pool, 'env.surrogate.theta=true', 'reward.shape_tilt=4')
    for r in h4:
        assert {'reward', 'reward_train', 'shape_F', 'theta_mean'} <= set(r)
        assert np.isfinite(r['reward_train']) and abs(r['reward_train'] - r['reward'] - r['shape_F']) < 1e-6
        assert r['shape_F'] != 0.0



# ══════════════════════════ 09-24 리뷰 반영 시험 ══════════════════════════
def _env(pool, *extra, seed=777):
    e = _exp(pool, 'env.surrogate.theta=true', *extra)
    return SurrogateEnv(e.surrogate, e.scenario.attack, e.scenario.wind, 300, seed)


def _ref_theta_steps(env, rec):
    """독립 참조: 기록된 (q, prev_action, v, g) 로 P2S p6 32a 알고리즘을 그대로 다시 돌린다(캐시 규칙 포함)."""
    import math
    from scipy.spatial import cKDTree
    from scipy.stats import norm
    Q = env.Q; X = Q['X']; ep = Q['_ep']; c = env.cfg
    comp = lambda x: np.minimum(np.log1p(np.sqrt(np.maximum(x, 0.0))), 4.0) / 4.0
    n = len(X); start = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
    first = np.zeros(n, int)
    for i, s0 in enumerate(start):
        first[s0:(start[i + 1] if i + 1 < len(start) else n)] = s0
    WP = np.zeros((n, 8))
    for j in range(4):
        idx = np.maximum(np.arange(n) - (3 - j), first)
        WP[:, j] = comp(X[idx, 9]); WP[:, 4 + j] = comp(X[idx, 10])
    WS = WP.std(0) + 1e-9
    B = np.zeros((n, 6)); a = X[:, 5]
    for L in range(1, 7):
        B[:, L - 1] = np.where(np.r_[np.zeros(L, bool), ep[L:] == ep[:-L]], np.r_[np.zeros(L), a[:-L]], 0.0)
    F = np.c_[env._F, B * c.theta_abits_w]
    tree = cKDTree(F, balanced_tree=False, compact_nodes=False)
    TH = X[:, 12]; K1 = min(int(c.theta_k1), n); M = int(c.theta_m)
    sm2, w1, r1, w2, r2 = env._theta_cop; nug2 = max(0.0, 1 - sm2 - w1 - w2)
    out = []
    for ep_idx, steps in rec:
        rt = np.random.default_rng([env.seed0 & 0xFFFFFFFF, ep_idx + 1, 0x7E7A])
        tm = math.sqrt(sm2) * rt.normal(); t1 = rt.normal(); t2 = rt.normal()
        ah = [0] * 6; wv = []; wg = []; cache = {}
        for (q, pa, v, g) in steps:
            qq = np.r_[q, np.array(ah[::-1], float) * c.theta_abits_w]; ah = ah[1:] + [int(pa)]
            key = (K1,) + tuple(np.round(qq, 1))
            nb = cache.get(key)
            if nb is None:
                nb = tree.query(qq, k=K1, workers=1)[1]; cache[key] = nb
            wv = (wv + [float(comp(v))])[-4:]; wg = (wg + [float(comp(g))])[-4:]
            w = np.array([wv[0]] * (4 - len(wv)) + wv + [wg[0]] * (4 - len(wg)) + wg)
            d = (((WP[nb] - w) / WS) ** 2).sum(1)
            S = np.sort(TH[nb[np.argpartition(d, M)[:M]]])
            t1 = r1 * t1 + math.sqrt(1 - r1 * r1) * rt.normal()
            t2 = r2 * t2 + math.sqrt(1 - r2 * r2) * rt.normal()
            z = tm + math.sqrt(nug2) * rt.normal() + math.sqrt(w1) * t1 + math.sqrt(w2) * t2
            u = norm.cdf(z)
            out.append(float(np.interp(u, (np.arange(len(S)) + 0.5) / len(S), S)))
    return np.array(out)


def test_theta_step_matches_independent_reference(pool):
    import sim.surrogate as SS
    env = _env(pool)
    rec = []; got = []
    orig = env._theta_step

    def wrap(q, pa, v, g):
        rec[-1][1].append((np.array(q, float).copy(), int(pa), float(v), float(g)))
        th = orig(q, pa, v, g); got.append(th); return th
    env._theta_step = wrap
    rng = np.random.default_rng(3)
    for _ in range(3):
        env.reset(); rec.append((env.ep_idx, [])); a = 0
        for _t in range(60):
            env.nis(a); a = int(rng.random() < 0.2)
            if env.step(): break
    ref = _ref_theta_steps(env, rec)
    # 참조의 분위 보간이 구현 _iq 와 같은 격자인지부터 확인(아니면 이 비교 자체가 무의미)
    S = np.sort(np.random.default_rng(0).random(32)); u = 0.37
    assert abs(SS._iq(S, u) - float(np.interp(u, (np.arange(32) + 0.5) / 32, S))) < 1e-12
    assert len(got) > 100
    np.testing.assert_allclose(np.array(got), ref, rtol=0, atol=1e-12)


def _theta_trace(env, n_ep=2, seed=5):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_ep):
        env.reset(); a = 0
        for _t in range(80):
            env.nis(a); out.append(env.theta); a = int(rng.random() < 0.2)
            if env.step(): break
    return np.array(out)


def test_theta_cache_instance_scoped_and_keyed_by_k1(pool):
    base = _theta_trace(_env(pool))
    other = _env(pool, 'scenario.wind.kind=none', seed=999)          # 다른 인스턴스가 먼저 많이 돌아도
    _theta_trace(other, n_ep=4)
    again = _theta_trace(_env(pool))
    assert np.array_equal(base, again)                                # 같은 시드·설정 θ 불변
    e64 = _env(pool, 'env.surrogate.theta_k1=64')
    t64 = _theta_trace(e64)
    assert all(len(v) == 64 for v in e64._theta_c1.values()) and len(e64._theta_c1) > 0
    assert not np.array_equal(base, t64)                              # k1 변경이 실제로 반영된다


def test_copula_source_priority_and_fit_roundtrip(pool, tmp_path):
    import warnings
    import fit_theta_copula as FT
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        import sim.surrogate as SS
        SS._POOL_CACHE.clear()
        e = _env(pool)
        assert e.theta_copula_src == 'J32a' and e._theta_cop == SS.THETA_COPULA_J32A
        assert any('theta_copula' in str(x.message) for x in w)
    e2 = _env(pool, 'env.surrogate.theta_copula=[0.1,0.2,0.3,0.4,0.5]')
    assert e2.theta_copula_src == 'config' and e2._theta_cop == (0.1, 0.2, 0.3, 0.4, 0.5)
    # 풀에 없는 원본 에피(홀드아웃)로 적합 → 풀에 쓰기 → 로더가 풀 값을 쓴다
    src = [str(s) for s in np.load(os.path.join(MAIN, POOL_REL))['sources']]
    files = [s if s.startswith('/') else os.path.join(MAIN, s) for s in src][3::6][:60]
    files = [f for f in files if os.path.exists(f)]
    H = FT.load_holdout_files([files])
    assert FT.check_overlap(pool, H) == 0
    z, mu = FT.pit_z(e, H, workers=2)
    assert np.all(np.isfinite(z)) and 0.8 < z.std() < 1.25                       # PIT 가 대략 표준정규
    tgt = FT.acf_ep(z, H['ep'])
    cop, err = FT.fit_copula(tgt)
    assert err < 0.08 and cop[0] + cop[1] + cop[3] <= 1.0 + 1e-9 and tgt[0] > 0.4
    p2 = str(tmp_path / 'pool_fit.npz'); shutil.copy(pool, p2)
    FT.write_pool(p2, cop, dict(test=True))
    e3 = _env(p2)
    assert e3.theta_copula_src == 'pool' and np.allclose(e3._theta_cop, cop)
    assert e3.Q['_theta_fit_meta'] and _env(p2, 'env.surrogate.theta_copula=[0.1,0.2,0.3,0.4,0.5]').theta_copula_src == 'config'
