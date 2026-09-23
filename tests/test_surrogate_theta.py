"""surrogate θ 채널 스모크 (CPU, ~1 분):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_surrogate_theta.py

  기존 풀(pool_knn_v2clean)의 출처 에피소드 일부로 knn_v4 풀을 임시 재생성(scripts/build_pool_knn.build) 한 뒤
  1) θ 를 켜도 NIS·공격 라벨·추락이 비트 동일(θ 전용 난수 스트림)
  2) θ 통계가 P2S(J32a) 보고 수준: 평시 track 평균이 풀 평균 ±25%, 공격 중 θ 가 평시보다 크다, 에피 내 ACF lag1 0.55–0.85
  3) train.run_surrogate: λ=0 이면 θ 켜/끔 학습 이력 동일, λ4 성형이면 reward(r^G)·reward_train·shape_F·theta_mean 기록
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
    assert np.corrcoef(th, np.log(R[:, 1] + 1e-6))[0, 1] > 0.15                  # P2S 0.33 (Isaac 0.33)


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
