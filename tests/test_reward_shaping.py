"""P2′ 자세 퍼텐셜 성형(env/reward.py) 단위 시험 — CPU, 수 초.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_reward_shaping.py

  1) shape_tilt=0 이면 7f77609 의 RewardTracker 와 비트 동일(roll·pitch 를 넘겨도)
  2) 망원 항등식: 할인 성형 리턴 − 무성형 리턴 = γ^T Φ_T − Φ_0 (terminal 이면 Φ_T=0 → −Φ_0), 첫 행 F=0
  3) 동결 규칙: 전환 행부터 shape_freeze 행은 직전 유효 θ̂, 동결 중 재전환은 그 시점 유효값으로 재시작
  4) n-step(실제 TensorReplayBuffer) 합이 γⁿΦ_{t+n} − Φ_t 로 망원
  5) 표형 MDP(확장 상태 = 공격·모드·유효 θ 수준·동결 카운터): 가치반복 최적정책이 λ 와 무관, Q′ = Q − Φ(s)
     — 보상은 RewardTracker 자체로 계산(상태를 심어 한 스텝 호출)
  6) cfgload 검증: γ 자동 채움·불일치 거부·surrogate θ 필수
"""
import itertools
import math
import os
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.reward import RewardConfig, RewardTracker   # noqa: E402

G = 0.97


def _ref_tracker_cls():
    src = subprocess.check_output(['git', 'show', '7f77609:env/reward.py'], cwd=ROOT).decode()
    m = types.ModuleType('reward_ref'); sys.modules['reward_ref'] = m      # dataclass 는 모듈 등록이 필요
    exec(compile(src, 'reward_ref', 'exec'), m.__dict__)
    return m.RewardConfig, m.RewardTracker


def _rand_seq(rng, n):
    atk = np.zeros(n, bool)
    for s in rng.integers(0, n, 3):
        atk[s:s + int(rng.integers(3, 20))] = True
    acts = (rng.random(n) < 0.3).astype(int)
    return [(int(a), bool(k), int(rng.integers(0, 8)), float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1)))
            for a, k in zip(acts, atk)]


@pytest.mark.parametrize('mode', ['cost', 'label4'])
def test_default_bit_identical_to_ref(mode):
    RC0, RT0 = _ref_tracker_cls()
    kw = dict(mode=mode, scale=0.7, terminal_penalty=3.0, alive=1.0, c_fa=1.5, c_d=1.5, bonus=4.0, fp_escalate=(mode == 'label4'))
    rng = np.random.default_rng(0)
    for ep in range(20):
        seq = _rand_seq(rng, 120)
        a, b = RT0(RC0(**kw)), RT0(RC0(**kw))
        n = RewardTracker(RewardConfig(**kw)); n2 = RewardTracker(RewardConfig(**kw))
        for i, (act, atk, d, ro, pi) in enumerate(seq):
            term = (i == len(seq) - 1) and ep % 2 == 0
            r_ref = a.step(act, atk, d, terminated=term)
            assert r_ref == n.step(act, atk, d, terminated=term)                       # 구 호출 형식
            assert r_ref == n2.step(act, atk, d, terminated=term, roll=ro, pitch=pi)    # roll·pitch 를 넘겨도 무시
            assert n2.last_rG == r_ref and n2.last_F == 0.0


def _run(rc, seq, term_last):
    tr = RewardTracker(rc); rs, Fs, phis = [], [], []
    for i, (act, atk, d, ro, pi) in enumerate(seq):
        r = tr.step(act, atk, d, terminated=(term_last and i == len(seq) - 1), roll=ro, pitch=pi)
        rs.append(r); Fs.append(tr.last_F); phis.append(tr.last_phi)
        assert r == tr.last_rG + tr.last_F
    return np.array(rs), np.array(Fs), np.array(phis)


@pytest.mark.parametrize('term_last', [True, False])
def test_telescoping_identity(term_last):
    rng = np.random.default_rng(1)
    kw = dict(mode='cost', alive=1.0, c_fa=1.5, c_d=1.5, bonus=4.0, scale=1.0)
    for _ in range(10):
        seq = _rand_seq(rng, 90)
        r0, _, _ = _run(RewardConfig(**kw), seq, term_last)
        r4, F, phi = _run(RewardConfig(shape_tilt=4.0, shape_gamma=G, **kw), seq, term_last)
        assert F[0] == 0.0                                             # 첫 호출 행: Φ_prev 초기화만(전이 없음)
        # 전이 k=1..T (행 0 은 전이를 만들지 않는다): Σ γ^{k−1} (r4_k − r0_k) = γ^T Φ_T − Φ_0
        T = len(seq) - 1
        disc = G ** np.arange(T)
        lhs = float((disc * (r4[1:] - r0[1:])).sum())
        rhs = (G ** T) * phi[-1] - phi[0]
        assert math.isclose(lhs, rhs, rel_tol=1e-10, abs_tol=1e-9), (lhs, rhs)
        if term_last:
            assert phi[-1] == 0.0 and math.isclose(lhs, -phi[0], rel_tol=1e-10, abs_tol=1e-9)
        np.testing.assert_allclose(r4 - r0, F, atol=1e-12)            # r^G 는 성형과 무관


def test_freeze_rule():
    rc = RewardConfig(mode='cost', shape_tilt=1.0, tilt_ref=1.0, shape_freeze=3, shape_gamma=G)
    tr = RewardTracker(rc)
    acts = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    th = [0.10, 0.11, 0.50, 0.51, 0.52, 0.53, 0.54, 0.20, 0.21, 0.60, 0.61, 0.62, 0.63, 0.64]
    # 행 2 전환 → 2,3,4 = θ̂_1(0.11); 행 5 부터 원값. 행 7 전환 → 7,8 = θ̂_6(0.54); 행 9 재전환(동결 중) → 9,10,11 = 유효 θ̂_8(0.54)
    exp = [0.10, 0.11, 0.11, 0.11, 0.11, 0.53, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.63, 0.64]
    got = []
    for a, t in zip(acts, th):
        tr.step(a, False, 0, roll=t, pitch=0.0); got.append(tr.last_theta)
    np.testing.assert_allclose(got, exp)
    # pitch 도 쓴다(θ̂ = √(roll²+pitch²)), 첫 행은 전환이어도 동결 대상 아님
    tr.reset(); tr.step(1, False, 0, roll=0.3, pitch=0.4)
    assert math.isclose(tr.last_theta, 0.5) and math.isclose(tr.last_phi, -0.5)
    # 동결 0 이면 원값
    rc0 = RewardConfig(mode='cost', shape_tilt=1.0, tilt_ref=1.0, shape_freeze=0, shape_gamma=G); tr0 = RewardTracker(rc0)
    np.testing.assert_allclose([(tr0.step(a, False, 0, roll=t), tr0.last_theta)[1] for a, t in zip(acts, th)], th)


def test_missing_theta_and_gamma_raise():
    with pytest.raises(ValueError):
        RewardTracker(RewardConfig(mode='cost', shape_tilt=4.0, shape_gamma=G)).step(0, False, 0)          # θ 없음
    with pytest.raises(ValueError):
        RewardTracker(RewardConfig(mode='cost', shape_tilt=4.0)).step(0, False, 0, roll=0.1, pitch=0.0)    # γ 미설정
    with pytest.raises(ValueError):
        RewardConfig(shape_tilt=-1.0)


def test_nstep_buffer_telescopes():
    import torch
    from rl.memory import TensorReplayBuffer
    n = 3
    cfg = types.SimpleNamespace(use_n_step=True, n_step_size=n, gamma=G, use_per=False)
    rng = np.random.default_rng(2)
    seq = _rand_seq(rng, 40)
    kw = dict(mode='cost', alive=1.0, c_fa=1.5, c_d=1.5, bonus=4.0)
    r0, _, _ = _run(RewardConfig(**kw), seq, True)
    r4, _, phi = _run(RewardConfig(shape_tilt=4.0, shape_gamma=G, **kw), seq, True)
    out = []
    for rr in (r0, r4):
        buf = TensorReplayBuffer(200, 1, 'cpu', cfg)
        for k in range(1, len(seq)):                                   # 전이 (s_{k−1}, a, r_k, s_k), 마지막이 terminal
            buf.push(np.array([k - 1.0]), 0, float(rr[k]), np.array([float(k)]), k == len(seq) - 1)
        out.append(buf.R[:buf.count].cpu().numpy().astype(float) if hasattr(buf, 'R') else None)
    assert out[0] is not None
    d = out[1] - out[0]
    T = len(seq) - 1
    for j in range(T):                                                 # 저장 j = 전이 시작 s_j, 끝 s_{min(j+n, T)}
        e = min(j + n, T)
        want = (G ** (e - j)) * phi[e] - phi[j]
        assert math.isclose(d[j], want, rel_tol=1e-5, abs_tol=2e-4), (j, d[j], want)


# ── 5) 표형 MDP: 최적정책 불변 ───────────────────────────────────────────────────────
LV = [0.04, 0.10, 0.20, 0.40]                                          # θ 수준(rad)
FRZ = 3


def _theta_dist(atk, a):
    """다음 행 θ 수준 분포 — 공격·hover 에 따라 기울기가 달라진다(행동 의존 = 성형이 결정 행에 붙는다)."""
    base = {(0, 0): [0.5, 0.35, 0.1, 0.05], (0, 1): [0.2, 0.4, 0.3, 0.1],
            (1, 0): [0.05, 0.15, 0.4, 0.4], (1, 1): [0.1, 0.3, 0.4, 0.2]}
    return base[(atk, a)]


def _build(lam):
    rc = RewardConfig(mode='cost', alive=1.0, c_fa=1.5, c_d=1.2, bonus=0.0, terminal_penalty=5.0, scale=1.0,
                      shape_tilt=lam, tilt_ref=0.1, shape_freeze=FRZ, shape_gamma=G)
    S = list(itertools.product([0, 1], [0, 1], range(len(LV)), range(FRZ + 1)))   # (atk, 모드 m = 직전 행동, 유효 θ 수준, 동결 잔여)
    idx = {s: i for i, s in enumerate(S)}
    p_atk = {0: 0.15, 1: 0.8}; p_crash = 0.25
    trans = {}                                                         # (i, a) → [(p, r, j or None)]
    for s in S:
        atk, m, k, fz = s
        for a in (0, 1):
            outs = []
            for atk2 in (0, 1):
                pa = p_atk[atk] if atk2 else 1 - p_atk[atk]
                sw = a != m
                if sw and FRZ > 0:
                    th_next = [(1.0, k, FRZ - 1)]                       # 전환 행: 직전 유효 θ 로 동결 시작
                elif fz > 0:
                    th_next = [(1.0, k, fz - 1)]
                else:
                    th_next = [(q, k2, 0) for k2, q in enumerate(_theta_dist(atk2, a))]
                for q, k2, fz2 in th_next:
                    for crash in ((False, True) if (atk2 and a == 0) else (False,)):
                        pc = (p_crash if crash else 1 - p_crash) if (atk2 and a == 0) else 1.0
                        tr = RewardTracker(rc)                          # 상태 s 를 심고 한 스텝
                        tr._pprev_action = m; tr._prev_atk = bool(atk); tr._paid = True
                        if lam > 0:
                            tr._phi_prev = -lam * LV[k] / rc.tilt_ref; tr._th_prev = LV[k]
                            tr._frz_left = fz; tr._frz_val = LV[k]
                        r = tr.step(a, bool(atk2), 1, terminated=crash, roll=LV[k2 if not (sw or fz > 0) else k2], pitch=0.0)
                        if lam > 0:
                            assert math.isclose(tr.last_theta, LV[k2]), (s, a, tr.last_theta, LV[k2])
                        outs.append((pa * q * pc, r, None if crash else idx[(atk2, a, k2, fz2)]))
            trans[(idx[s], a)] = outs
    return S, trans


def _vi(S, trans, iters=3000):
    V = np.zeros(len(S))
    for _ in range(iters):
        Q = np.array([[sum(p * (r + (0.0 if j is None else G * V[j])) for p, r, j in trans[(i, a)]) for a in (0, 1)]
                      for i in range(len(S))])
        V2 = Q.max(1)
        if np.max(np.abs(V2 - V)) < 1e-12:
            break
        V = V2
    return Q


def test_tabular_policy_invariance():
    S, t0 = _build(0.0)
    Q0 = _vi(S, t0)
    for lam in (4.0, 8.0):
        _, tl = _build(lam)
        Ql = _vi(S, tl)
        phi = np.array([-lam * LV[s[2]] / 0.1 for s in S])
        np.testing.assert_allclose(Ql, Q0 - phi[:, None], atol=1e-7)   # Q′ = Q − Φ(s)
        gap = np.abs(Q0[:, 0] - Q0[:, 1])
        ok = gap > 1e-6
        assert ok.sum() > len(S) // 2
        assert np.array_equal(Q0[ok].argmax(1), Ql[ok].argmax(1))      # 최적정책 불변
    # 정책이 자명하지 않다(평시는 track, 공격 중은 hover 가 섞여 있다)
    pol = Q0.argmax(1)
    assert 0 < pol.sum() < len(S)


# ── 6) cfgload ───────────────────────────────────────────────────────────────────────
def test_cfgload_shaping_validation():
    os.chdir(ROOT)
    from cfgload import load_experiment
    e = load_experiment(['configs/isaac_v5_swirl.yaml'], ['run.device=cpu', 'env.kind=isaac', 'reward.shape_tilt=4'])   # Isaac: θ = cur_euler
    assert e.reward.shape_gamma == e.cfg.gamma and e.resolved['reward']['shape_gamma'] == e.cfg.gamma
    e0 = load_experiment(['configs/isaac_v5_swirl.yaml'], ['run.device=cpu'])
    assert e0.reward.shape_tilt == 0 and e0.reward.shape_gamma == 0 and 'shape_gamma' not in (e0.resolved.get('reward') or {})
    with pytest.raises(ValueError):
        load_experiment(['configs/isaac_v5_swirl.yaml'], ['run.device=cpu', 'env.kind=isaac', 'reward.shape_tilt=4', 'reward.shape_gamma=0.5'])
    with pytest.raises(ValueError):                                     # surrogate 는 θ 채널 필수
        load_experiment(['configs/newenv_v5.yaml'], ['run.device=cpu', 'reward.shape_tilt=4'])
    e2 = load_experiment(['configs/newenv_v5.yaml'], ['run.device=cpu', 'reward.shape_tilt=4', 'env.surrogate.theta=true'])
    assert e2.surrogate.theta and e2.reward.shape_gamma == e2.cfg.gamma


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            if k == 'test_default_bit_identical_to_ref':
                f('cost'); f('label4')
            elif k == 'test_telescoping_identity':
                f(True); f(False)
            else:
                f()
            print('ok', k, flush=True)
