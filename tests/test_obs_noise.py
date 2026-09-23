"""입력 인위 잡음(obs.noise_std, 09-23) 단위 시험 — 끔이면 비트 동일, 켜면 공통난수·σ·행동칸 불변·창 공유."""
import numpy as np

from env.observation import ObsBuilder, ObsSpec


def _feed(ob, n=60, key_off=0):
    rng = np.random.default_rng(3)
    out = []
    for t in range(n):
        s = ob.push(float(rng.gamma(2.0, 0.5)), float(rng.gamma(2.0, 0.5)), int(t % 7 == 0), key=(0, 5, t + key_off))
        out.append(None if s is None else s.copy())
    return out


def test_off_is_identical_to_keyless():
    a = _feed(ObsBuilder(ObsSpec()))
    b = ObsBuilder(ObsSpec())
    rng = np.random.default_rng(3)
    for t, sa in enumerate(a):
        sb = b.push(float(rng.gamma(2.0, 0.5)), float(rng.gamma(2.0, 0.5)), int(t % 7 == 0))   # 구 호출(키 없음)
        assert (sa is None) == (sb is None)
        if sa is not None:
            assert np.array_equal(sa, sb)


def test_common_random_numbers_and_sigma():
    sp = ObsSpec(noise_std=0.1, noise_seed=11)
    a, b = _feed(ObsBuilder(sp)), _feed(ObsBuilder(sp))
    for sa, sb in zip(a, b):
        if sa is not None:
            assert np.array_equal(sa, sb)          # 같은 (ep, step) → 같은 잡음
    clean = _feed(ObsBuilder(ObsSpec()))
    d = np.array([sa - sc for sa, sc in zip(a, clean) if sa is not None])
    assert np.allclose(d[:, 2::3], 0.0)            # 행동 칸(features 순서 vel,gyro,action)은 잡음 없음
    z = d[:, -3:-1].reshape(-1)                    # 최신 프레임의 vel·gyro 잡음
    assert 0.07 < float(z.std()) < 0.13


def test_frame_noise_shared_across_windows():
    sp = ObsSpec(noise_std=0.2)
    s = [x for x in _feed(ObsBuilder(sp), n=12) if x is not None]
    # s_t 의 최신 프레임 == s_{t+1} 의 직전 프레임 (프레임당 1회 뽑아 창에 저장)
    for s0, s1 in zip(s[:-1], s[1:]):
        assert np.array_equal(s0[-3:], s1[-6:-3])


def test_last_scaled_stays_clean():
    ob = ObsBuilder(ObsSpec(noise_std=0.5))
    ob.push(4.0, 9.0, 0, key=(0, 1, 1))
    spec = ObsSpec()
    assert ob.last_scaled == (spec.scale(4.0), spec.scale(9.0))
    assert ob.last_noise != (0.0, 0.0)
