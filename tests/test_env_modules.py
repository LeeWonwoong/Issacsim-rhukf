"""공용 모듈 단위 테스트: python3 -m pytest -q tests/  (또는 python3 tests/test_env_modules.py)"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.observation import ObsSpec, ObsBuilder, compressed_to_raw
from env.reward import RewardConfig, RewardTracker, label4_reward
from env.attack import AttackConfig, sample_attack


def test_obs_roundtrip_and_window():
    sp = ObsSpec(clip=4.0, div=4.0, window=4)
    for x in [0.0, 0.3, 1.7, 3.99]:
        assert np.float32(sp.scale(compressed_to_raw(x))) == np.float32(x / 4.0)   # 역변환: float64 1e-16 오차, 관측(float32)에선 동일
    assert sp.scale(1e9) == 1.0                                   # 클립 후 [0,1]
    ob = ObsBuilder(sp)
    outs = [ob.push(1.0, 2.0, a) for a in (0, 1, 0, 1)]
    assert outs[:3] == [None, None, None] and outs[3].shape == (12,)
    assert ObsSpec(features=['gyro', 'action']).dim == 8


def test_label4_matches_legacy_table():
    rc = RewardConfig(fn_onset_mult=1.0)
    assert label4_reward(rc, 0, False, 0) == 0.5 and label4_reward(rc, 1, False, 0) == -0.7
    assert label4_reward(rc, 1, True, 3) == 1.0
    assert [round(label4_reward(rc, 0, True, d), 2) for d in range(7)] == [-0.4, -0.75, -1.1, -1.45, -1.8, -2.15, -2.15]
    rc2 = RewardConfig(fn_onset_mult=1.8)
    assert round(label4_reward(rc2, 0, True, 1), 3) == round(-0.75 * 1.8, 3) and label4_reward(rc2, 0, True, 0) == -0.4


def test_cost_bonus_once_per_burst_and_scale():
    rc = RewardConfig(mode='cost', c_fa=0.7, c_d=0.3, bonus=1.0, alive=0.0, scale=2.0)
    tr = RewardTracker(rc)
    seq = [(0, False), (1, False), (0, True), (1, True), (1, True), (0, True), (1, True), (0, False), (1, True), (1, True)]
    got = [tr.step(a, atk, 0) for a, atk in seq]
    # 평시 track 0 · 평시 hover −0.7 · 공격 track −0.3 · 첫 hover +1 · hover 유지 0 · 재발 track −0.3 · 재진입 0(같은 버스트)
    # · 버스트 종료 후 새 버스트(이미 hover 중이어도) 첫 스텝 +1 · 유지 0   (모두 ×2)
    assert got == [0.0, -1.4, -0.6, 2.0, 0.0, -0.6, 0.0, 0.0, 2.0, 0.0], got


def test_cost_alive_and_terminal():
    tr = RewardTracker(RewardConfig(mode='cost', alive=0.5, terminal_penalty=15))
    assert tr.step(0, False, 0) == 0.5 and tr.step(0, False, 0, terminated=True) == 0.5 - 15


def test_v5_attack_ranges():
    cfg = AttackConfig(p_attack=1.0)
    for s in range(200):
        p = sample_attack(np.random.default_rng(s), cfg, 300)
        on = np.flatnonzero(p.active)
        assert p.has_attack and 60 <= on[0] <= 200 and 25 <= len(on) <= 40
        assert 0.10 <= p.dmax <= 0.84 and np.ptp(p.delta[on]) == 0


def test_profile_prefix_shared_then_diverges():
    cls = [dict(name='wb', weight=1, delta=[0.3, 0.3], kind='burst', dur=[6, 6]),
           dict(name='wp', weight=1, delta=[0.3, 0.3], kind='persistent', dur=[20, 20], grow=[2.0, 2.0], grow_steps=6)]
    cfg = AttackConfig(p_attack=1.0, family='profile', classes=cls)
    seen = {}
    for s in range(50):
        p = sample_attack(np.random.default_rng(s), cfg, 300)
        seen[p.cls] = p.delta[p.active]
    b, q = seen['wb'], seen['wp']
    assert np.allclose(b[:4], q[:4])                     # prefix 공통 (t0–t3)
    assert len(b) == 6 and len(q) == 20
    assert b[-1] < b[3] and q[-1] > q[3]                 # burst 는 사그라들고 persistent 는 커진다
    assert math.isclose(q[-1], 0.6)




def test_multi_event_profile_and_grow_range():
    cls = [dict(name='wp', weight=1, delta=[0.1, 0.3], kind='persistent', dur=[15, 30], grow=[1.8, 2.2], grow_steps=[3, 6]),
           dict(name='wb', weight=1, delta=[0.1, 0.3], kind='burst', dur=[3, 6])]
    cfg = AttackConfig(p_attack=1.0, family='profile', classes=cls, events=(1, 3), event_gap=(30, 60))
    ks, gss = set(), set()
    for s in range(300):
        p = sample_attack(np.random.default_rng(s), cfg, 300)
        ev = p.events; ks.add(len(ev))
        assert 1 <= len(ev) <= 3 and ev[0]['start'] >= 60
        for a, b in zip(ev, ev[1:]):
            assert b['start'] - a['end'] >= 30                    # 간격 ≥ 30, 겹침 없음
        for e_ in ev:
            assert e_['end'] <= 295 and p.cls_at(e_['start']) == e_['cls']
            if e_['cls'] == 'wp': gss.add(e_['grow_steps'])
    assert ks == {1, 2, 3} and gss == {3, 4, 5, 6}


def test_window_policy_per_event():
    from env.capture import make_policy
    from env.attack import _empty
    p = _empty(300); p.active[100:110] = True; p.active[160:170] = True
    p.events = [dict(start=100, end=110), dict(start=160, end=170)]
    pol = make_policy('window:2:3', np.random.default_rng(0))
    a = np.array([pol.act(t, p) for t in range(300)])
    assert a[102:113].all() and not a[113:162].any() and a[162:173].all() and not a[173:].any()


def test_eps_z_greedy():
    """εz-greedy: 한 번의 탐험 결정이 n~zeta(2)(상한 30) 스텝 유지, eps=0 평가는 무시, 에피소드 경계에서 초기화."""
    import sys; sys.path.insert(0, '.')
    from cfgload import load_experiment
    import train
    exp = load_experiment(['configs/newenv.yaml', 'configs/overlays/surrogate_knn.yaml'],
                          ['agent.type=adam', 'run.device=cpu', 'agent.eps.z_mu=2', 'agent.eps.z_cap=30'])
    ag = train.make_agent(exp); s = np.zeros(12, np.float32)
    starts = 0
    for _ in range(20000):
        starts += ag._z_left == 0; ag.act(s, 1.0)
    assert 2.8 < 20000 / starts < 3.25                 # 이론 평균 3.03
    ag._z_left, ag._z_a = 5, 1
    ag.act(s, 0.0); assert ag._z_left == 5              # eps=0 → 지속 소비 안 함
    ag.end_episode(0.0, 1); assert ag._z_left == 0


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
