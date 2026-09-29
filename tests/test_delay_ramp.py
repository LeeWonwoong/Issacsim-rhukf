"""★09-29 증가형 지연 비용(reward.c_d_ref) — 기본 0 은 상수 c_d 와 비트 동일, 켜면 c_d·min(d, cap)/ref (d = attack_delay+1)."""
import random
from env.reward import RewardConfig, RewardTracker


def _run(rc, seq):
    t = RewardTracker(rc)
    return [t.step(pa, atk, d) for pa, atk, d in seq]


def test_default_is_constant():
    random.seed(0); seq = []; run = 0
    for _ in range(2000):
        atk = random.random() < 0.3; run = run + 1 if atk else 0
        seq.append((random.randint(0, 1), atk, max(run - 1, 0)))
    for kw in (dict(), dict(alive=1, c_fa=1, c_d=1, bonus=3), dict(alive=1, c_fa=1.5, c_d=1.5, bonus=4)):
        assert _run(RewardConfig(mode='cost', **kw), seq) == _run(RewardConfig(mode='cost', c_d_ref=0.0, c_d_cap=6, **kw), seq)


def test_ramp_values():
    t = RewardTracker(RewardConfig(mode='cost', alive=1, c_fa=1.5, c_d=1.5, bonus=4, c_d_ref=3, c_d_cap=6))
    pen = [round(1 - t.step(0, True, k), 9) for k in range(9)]
    assert pen == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0]
    t = RewardTracker(RewardConfig(mode='cost', alive=1, c_fa=1, c_d=1, bonus=3, c_d_ref=3, c_d_cap=6))
    assert round(t.step(1, False, 0), 9) == 0.0          # 오경보는 영향 없음
    assert round(t.step(1, True, 7), 9) == 4.0           # 탐지 보너스도 그대로
