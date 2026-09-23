"""모드 전환 비용 c_sw(09-23): 끔이면 기존 보상과 동일, 켜면 track↔hover 전환 스텝마다 −c_sw·scale, n_switch 계수."""
from env.reward import RewardConfig, RewardTracker

SEQ = [(0, False), (0, False), (1, False), (1, True), (0, True), (1, True), (1, False), (0, False)]


def _run(rc):
    tr = RewardTracker(rc); out = []
    for a, atk in SEQ:
        out.append(tr.step(a, atk, 0))
    return out, tr.n_switch


def test_off_identical():
    base, n0 = _run(RewardConfig(mode='cost', c_fa=1.5, c_d=1.5, bonus=4.0, alive=1.0))
    same, n1 = _run(RewardConfig(mode='cost', c_fa=1.5, c_d=1.5, bonus=4.0, alive=1.0, c_sw=0.0))
    assert base == same and n0 == n1 == 4


def test_switch_charged():
    base, _ = _run(RewardConfig(mode='cost', c_fa=1.5, c_d=1.5, bonus=4.0, alive=1.0, scale=2.0))
    sw, n = _run(RewardConfig(mode='cost', c_fa=1.5, c_d=1.5, bonus=4.0, alive=1.0, scale=2.0, c_sw=1.0))
    assert n == 4
    d = [b - s for b, s in zip(base, sw)]
    assert d == [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]
