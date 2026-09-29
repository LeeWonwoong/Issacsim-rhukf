"""★09-30 비용가중 분류기 베이스라인(agent.type=clf): 보상→라벨 복원·임계·설정 거부."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfgload import load_experiment
from env.reward import RewardTracker

BASE = ['configs/newenv_vfinal.yaml', 'configs/overlays/surrogate_knn_v6.yaml']
SETS = ['agent.type=clf', 'run.device=cpu', 'agent.hidden=[24,24]', 'obs.window=6']


def _agent(extra=()):
    from rl.agent_clf import OnlineCostClassifierAgent
    e = load_experiment(BASE, SETS + list(extra))
    return OnlineCostClassifierAgent(e.cfg), e


@pytest.mark.parametrize('extra', [(), ('reward.alive=1', 'reward.c_fa=1', 'reward.c_d=1', 'reward.bonus=3'),
                                   ('reward.alive=0', 'reward.c_fa=1', 'reward.c_d=2', 'reward.bonus=4'),
                                   ('reward.c_d_ref=3',)])
def test_label_recovery_matches_truth(extra):
    ag, e = _agent(extra)
    tr = RewardTracker(e.reward)
    import random
    random.seed(1); d = 0
    for _ in range(3000):
        atk = random.random() < 0.3; d = d + 1 if atk else 0; a = random.randint(0, 1)
        r = tr.step(a, atk, max(d - 1, 0))
        assert ag.label_of(a, r) == int(atk)


def test_thresholds_and_param_count():
    ag, e = _agent(('agent.clf.threshold=bonus',))
    assert abs(ag._th_stay - 0.5) < 1e-12 and abs(ag._th_enter - 1.5 / 7.0) < 1e-12
    from rl.agent_adam import OnlineAdamAgent
    e2 = load_experiment(BASE, ['agent.type=adam', 'run.device=cpu', 'agent.hidden=[24,24]', 'obs.window=6'])
    assert sum(p.numel() for p in ag.net.parameters()) == sum(p.numel() for p in OnlineAdamAgent(e2.cfg).net.parameters())


def test_rejects_non_cost_reward():
    with pytest.raises(ValueError):
        _agent(('reward.terminal_penalty=5',))
