"""★09-30 EKF-TD 해석 야코비안(per_sample_jacobian)이 이전 autograd.functional.jacobian 과 수치 동일하고, ekf_step 결과(θ·P·loss)도 같은지."""
import os, sys, copy
import numpy as np, pytest, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfgload import load_experiment
from rl import rhukf_core as C
from rl.network import create_network_info

BASE = ['configs/newenv_vfinal.yaml', 'configs/overlays/surrogate_knn_v6.yaml']


def _setup(act='silu', resid=False, hidden='[24,24]'):
    e = load_experiment(BASE, ['agent.type=ekf', 'run.device=cpu', f'agent.hidden={hidden}', 'obs.window=6', 'agent.batch=128'])
    cfg = e.cfg; cfg.activation_fn = act; cfg.use_residual = resid
    info = create_network_info(cfg.dimS, cfg.num_actions, cfg)
    g = torch.Generator().manual_seed(0)
    th = torch.randn(info['total_params'], generator=g) * 0.3
    s = torch.rand(128, cfg.dimS, generator=g); a = torch.randint(0, cfg.num_actions, (128,), generator=g)
    return cfg, info, th, s, a


@pytest.mark.parametrize('act,resid', [('silu', False), ('relu', False), ('tanh', False), ('silu', True)])
def test_jacobian_matches_autograd(act, resid):
    cfg, info, th, s, a = _setup(act, resid)
    H, q = C.per_sample_jacobian(th, info, s, a)
    t = th.clone().requires_grad_(True)
    f = lambda t: C._FS_EAGER(t, info, s)[a, torch.arange(128)].to(torch.float32)
    Href = torch.autograd.functional.jacobian(f, t, vectorize=True)
    assert torch.allclose(q, f(th).detach(), atol=1e-5)
    assert (H - Href).abs().max().item() < 1e-5 * max(1.0, Href.abs().max().item())


def test_ekf_step_same_both_paths():
    from rl.agent import OnlineRHUKFAgent
    cfg, info, th, s, a = _setup()
    ag = OnlineRHUKFAgent(cfg); rng = np.random.default_rng(0)
    for _ in range(300):
        ag.push(rng.random(cfg.dimS).astype(np.float32), int(rng.integers(2)), float(rng.normal()), rng.random(cfg.dimS).astype(np.float32), False)
    ag.learn()
    b = ag.batch_hist[-1]; P0 = 0.01 * torch.eye(info['total_params'])
    outs = []
    for mode in ('analytic', 'autograd'):
        c2 = copy.copy(ag.cfg); c2.ekf_jac = mode
        outs.append(C.ekf_step(ag.theta.clone(), ag.theta_target.clone(), P0.clone(), b, ag.sp, False, 0.01, ag.fv_cache, c2))
    (t1, P1, l1, *_), (t2, P2, l2, *_) = outs
    assert (t1 - t2).abs().max().item() < 1e-6 and (P1 - P2).abs().max().item() < 1e-7 and abs(l1 - l2) < 1e-6
