"""
agent_adam.py — Online Adam DDQN Agent (Huber-loss baseline)
=============================================================
RHUKF-FV의 FIR(receding-horizon) 구조 '순기여'를 isolate하기 위한 baseline.

설계 원칙 (공정 비교):
  - 동일 네트워크 구조(shared_layers → q_layers → nA), 동일 파라미터 수
  - 동일 입력([0,1] NIS), 동일 버퍼(TensorReplayBuffer, n-step/PER 공유 cfg)
  - 손실은 Huber(smooth_l1) — RHUKF의 측정모델/Huber-R과 정렬해서
    "손실함수 차이"가 아니라 "업데이트 구조(FIR vs Adam IIR)" 차이만 남도록 함
  - soft target update (tau_srrhuif 공유)

learn()은 (loss, dt_ms, z_var) 3-튜플 반환 → online_rl_main 언팩과 일치.
인터페이스: act / push / learn / end_episode / save / load /
            get_epsilon / get_q_values / warmup_compile / _compute_adaptive_p / buffer
"""
import os
import time as pytime
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import TensorReplayBuffer
from .network import apply_tf32_config


_ACT = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh,
        'gelu': nn.GELU, 'mish': nn.Mish, 'leaky_relu': nn.LeakyReLU}


class _DDQNNet(nn.Module):
    """순수 DDQN MLP: shared_layers → q_layers → nA (dueling 없음). RHUKF와 동일 구조."""
    def __init__(self, dimS, nA, shared_layers, q_layers, act_name='silu'):
        super().__init__()
        act = _ACT.get(act_name, nn.SiLU)
        layers = []
        in_dim = dimS
        for h in list(shared_layers) + list(q_layers):
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h
        layers.append(nn.Linear(in_dim, nA))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OnlineAdamAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        # 전역 FP32 고정 (forward TF32는 cfg.use_tf32_forward 따름)
        apply_tf32_config(cfg)

        self.net = _DDQNNet(cfg.dimS, cfg.num_actions, cfg.shared_layers,
                            cfg.q_layers, cfg.activation_fn).float().to(cfg.device)
        if str(getattr(cfg, 'adam_init', 'default')).lower() == 'he':   # ★09-14 초기화 동일화(필터 에이전트의 He-normal·bias0 과 같게)
            with torch.no_grad():
                for _m in self.net.modules():
                    if isinstance(_m, nn.Linear):
                        _m.weight.normal_(0.0, (2.0 / _m.in_features) ** 0.5); _m.bias.zero_()
        self.target_net = copy.deepcopy(self.net)
        # env override: ADAM_LR (학습률), ADAM_AMSGRAD (0|1) — Adam baseline sweep용
        _lr = float(cfg.adam_lr)
        self._amsgrad = bool(getattr(cfg, 'adam_amsgrad', False))
        # env OPT=sgd → 순수 SGD(momentum 0) 베이스라인 (2026-09-02 3-옵티마이저 비교)
        self._opt_type = str(getattr(cfg, 'adam_optimizer', 'adam')).lower()
        if self._opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=_lr, momentum=0.0)
        else:
            self._opt_type = 'adam'
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=_lr, amsgrad=self._amsgrad)
        self._eff_lr = _lr
        self.buffer = TensorReplayBuffer(cfg.buffer_size, cfg.dimS, cfg.device, cfg)

        self.steps_done = 0
        self._z_left, self._z_a = 0, 0                    # εz-greedy 지속 상태
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self._learn_call_count = 0
        self._td_hist = deque(maxlen=getattr(cfg, 'td_hist_size', 5000))  # TD-오차 첨도(공정 비교용)
        self.target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma

        n = sum(p.numel() for p in self.net.parameters())
        gpu = torch.cuda.get_device_name(0) if (cfg.device == 'cuda' and torch.cuda.is_available()) else 'N/A'
        _loss_nm = 'MSE' if str(getattr(cfg, 'adam_loss', 'huber')).lower() == 'mse' else 'Huber'
        print(f"  Agent: {self._opt_type.upper()} DDQN + {_loss_nm} (baseline) | Params: {n} | "
              f"Device: {cfg.device} ({gpu}) | lr={self._eff_lr} | "
              f"AMSGrad: {'ON' if self._amsgrad else 'off'} | "
              f"PER: {'ON' if cfg.use_per else 'off'} | n-step: {cfg.n_step_size if cfg.use_n_step else 1}")

    # ─────────────────────────────────────────────────────────
    def warmup_compile(self):
        """use_compile=True면 net을 컴파일 + startup 더미 워밍업(학습 B=batch / act B=1).
        백엔드 캐스케이드: inductor → aot_eager → eager. 실패해도 런 안 죽음."""
        if not getattr(self.cfg, 'use_compile', False):
            return
        import torch._dynamo as _dyn
        _prev = getattr(_dyn.config, 'suppress_errors', False)
        _dyn.config.suppress_errors = False
        base_net = self.net
        for backend in ('inductor', 'aot_eager'):
            try:
                _dyn.reset()
                compiled = torch.compile(base_net, backend=backend)
                compiled.train()
                for B in (self.cfg.batch_size, 1):           # 학습(B=batch) + act(B=1)
                    x = torch.zeros(B, self.cfg.dimS, dtype=torch.float32, device=self.device)
                    compiled(x).sum().backward()
                compiled.zero_grad(set_to_none=True)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                self.net = compiled
                _dyn.config.suppress_errors = _prev
                print(f"  [compile] Adam net 컴파일+워밍업 완료 (backend={backend})")
                return
            except Exception as e:
                _dyn.reset()
                print(f"  [compile] Adam backend={backend} 실패 → 다음 시도: {str(e)[:160]}")
                continue
        _dyn.config.suppress_errors = _prev
        print("  [compile] Adam inductor/aot_eager 모두 실패 → eager 유지")

    def act(self, state, eps):
        self.steps_done += 1
        if eps > 0 and self._z_left > 0:                 # εz-greedy: 탐험 행동 지속 중 (eps=0 평가는 영향 없음)
            self._z_left -= 1
            return self._z_a
        if np.random.rand() < eps:
            a = int(np.random.choice([0, 1], p=self.cfg.eps_action_probs))
            if self.cfg.eps_z_mu > 1.0:                    # 지속 n ~ zeta(μ), 상한 z_cap — 이번 스텝 포함 n 스텝
                self._z_a = a
                self._z_left = min(int(np.random.zipf(self.cfg.eps_z_mu)), int(self.cfg.eps_z_cap)) - 1
            return a
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.net(t).squeeze(0).argmax().item())

    def get_q_values(self, state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net(t).squeeze(0).cpu().numpy()

    def push(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, float(done))

    def _per_beta(self):
        frac = min(1.0, self.episode_count / max(1, self.cfg.max_episodes))
        return self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * frac

    # ─────────────────────────────────────────────────────────
    #  Learning (DDQN + Huber)
    # ─────────────────────────────────────────────────────────
    def learn(self):
        cfg = self.cfg
        if self.buffer.current_size < cfg.batch_size:
            return 0.0, 0.0, 0.0

        # update_interval 게이트 (RHUKF와 동일 주파수)
        self._learn_call_count += 1
        if cfg.update_interval > 1 and (self._learn_call_count % cfg.update_interval) != 0:
            return 0.0, 0.0, 0.0

        if cfg.use_per:
            batch = self.buffer.sample_batch_beta(cfg.batch_size, self._per_beta())
        else:
            batch = self.buffer.sample_batch(cfg.batch_size)

        s = batch['s'].t().float()
        s_next = batch['s_next'].t().float()
        a = batch['a'].long()
        r = batch['r'].float()
        term = batch['term'].float()
        is_w = batch['is_weights'].float()

        t0 = pytime.perf_counter()
        q_a = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_best = self.net(s_next).argmax(dim=1)
            q_next = self.target_net(s_next).gather(1, a_best.unsqueeze(1)).squeeze(1)
            q_target = r + self.target_gamma * (1 - term) * q_next
        td = q_target - q_a
        if getattr(cfg, 'log_td_kurtosis', True):
            try:
                self._td_hist.extend(td.detach().abs().cpu().numpy().ravel().tolist())
            except Exception:
                pass
        # Huber(smooth_l1) 기본 / env ADAM_LOSS=mse → 순수 MSE (2026-09-02 순수 베이스라인)
        if getattr(self, '_loss_mse', None) is None:
            self._loss_mse = (str(getattr(self.cfg, 'adam_loss', 'huber')).lower() == 'mse')
        if self._loss_mse:
            loss = (is_w * F.mse_loss(q_a, q_target, reduction='none')).mean()
        else:
            loss = (is_w * F.smooth_l1_loss(q_a, q_target, reduction='none', beta=float(getattr(self.cfg, 'adam_huber_beta', 1.0)))).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(getattr(self.cfg, 'adam_grad_clip', 1.0)))
        self.optimizer.step()

        # soft target update
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.copy_((1 - cfg.tau_srrhuif) * tp.data + cfg.tau_srrhuif * p.data)

        if cfg.use_per:
            self.buffer.update_priorities(batch['indices'], td.detach())

        dt_ms = (pytime.perf_counter() - t0) * 1000.0
        return loss.item(), dt_ms, q_target.var().item()

    def get_epsilon(self):
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
            np.exp(-self.steps_done / self.cfg.eps_decay_steps)

    def td_kurtosis(self):
        """누적 |TD|의 (n, mean, excess_kurtosis) — RHUKF와 동일 정의(공정 비교)."""
        if len(self._td_hist) < 50:
            return (len(self._td_hist), 0.0, 0.0)
        x = np.asarray(self._td_hist, dtype=np.float64)
        m, s = x.mean(), x.std()
        if s < 1e-9:
            return (len(x), float(m), 0.0)
        exkurt = float(np.mean(((x - m) / s) ** 4) - 3.0)
        return (len(x), float(m), exkurt)

    def end_episode(self, total_reward, episode_length):
        self.episode_count += 1
        self._z_left = 0                                   # εz 지속은 에피소드를 넘기지 않는다
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.buffer.reset_n_step_cache()
        self.buffer.set_current_episode(self.episode_count)

    def _compute_adaptive_p(self, *args, **kwargs):
        return 0.0   # 로깅 호환 (Adam은 P 개념 없음)

    @property
    def theta(self):
        # 진단/궤적 로깅용: 전체 파라미터 flat 벡터
        return torch.cat([p.data.view(-1) for p in self.net.parameters()]).unsqueeze(1)

    def save(self, path):
        torch.save({
            'net': self.net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': self.cfg,
        }, path)
        print(f"  [Save] {path} (Adam DDQN baseline)")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # 우리 체크포인트(config 객체 포함) — torch>=2.6 기본 weights_only=True 회피
        _strip = lambda sd: {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}   # ★09-15 torch.compile 된 net 의 저장 키('_orig_mod.' 접두) 호환
        self.net.load_state_dict(_strip(ckpt['net']))
        self.target_net.load_state_dict(_strip(ckpt['target_net']))
        self.steps_done = ckpt['steps_done']
        self.episode_count = ckpt['episode_count']
        self.episode_rewards = ckpt['episode_rewards']
        self.episode_lengths = ckpt['episode_lengths']
        print(f"  [Load] {path} (Adam DDQN baseline, ep={self.episode_count})")
