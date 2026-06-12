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
import time as pytime
import copy

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
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.adam_lr)
        self.buffer = TensorReplayBuffer(cfg.buffer_size, cfg.dimS, cfg.device, cfg)

        self.steps_done = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self._learn_call_count = 0
        self.target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma

        n = sum(p.numel() for p in self.net.parameters())
        gpu = torch.cuda.get_device_name(0) if (cfg.device == 'cuda' and torch.cuda.is_available()) else 'N/A'
        print(f"  Agent: Adam DDQN + Huber (baseline) | Params: {n} | "
              f"Device: {cfg.device} ({gpu}) | lr={cfg.adam_lr} | "
              f"PER: {'ON' if cfg.use_per else 'off'} | n-step: {cfg.n_step_size if cfg.use_n_step else 1}")

    # ─────────────────────────────────────────────────────────
    def warmup_compile(self):
        """use_compile=True면 net을 default 모드로 컴파일 + startup 더미 워밍업(학습 B=batch / act B=1).
        실패 시 eager 폴백. (옵티마이저는 동일 파라미터를 가리키므로 재바인딩 불필요.)"""
        if not getattr(self.cfg, 'use_compile', False):
            return
        try:
            self.net = torch.compile(self.net)          # default mode
            self.net.train()
            for B in (self.cfg.batch_size, 1):           # 학습(B=batch) + act(B=1) 경로 컴파일
                x = torch.zeros(B, self.cfg.dimS, dtype=torch.float32, device=self.device)
                self.net(x).sum().backward()
            self.net.zero_grad(set_to_none=True)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            print("  [compile] Adam net 컴파일+워밍업 완료 (default)")
        except Exception as e:
            print(f"  [compile] Adam 컴파일 실패 → eager 유지: {e}")

    def act(self, state, eps):
        self.steps_done += 1
        if np.random.rand() < eps:
            return int(np.random.choice([0, 1], p=self.cfg.eps_action_probs))
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
        # Huber(smooth_l1) per-sample + (PER off면 is_w=1)
        loss = (is_w * F.smooth_l1_loss(q_a, q_target, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
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

    def end_episode(self, total_reward, episode_length):
        self.episode_count += 1
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
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.steps_done = ckpt['steps_done']
        self.episode_count = ckpt['episode_count']
        self.episode_rewards = ckpt['episode_rewards']
        self.episode_lengths = ckpt['episode_lengths']
        print(f"  [Load] {path} (Adam DDQN baseline, ep={self.episode_count})")
