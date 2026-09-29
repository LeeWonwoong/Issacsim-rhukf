"""
agent_clf.py — 비용가중 지도 분류기 베이스라인 (★09-30 리뷰어 R6 "근시 문제면 지도 분류기로 되지 않나" 대응)
===================================================================================================
DQN 과 같은 정보·같은 조건에서 가치(부트스트랩) 대신 공격 사후확률을 직접 배운다.

  - 같은 망 _DDQNNet(shared_layers → nA=2), 같은 He 초기화·파라미터 수, 같은 Adam(lr = agent.adam.lr)
  - 같은 관측 창, 같은 ε(z)-greedy 탐험·감쇠, 같은 버퍼 크기·배치·update_interval
  - 학습 신호 = 같은 보상 r 만. 라벨은 (직전 행동 a, r) 에서 복원한다(공격 참값을 따로 받지 않는다):
        base = alive·scale
        a=track: r < base → 공격 중 미탐(−c_d) → 1,   r = base → 평시 → 0
        a=hover: r < base → 평시 오경보(−c_fa) → 0,  r ≥ base → 공격 중(보너스 포함) → 1
    (terminal_penalty·c_sw·성형이 0 인 탐지기형 보상에서만 정확 — 생성자에서 확인)
  - 손실 = 비용가중 교차엔트로피: 공격 라벨 가중 c_d, 평시 라벨 가중 c_fa → 최적 임계가 0.5 (최소 위험 분류)
  - 행동 규칙 (agent.clf.threshold)
        cost  : p(공격) > 0.5  (가중이 비용비를 담으므로 = 1스텝 베이즈 임계 c_fa/(c_fa+c_d))
        bonus : 직전 행동이 track 이면 진입 임계 c_fa/(c_fa+c_d+B), hover 면 유지 임계 c_fa/(c_fa+c_d)
                (사건 첫 hover 보너스 B 를 반영한 근시 베이즈 규칙, 가중 전 확률로 환산)
learn() 은 (loss, dt_ms, None) 3-튜플, 인터페이스는 OnlineAdamAgent 와 같다.
"""
import copy
import time as pytime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_adam import _DDQNNet
from .network import apply_tf32_config


class OnlineCostClassifierAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        apply_tf32_config(cfg)
        rc = cfg.reward
        if rc.mode != 'cost' or float(rc.terminal_penalty) != 0 or float(getattr(rc, 'c_sw', 0.0)) != 0 or float(rc.shape_tilt) != 0:
            raise ValueError('agent.type=clf 는 탐지기형 cost 보상(terminal_penalty·c_sw·shape_tilt = 0)에서만 라벨을 복원할 수 있다')
        self._base = float(rc.alive) * float(rc.scale)
        self._eps_r = 1e-6 * max(1.0, abs(float(rc.scale)))
        self._w = torch.tensor([float(rc.c_fa), float(rc.c_d)], dtype=torch.float32, device=cfg.device)   # [평시, 공격] 가중
        self.mode = str(getattr(cfg, 'clf_threshold', 'cost')).lower()
        if self.mode not in ('cost', 'bonus'):
            raise ValueError(f'agent.clf.threshold={self.mode!r} (cost|bonus)')
        cfa, cd, B = float(rc.c_fa), float(rc.c_d), float(rc.bonus)
        self._th_stay = cfa / (cfa + cd)                 # 비가중 사후확률 기준 임계
        self._th_enter = cfa / (cfa + cd + B) if self.mode == 'bonus' else self._th_stay

        self.net = _DDQNNet(cfg.dimS, cfg.num_actions, cfg.shared_layers, cfg.q_layers, cfg.activation_fn).float().to(cfg.device)
        if str(getattr(cfg, 'adam_init', 'default')).lower() == 'he':
            with torch.no_grad():
                for _m in self.net.modules():
                    if isinstance(_m, nn.Linear):
                        _m.weight.normal_(0.0, (2.0 / _m.in_features) ** 0.5); _m.bias.zero_()
        self.target_net = copy.deepcopy(self.net)        # 인터페이스 호환(사용 안 함)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.adam_lr))
        n = int(cfg.buffer_size)
        self._S = np.zeros((n, cfg.dimS), np.float32); self._Y = np.zeros(n, np.int64); self._n = 0; self._i = 0
        self.steps_done = 0
        self._z_left, self._z_a = 0, 0
        self.episode_count = 0; self.episode_rewards = []; self.episode_lengths = []
        self._learn_call_count = 0
        self.n_label = np.zeros(2, np.int64)
        print(f"  Agent: 비용가중 분류기(clf) | Params: {sum(p.numel() for p in self.net.parameters())} | lr={cfg.adam_lr} | "
              f"임계 {self.mode}: 진입 {self._th_enter:.3f} · 유지 {self._th_stay:.3f} | 가중 c_fa {cfa} · c_d {cd}")

    # ── 라벨 복원 ─────────────────────────────────────────
    def label_of(self, a, r):
        low = float(r) < self._base - self._eps_r
        return int(low) if int(a) == 0 else int(not low)

    # ── 확률(비가중 사후확률로 환산) ───────────────────────
    def _p_attack(self, logits):
        """가중 CE 의 최적 출력 q = c_d·p / (c_d·p + c_fa·(1−p)) → 비가중 p 로 되돌린다."""
        q = torch.softmax(logits, dim=-1)[..., 1]
        cfa, cd = float(self._w[0]), float(self._w[1])
        return (cfa * q) / (cfa * q + cd * (1.0 - q) + 1e-12)

    def act(self, state, eps, greedy=False):
        self.steps_done += 1
        if eps > 0 and self._z_left > 0:
            self._z_left -= 1
            return self._z_a
        if np.random.rand() < eps:
            a = int(np.random.choice([0, 1], p=self.cfg.eps_action_probs))
            if self.cfg.eps_z_mu > 1.0:
                self._z_a = a
                self._z_left = min(int(np.random.zipf(self.cfg.eps_z_mu)), int(self.cfg.eps_z_cap)) - 1
            return a
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            p = float(self._p_attack(self.net(t)).item())
        prev = int(round(float(state[-1])))              # 관측 프레임 마지막 원소 = 직전 행동(features 끝이 action)
        return int(p > (self._th_stay if prev == 1 else self._th_enter))

    def get_q_values(self, state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net(t).squeeze(0).cpu().numpy()

    def push(self, s, a, r, s_next, done):
        y = self.label_of(a, r)
        self._S[self._i] = s; self._Y[self._i] = y; self.n_label[y] += 1
        self._i = (self._i + 1) % len(self._Y); self._n = min(self._n + 1, len(self._Y))

    def learn(self):
        cfg = self.cfg
        if self._n < cfg.batch_size:
            return 0.0, 0.0, None
        self._learn_call_count += 1
        if cfg.update_interval > 1 and (self._learn_call_count % cfg.update_interval) != 0:
            return 0.0, 0.0, None
        idx = np.random.randint(0, self._n, cfg.batch_size)
        s = torch.as_tensor(self._S[idx], device=self.device); y = torch.as_tensor(self._Y[idx], device=self.device)
        t0 = pytime.perf_counter()
        loss = F.cross_entropy(self.net(s), y, weight=self._w)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item(), (pytime.perf_counter() - t0) * 1000.0, None

    def get_epsilon(self):
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * np.exp(-self.steps_done / self.cfg.eps_decay_steps)

    def end_episode(self, total_reward, episode_length):
        self.episode_count += 1; self._z_left = 0
        self.episode_rewards.append(total_reward); self.episode_lengths.append(episode_length)

    def _compute_adaptive_p(self, *args, **kwargs):
        return 0.0

    @property
    def theta(self):
        return torch.cat([p.data.view(-1) for p in self.net.parameters()]).unsqueeze(1)

    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'steps_done': self.steps_done, 'episode_count': self.episode_count,
                    'n_label': self.n_label.tolist(), 'mode': self.mode}, path)
        print(f"  [Save] {path} (비용가중 분류기)")
