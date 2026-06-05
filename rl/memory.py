"""
memory.py — Tensor Replay Buffer (FP32, N-step + PER)
======================================================
GPU 텐서 기반 경험 리플레이.
  - use_n_step=True : 길이 n_step_size deque로 N-step return 캐싱
  - use_per=True    : Proportional PER 샘플링 + IS weight 반환
                      (IS weight의 실제 적용=R^-1 스케일은 필터 step에서 처리)
sample_batch()는 항상 'indices'와 'is_weights'를 포함 (PER off면 is_weights=1).
"""
import torch
from collections import deque
from typing import Dict

DTYPE = torch.float32


class TensorReplayBuffer:
    def __init__(self, capacity: int, dimS: int, device: str, cfg):
        self.capacity, self.count, self.device = capacity, 0, device
        self.S = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.ep_id = torch.zeros(capacity, dtype=torch.long, device=device)
        self.current_ep = 0

        # ── N-step ──
        self.use_n_step = cfg.use_n_step
        self.n_step = cfg.n_step_size if self.use_n_step else 1
        self.gamma = cfg.gamma
        self.n_step_cache = deque(maxlen=self.n_step)

        # ── PER ──
        self.use_per = cfg.use_per
        if self.use_per:
            self.priorities = torch.ones(capacity, dtype=DTYPE, device=device)
            self.max_priority = 1.0
            self.per_alpha = cfg.per_alpha
            self.per_eps = cfg.per_eps
            self.per_apply_is_weight = cfg.per_apply_is_weight

    # ── N-step return 계산 ──
    def _get_n_step_info(self):
        reward = 0.0
        next_state = self.n_step_cache[-1][3]
        done = self.n_step_cache[-1][4]
        for i, tr in enumerate(self.n_step_cache):
            reward += (self.gamma ** i) * tr[2]
            if tr[4]:
                next_state, done = tr[3], True
                break
        return reward, next_state, done

    def push(self, s, a, r, s_next, done):
        if not self.use_n_step:
            self._push_tensor(s, a, r, s_next, done)
            return

        self.n_step_cache.append((s, a, r, s_next, done))
        if len(self.n_step_cache) == self.n_step:
            r_n, s_n, d_n = self._get_n_step_info()
            s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
            self._push_tensor(s_0, a_0, r_n, s_n, d_n)
            self.n_step_cache.popleft()

        # 에피소드 종료 시 자투리(길이<n_step) flush
        if done:
            while len(self.n_step_cache) > 0:
                r_n, s_n, d_n = self._get_n_step_info()
                s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
                self._push_tensor(s_0, a_0, r_n, s_n, d_n)
                self.n_step_cache.popleft()

    def _push_tensor(self, s, a, r, s_next, done):
        idx = self.count % self.capacity
        self.S[idx] = torch.as_tensor(s, dtype=DTYPE, device=self.device)
        self.A[idx] = a
        self.R[idx] = r
        self.S_next[idx] = torch.as_tensor(s_next, dtype=DTYPE, device=self.device)
        self.term[idx] = float(done)
        self.ep_id[idx] = self.current_ep
        if self.use_per:
            self.priorities[idx] = self.max_priority
        self.count += 1

    def reset_n_step_cache(self):
        """에피소드 경계에서 N-step deque 비우기 (online 루프 reset 시 호출 권장)."""
        self.n_step_cache.clear()

    def set_current_episode(self, ep):
        self.current_ep = ep

    @property
    def current_size(self):
        return min(self.count, self.capacity)

    @property
    def is_saturated(self):
        return self.count >= self.capacity

    @property
    def fill_ratio(self):
        return self.current_size / self.capacity

    def sample_batch(self, batch_size: int) -> Dict:
        if not self.use_per:
            indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
            return {
                's': self.S[indices].t(),
                'a': self.A[indices],
                'r': self.R[indices],
                's_next': self.S_next[indices].t(),
                'term': self.term[indices],
                'indices': indices,
                'is_weights': torch.ones(batch_size, dtype=DTYPE, device=self.device),
            }

        # ── PER 샘플링 ──
        sz = self.current_size
        probs_unnorm = self.priorities[:sz] ** self.per_alpha
        probs = probs_unnorm / (probs_unnorm.sum() + 1e-12)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        # IS weight: w_i = (N·P(i))^(-β), max-normalize → (0,1]
        #   (β는 agent가 per_beta 스케줄로 cfg에 주입; 여기선 β=1 기준 계산 후
        #    agent에서 β 적용하거나, beta 인자를 받도록 확장 가능)
        sampling_prob = probs[indices].clamp(min=1e-12)
        is_weights = (sz * sampling_prob) ** (-1.0)
        is_weights = is_weights / is_weights.max().clamp(min=1e-12)
        return {
            's': self.S[indices].t(),
            'a': self.A[indices],
            'r': self.R[indices],
            's_next': self.S_next[indices].t(),
            'term': self.term[indices],
            'indices': indices,
            'is_weights': is_weights.to(DTYPE),
        }

    def sample_batch_beta(self, batch_size: int, beta: float) -> Dict:
        """β를 명시적으로 받아 IS weight를 계산하는 변형 (PER β annealing용)."""
        if not self.use_per:
            return self.sample_batch(batch_size)
        sz = self.current_size
        probs_unnorm = self.priorities[:sz] ** self.per_alpha
        probs = probs_unnorm / (probs_unnorm.sum() + 1e-12)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        sampling_prob = probs[indices].clamp(min=1e-12)
        is_weights = (sz * sampling_prob) ** (-beta)
        is_weights = is_weights / is_weights.max().clamp(min=1e-12)
        return {
            's': self.S[indices].t(),
            'a': self.A[indices],
            'r': self.R[indices],
            's_next': self.S_next[indices].t(),
            'term': self.term[indices],
            'indices': indices,
            'is_weights': is_weights.to(DTYPE),
        }

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """필터 horizon 종료 후 |TD|로 priority 갱신."""
        if not self.use_per:
            return
        new_p = (td_errors.detach().abs() + self.per_eps).to(self.priorities.dtype)
        self.priorities[indices] = new_p
        cur_max = new_p.max().item()
        if cur_max > self.max_priority:
            self.max_priority = cur_max
