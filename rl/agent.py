"""
agent.py — Online RHUKF-FV Agent
==================================
10Hz 실시간 온라인 학습 에이전트 (RHUKF, full-vector covariance).
state_form='error'(기본) 또는 'absolute' 를 cfg로 스위치.

인터페이스(기존 online_rl_main.py 호환):
    act / push / learn / end_episode / save / load
    get_epsilon / get_q_values / warmup_compile / _compute_adaptive_p

learn()은 (loss, dt_ms, z_var) 3-튜플을 반환 (online_rl_main 언팩과 일치).
"""
import time as pytime
from collections import deque
from typing import Tuple

import numpy as np
import torch

from .memory import TensorReplayBuffer
from .network import (
    create_network_info, initialize_theta, FilterCacheFV,
    InputNormalizer, forward_single, DTYPE, apply_tf32_config,
)
from .rhukf_core import (
    rhukf_step_fv, rhukf_step_fv_error, init_error_horizon, compute_per_priorities,
)


class OnlineRHUKFAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        # ── 정밀도: 전역 FP32 고정, forward만 스코프 TF32 (use_tf32_forward) ──
        apply_tf32_config(cfg)

        # ── 네트워크 구조 + 파라미터 ──
        self.info = create_network_info(cfg.dimS, cfg.num_actions, cfg)
        self.theta = initialize_theta(self.info, cfg.device, cfg).view(-1, 1)
        self.theta_target = self.theta.clone()

        # ── FV 필터 캐시 ──
        self.fv_cache = FilterCacheFV(self.info, cfg, cfg.device)

        # ── 입력 정규화 (항상 ON; 드론 NIS는 [0,1]) ──
        self.normalizer = InputNormalizer(cfg.device, scale=cfg.obs_scale) if cfg.use_input_norm else None

        # ── 버퍼 ──
        self.buffer = TensorReplayBuffer(cfg.buffer_size, cfg.dimS, cfg.device, cfg)
        self.batch_hist = deque(maxlen=cfg.N_horizon)
        self._td_hist = deque(maxlen=getattr(cfg, 'td_hist_size', 5000))

        # ── 공유 파라미터 dict (필터 step 입력) ──
        self.sp = {
            'device': cfg.device,
            'info': self.info,
            'batch_sz': cfg.batch_size,
            'normalizer': self.normalizer,
            'current_r_std': cfg.r_init,
            'theta_init': self.theta.clone(),
        }

        # ── 학습 추적 ──
        self.steps_done = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self._last_z_var = 0.0
        self._learn_call_count = 0   # update_interval 게이트용

        n_params = self.info['total_params']
        gpu = torch.cuda.get_device_name(0) if (cfg.device == 'cuda' and torch.cuda.is_available()) else 'N/A'
        print(f"  Agent: RHUKF-FV ({cfg.state_form}) | Params: {n_params} | "
              f"Device: {cfg.device} ({gpu}) | act: {cfg.activation_fn} | "
              f"PER: {'ON' if cfg.use_per else 'off'} | n-step: {cfg.n_step_size if cfg.use_n_step else 1}")

    # ═════════════════════════════════════════════════════════
    #  Warmup / Compile (학습 hot path만; act는 eager; startup에서 사전 워밍업)
    # ═════════════════════════════════════════════════════════
    def warmup_compile(self):
        """use_compile=True면 학습 hot path(forward_single/forward_bmm)를 컴파일하고
        startup 중 실제 shape 더미로 미리 워밍업(첫 컴파일 지연을 라이브 전에 흡수 = "기다렸다 컴파일 후 실행").
        백엔드 캐스케이드: inductor(최적) → aot_eager(C++ codegen 불필요) → eager.
        act()(제어 경로)는 항상 eager 유지. 실패해도 런 안 죽음."""
        if not getattr(self.cfg, 'use_compile', False):
            return
        from . import network as net_mod
        from . import rhukf_core as core_mod
        import torch._dynamo as _dyn
        # 워밍업 동안엔 실패를 '잡아서' 다음 백엔드로 넘기려고 suppress_errors를 잠시 끔
        _prev = getattr(_dyn.config, 'suppress_errors', False)
        _dyn.config.suppress_errors = False

        n_x = self.info['total_params']; num_sigma = 2 * n_x + 1; B = self.cfg.batch_size
        # 실제 호출 방향과 동일 [B, dimS] (forward_bmm 내부 x.t()→[dimS,B] expand)
        s_b = torch.zeros(B, self.cfg.dimS, dtype=DTYPE, device=self.device)
        sig = torch.zeros(num_sigma, n_x, dtype=DTYPE, device=self.device)
        th = self.theta.squeeze()

        for backend in ('inductor', 'aot_eager'):
            try:
                _dyn.reset()   # 이전 시도 캐시 정리
                c_single = torch.compile(net_mod.forward_single, backend=backend)
                c_bmm = torch.compile(net_mod.forward_bmm, backend=backend)
                with torch.no_grad():
                    _ = c_bmm(sig, self.info, s_b)        # 시그마 forward (학습 hot path)
                    _ = c_single(th, self.info, s_b)      # filter 내 a_best (B=batch)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                # 성공: 학습 경로 전역 이름 교체 (agent.py import한 forward_single=act경로는 eager 유지)
                net_mod.forward_single = c_single; net_mod.forward_bmm = c_bmm
                core_mod.forward_single = c_single; core_mod.forward_bmm = c_bmm
                _dyn.config.suppress_errors = _prev
                print(f"  [compile] RHUKF 컴파일+워밍업 완료 (backend={backend}). act는 eager")
                return
            except Exception as e:
                _dyn.reset()
                print(f"  [compile] backend={backend} 실패 → 다음 시도: {str(e)[:160]}")
                continue
        _dyn.config.suppress_errors = _prev
        print("  [compile] inductor/aot_eager 모두 실패 → eager 유지 (런 정상, 레퍼런스도 eager)")

    # ═════════════════════════════════════════════════════════
    #  Action Selection
    # ═════════════════════════════════════════════════════════
    def act(self, state: np.ndarray, eps: float) -> int:
        self.steps_done += 1
        if np.random.rand() < eps:
            return int(np.random.choice([0, 1], p=self.cfg.eps_action_probs))
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=DTYPE, device=self.device)
            if self.normalizer:
                s_t = self.normalizer.normalize(s_t)
            q = forward_single(self.theta.squeeze(), self.info, s_t)
            return q.squeeze().argmax().item()

    # ═════════════════════════════════════════════════════════
    #  Experience Storage
    # ═════════════════════════════════════════════════════════
    def push(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        self.buffer.push(s, a, r, s_next, float(done))

    # ═════════════════════════════════════════════════════════
    #  PER β annealing
    # ═════════════════════════════════════════════════════════
    def _per_beta(self) -> float:
        frac = min(1.0, self.episode_count / max(1, self.cfg.max_episodes))
        return self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * frac

    # ═════════════════════════════════════════════════════════
    #  Learning (RHUKF-FV Receding Horizon)
    # ═════════════════════════════════════════════════════════
    def learn(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        if self.buffer.current_size < cfg.batch_size:
            return 0.0, 0.0, 0.0

        # ── update_interval 게이트: N번 호출마다 1번만 실제 업데이트 ──
        self._learn_call_count += 1
        if cfg.update_interval > 1 and (self._learn_call_count % cfg.update_interval) != 0:
            return 0.0, 0.0, 0.0

        # r-std 스케줄 (eps와 동일 지수감쇠; r_init==r_end면 상수)
        decay = float(np.exp(-self.steps_done / cfg.eps_decay_steps))
        self.sp['current_r_std'] = cfg.r_end + (cfg.r_init - cfg.r_end) * decay

        # 윈도우에 배치 추가 (PER이면 β annealing IS weight)
        if cfg.use_per:
            batch = self.buffer.sample_batch_beta(cfg.batch_size, self._per_beta())
        else:
            batch = self.buffer.sample_batch(cfg.batch_size)
        self.batch_hist.append(batch)
        if len(self.batch_hist) < cfg.N_horizon:
            return 0.0, 0.0, 0.0

        t0 = pytime.perf_counter()
        loss = 0.0
        z_var_sum = 0.0

        if cfg.state_form == 'error':
            ctx = init_error_horizon(self.theta, self.theta_target,
                                     list(self.batch_hist), self.sp, cfg, self.fv_cache)
            fs = None
            for h in range(cfg.N_horizon):
                self.theta, fs, l_val, t_var, _, _ = rhukf_step_fv_error(
                    fs, ctx, self.batch_hist[h], h, self.sp, cfg, self.fv_cache)
                loss = l_val
                z_var_sum += t_var
        else:  # absolute
            filter_state = None
            for h in range(cfg.N_horizon):
                self.theta, filter_state, l_val, t_var, _, _ = rhukf_step_fv(
                    self.theta, self.theta_target, filter_state, self.batch_hist[h],
                    self.sp, (h == 0), cfg.p_init, self.fv_cache, cfg)
                loss = l_val
                z_var_sum += t_var

        # ── Soft target update ──
        self.theta_target = (1.0 - cfg.tau_srrhuif) * self.theta_target + cfg.tau_srrhuif * self.theta

        # ── PER priority 갱신 + TD-오차 첨도 로깅(무거운 꼬리 증거) ──
        td = None
        if cfg.use_per:
            idx, td = compute_per_priorities(self.theta, self.theta_target,
                                             list(self.batch_hist), self.sp, cfg)
            if idx is not None:
                self.buffer.update_priorities(idx, td)
        elif getattr(cfg, 'log_td_kurtosis', True):
            _, td = compute_per_priorities(self.theta, self.theta_target,
                                           list(self.batch_hist), self.sp, cfg, force=True)
        if td is not None:
            try:
                self._td_hist.extend(td.detach().cpu().numpy().ravel().tolist())
            except Exception:
                pass

        dt_ms = (pytime.perf_counter() - t0) * 1000.0
        avg_z = z_var_sum / cfg.N_horizon
        self._last_z_var = avg_z
        return loss, dt_ms, avg_z

    # 로깅 호환 (기존 _compute_adaptive_p 자리)
    def _compute_adaptive_p(self, *args, **kwargs) -> float:
        return self.cfg.p_init

    def td_kurtosis(self):
        """누적 |TD|의 (n, mean, excess_kurtosis). 무거운 꼬리(>0)면 Huber/유계영향 이점 근거."""
        if len(self._td_hist) < 50:
            return (len(self._td_hist), 0.0, 0.0)
        x = np.asarray(self._td_hist, dtype=np.float64)
        m, s = x.mean(), x.std()
        if s < 1e-9:
            return (len(x), float(m), 0.0)
        exkurt = float(np.mean(((x - m) / s) ** 4) - 3.0)
        return (len(x), float(m), exkurt)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=DTYPE, device=self.device)
            if self.normalizer:
                s_t = self.normalizer.normalize(s_t)
            q = forward_single(self.theta.squeeze(), self.info, s_t)
            return q.squeeze().cpu().numpy()

    # ═════════════════════════════════════════════════════════
    #  Epsilon
    # ═════════════════════════════════════════════════════════
    def get_epsilon(self) -> float:
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
            np.exp(-self.steps_done / self.cfg.eps_decay_steps)

    # ═════════════════════════════════════════════════════════
    #  Episode Lifecycle
    # ═════════════════════════════════════════════════════════
    def end_episode(self, total_reward: float, episode_length: int):
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        # 에피소드 경계에서 N-step deque 비우기 (다음 에피소드로 누수 방지)
        self.buffer.reset_n_step_cache()
        self.buffer.set_current_episode(self.episode_count)

    # ═════════════════════════════════════════════════════════
    #  Save / Load
    # ═════════════════════════════════════════════════════════
    def save(self, path: str):
        torch.save({
            'theta': self.theta,
            'theta_target': self.theta_target,
            'info': self.info,
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': self.cfg,
        }, path)
        print(f"  [Save] {path} ({self.info['total_params']} params)")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.theta = ckpt['theta']
        self.theta_target = ckpt['theta_target']
        self.steps_done = ckpt['steps_done']
        self.episode_count = ckpt['episode_count']
        self.episode_rewards = ckpt['episode_rewards']
        self.episode_lengths = ckpt['episode_lengths']
        print(f"  [Load] {path} (ep={self.episode_count}, steps={self.steps_done})")
