"""
custom_env.py — RHUKF-FV vs Adam DDQN 오프라인 검증
=====================================================
RealisticCliffEnv(공격/기동/NIS 겹침 모사)에서 RHUKF-FV 에이전트가
정상 학습되는지 + Adam DDQN 대비 reward 곡선을 비교한다.

온라인(online_rl_main.py)과 동일한 OnlineRHUKFAgent / rl.network / env.reward 사용.
weight-space loss landscape는 검증 단계라 생략, reward/loss 비교 + 4-Context Q-landscape만.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import time
import copy

from swrl_config import Config
from rl.agent import OnlineRHUKFAgent
from rl.memory import TensorReplayBuffer
from rl.network import forward_single, DTYPE
from env.reward import calculate_reward

# =========================================================================
# 1. Adam DDQN Agent (FP32 baseline, 새 buffer/n-step/PER 사용)
# =========================================================================
class DDQNNetwork(nn.Module):
    """순수 DDQN MLP: shared_layers → q_layers → nA (dueling 없음)."""
    def __init__(self, dimS, num_actions, shared_layers, q_layers):
        super().__init__()
        layers = []; in_dim = dimS
        for h in list(shared_layers) + list(q_layers):
            layers.append(nn.Linear(in_dim, h)); layers.append(nn.ReLU()); in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AdamDDQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg; self.device = cfg.device
        self.net = DDQNNetwork(cfg.dimS, cfg.num_actions, cfg.shared_layers,
                               cfg.q_layers).float().to(cfg.device)
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4,
                                           amsgrad=getattr(cfg, 'adam_amsgrad', True))
        self.buffer = TensorReplayBuffer(cfg.buffer_size, cfg.dimS, cfg.device, cfg)
        self.steps_done = 0; self.episode_count = 0
        self.episode_rewards = []; self.episode_lengths = []; self.info = None
        self._learn_call_count = 0   # update_interval 게이트용
        self.target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
        print(f"  Agent: Adam DDQN | Params: {sum(p.numel() for p in self.net.parameters())} | Device: {cfg.device}")

    def warmup_compile(self): pass

    def get_q_values(self, state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net(t).squeeze().cpu().numpy()

    def act(self, state, eps):
        self.steps_done += 1
        if np.random.rand() < eps:
            return int(np.random.choice([0, 1], p=self.cfg.eps_action_probs))
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net(t).squeeze().argmax().item()

    def push(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, float(done))

    def _per_beta(self):
        frac = min(1.0, self.episode_count / max(1, self.cfg.max_episodes))
        return self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * frac

    def learn(self):
        if self.buffer.current_size < self.cfg.batch_size:
            return 0.0, 0.0, 0.0, 3e-4
        self._learn_call_count += 1
        if self.cfg.update_interval > 1 and (self._learn_call_count % self.cfg.update_interval) != 0:
            return 0.0, 0.0, 0.0, 3e-4
        if self.cfg.use_per:
            batch = self.buffer.sample_batch_beta(self.cfg.batch_size, self._per_beta())
        else:
            batch = self.buffer.sample_batch(self.cfg.batch_size)
        s = batch['s'].t().float(); s_next = batch['s_next'].t().float()
        a = batch['a'].long(); r = batch['r'].float(); term = batch['term'].float()
        is_w = batch['is_weights'].float()
        t0 = time.perf_counter()

        q_a = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_best = self.net(s_next).argmax(dim=1)
            q_next = self.target_net(s_next).gather(1, a_best.unsqueeze(1)).squeeze(1)
            q_target = r + self.target_gamma * (1 - term) * q_next
        td = q_target - q_a
        # PER: IS weight를 per-sample loss에 적용
        loss = (is_w * F.smooth_l1_loss(q_a, q_target, reduction='none')).mean()
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); self.optimizer.step()

        # soft target update
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.copy_((1 - self.cfg.tau_srrhuif) * tp.data + self.cfg.tau_srrhuif * p.data)

        if self.cfg.use_per:
            self.buffer.update_priorities(batch['indices'], td.detach())

        return loss.item(), (time.perf_counter() - t0) * 1000, q_target.var().item(), 3e-4

    def get_epsilon(self):
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
            np.exp(-self.steps_done / self.cfg.eps_decay_steps)

    def end_episode(self, total_reward, episode_length):
        self.episode_count += 1
        self.episode_rewards.append(total_reward); self.episode_lengths.append(episode_length)
        self.buffer.reset_n_step_cache(); self.buffer.set_current_episode(self.episode_count)

    @property
    def theta(self):
        return torch.cat([p.data.view(-1) for p in self.net.parameters()]).unsqueeze(1)


# =========================================================================
# 2. 환경 (현실 모사 — 다중공격, 기동프로파일, NIS 겹침, Done)
# =========================================================================
class RealisticCliffEnv(gym.Env):
    def __init__(self, cfg, window_size=4):
        super().__init__()
        self.cfg = cfg
        self.action_space = spaces.Discrete(2); self.window_size = window_size
        self.max_steps = cfg.episode_max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(window_size * 3,), dtype=np.float32)
        self.window = deque(maxlen=window_size)

    def _generate_nis(self, is_attack, action, ramp_steps=0):
        hover_penalty = 0.12 if action == 1 else 0.0
        maneuver = self._get_maneuver_profile()
        if not is_attack:
            base_vel = 0.05 + hover_penalty + maneuver
            base_gyro = 0.05 + hover_penalty + maneuver * 1.3
            nis_vel = np.clip(np.random.normal(base_vel, 0.025), 0.0, 1.0)
            nis_gyro = np.clip(np.random.normal(base_gyro, 0.03), 0.0, 1.0)
        else:
            ramp = min(1.0, ramp_steps / 2.0)
            atk = ramp * self.current_intensity
            base_vel = 0.05 + atk + hover_penalty + maneuver
            base_gyro = 0.05 + atk * 1.3 + hover_penalty + maneuver * 1.3
            nis_vel = np.clip(np.random.normal(base_vel, 0.025 + ramp * 0.015), 0.0, 1.0)
            nis_gyro = np.clip(np.random.normal(base_gyro, 0.03 + ramp * 0.015), 0.0, 1.0)
        return nis_vel, nis_gyro

    def _get_maneuver_profile(self):
        t = self.step_count; total = 0.0
        for start, peak_t, end in self.maneuver_events:
            if start <= t <= end:
                if t <= peak_t:
                    total += self.maneuver_amplitude * (t - start) / max(1, peak_t - start)
                else:
                    total += self.maneuver_amplitude * (end - t) / max(1, end - peak_t)
        return total

    def _check_attack(self, step):
        for s, e in self.attack_phases:
            if s <= step <= e: return True
        return False

    def _get_ramp_steps(self, step):
        for s, e in self.attack_phases:
            if s <= step <= e: return step - s
        return 0

    def _get_intensity(self, step):
        for i, (s, e) in enumerate(self.attack_phases):
            if s <= step <= e: return self.attack_intensities[i]
        return 0.0

    def _is_post_attack(self, step):
        for s, e in self.attack_phases:
            if step > e and step <= e + 30: return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.step_count = 1
        self.consecutive_fn = 0; self.consecutive_fp = 0; self.prev_action = 0
        self.maneuver_events = []
        self.maneuver_amplitude = np.random.uniform(0.08, 0.20)
        t = np.random.randint(5, 25)
        while t < self.max_steps - 20:
            dur = np.random.randint(10, 30); peak_t = t + dur // 2
            self.maneuver_events.append((t, peak_t, t + dur))
            t += dur + np.random.randint(8, 30)
        self.attack_phases = []; self.attack_intensities = []
        num_attacks = np.random.randint(2, 4)
        t = np.random.randint(15, 40)
        for _ in range(num_attacks):
            dur = np.random.randint(8, 25); intensity = np.random.uniform(0.10, 0.30)
            self.attack_phases.append((t, t + dur)); self.attack_intensities.append(intensity)
            t = t + dur + np.random.randint(15, 40)
            if t > self.max_steps - 15: break
        self.current_is_attack = self._check_attack(self.step_count)
        self.current_intensity = self._get_intensity(self.step_count)
        nis_vel, nis_gyro = self._generate_nis(self.current_is_attack, 0, self._get_ramp_steps(self.step_count))
        self.window.clear()
        for _ in range(self.window_size - 1): self.window.append([0.05, 0.05, 0.0])
        self.window.append([nis_vel, nis_gyro, 0.0])
        return np.array(self.window, dtype=np.float32).flatten(), {}

    def step(self, action):
        rc = self.cfg.reward
        if self.current_is_attack and action == 0: self.consecutive_fn += 1
        else: self.consecutive_fn = 0
        if not self.current_is_attack and action == 1 and self._is_post_attack(self.step_count):
            self.consecutive_fp += 1
        else: self.consecutive_fp = 0

        reward = calculate_reward(action, self.current_is_attack,
                                  self.consecutive_fn, self.consecutive_fp, rc=rc)
        terminated = False; truncated = False
        if self.consecutive_fn >= 3: terminated = True; reward = rc.terminal_penalty
        if self.consecutive_fp >= 4: terminated = True; reward = rc.terminal_penalty
        self.step_count += 1
        if self.step_count > self.max_steps: truncated = True
        done = terminated or truncated

        if done:
            obs = np.array(self.window, dtype=np.float32).flatten()
            self.current_is_attack = self._check_attack(self.step_count)
            self.current_intensity = self._get_intensity(self.step_count)
            return obs, reward, terminated, truncated, {}

        next_is_attack = self._check_attack(self.step_count)
        self.current_intensity = self._get_intensity(self.step_count)
        nis_vel, nis_gyro = self._generate_nis(next_is_attack, action, self._get_ramp_steps(self.step_count))
        self.window.append([nis_vel, nis_gyro, float(action)])
        obs = np.array(self.window, dtype=np.float32).flatten()
        self.current_is_attack = next_is_attack; self.prev_action = action
        return obs, reward, terminated, truncated, {}


# =========================================================================
# 3. 플롯터
# =========================================================================
class ComparisonPlotter:
    def __init__(self, max_episodes, max_reward, outdir, param_str):
        self.outdir = outdir; self.param_str = param_str; self.max_reward = max_reward
        self.data = {m: {'rewards': [], 'losses': [], 'p_inits': [], 'z_vars': [], 'k_gains': []}
                     for m in ['RHUKF', 'Adam']}
        self.fig, self.axes = plt.subplots(2, 5, figsize=(28, 8))
        titles = ['Reward', 'TD Loss', 'P / LR', 'Z_var', '||Δθ||']
        self.lines = {}
        for row, method in enumerate(['RHUKF', 'Adam']):
            self.lines[method] = {}
            for col, title in enumerate(titles):
                ax = self.axes[row, col]; ax.set_xlim(0, max_episodes)
                clrs = ['royalblue', 'red', 'green', 'magenta', 'darkorange']
                if col == 0:
                    self.lines[method]['r_raw'], = ax.plot([], [], clrs[col], alpha=0.3)
                    self.lines[method]['r_ma'], = ax.plot([], [], clrs[col], linewidth=2)
                    ax.axhline(y=max_reward * 0.9, color='g', linestyle='--', alpha=0.5)
                    ax.set_ylim(-20, max_reward * 1.1)
                else:
                    key = ['', 'loss', 'p_init', 'z_var', 'k_gain'][col]
                    self.lines[method][key], = ax.plot([], [], clrs[col], linewidth=1.5)
                ax.set_title(f'{method}: {title}')
        plt.tight_layout()

    def add(self, method, reward, loss, p_init, z_var, k_gain):
        d = self.data[method]; d['rewards'].append(reward); d['losses'].append(loss)
        d['p_inits'].append(p_init); d['z_vars'].append(z_var); d['k_gains'].append(k_gain)

    def refresh(self):
        for row, method in enumerate(['RHUKF', 'Adam']):
            d = self.data[method]
            if not d['rewards']: continue
            ep = range(len(d['rewards']))
            self.lines[method]['r_raw'].set_data(ep, d['rewards'])
            if len(d['rewards']) >= 20:
                ma = np.convolve(d['rewards'], np.ones(20) / 20, 'valid')
                self.lines[method]['r_ma'].set_data(range(19, len(d['rewards'])), ma)
            self.lines[method]['loss'].set_data(ep, d['losses'])
            self.lines[method]['p_init'].set_data(ep, d['p_inits'])
            self.lines[method]['z_var'].set_data(ep, d['z_vars'])
            self.lines[method]['k_gain'].set_data(ep, d['k_gains'])
            for col in range(5):
                self.axes[row, col].relim(); self.axes[row, col].autoscale_view()
            self.axes[row, 0].set_ylim(-20, self.max_reward * 1.1)
        plt.savefig(os.path.join(self.outdir, f"{self.param_str}_comparison_live.png"), dpi=120)

    def close(self): plt.close(self.fig)


class LivePlotter:
    def __init__(self, method_name, max_episodes, max_reward, outdir, param_str):
        self.method_name = method_name; self.outdir = outdir; self.max_reward = max_reward
        self.rewards, self.losses, self.p_inits, self.z_vars, self.k_gains, self.ep_lengths = [], [], [], [], [], []
        self.fig, self.axes = plt.subplots(1, 5, figsize=(25, 4))
        self.line_r_raw, = self.axes[0].plot([], [], 'b-', alpha=0.3)
        self.line_r_ma, = self.axes[0].plot([], [], 'b-', linewidth=2)
        self.axes[0].axhline(y=max_reward * 0.9, color='g', linestyle='--', alpha=0.5)
        self.axes[0].set_xlim(0, max_episodes); self.axes[0].set_ylim(-20, max_reward * 1.1)
        self.axes[0].set_title(f'Reward ({method_name})')
        self.line_l, = self.axes[1].plot([], [], 'r-', linewidth=1.5); self.axes[1].set_xlim(0, max_episodes); self.axes[1].set_title('TD Loss')
        self.line_p, = self.axes[2].plot([], [], 'g-', linewidth=2); self.axes[2].set_xlim(0, max_episodes); self.axes[2].set_title('P_init')
        self.line_z, = self.axes[3].plot([], [], 'm-', linewidth=1.5); self.axes[3].set_xlim(0, max_episodes); self.axes[3].set_title('Z_var')
        self.line_k, = self.axes[4].plot([], [], 'darkorange', linewidth=1.5); self.axes[4].set_xlim(0, max_episodes); self.axes[4].set_title('||Δθ||')
        plt.tight_layout(); self.filename = os.path.join(outdir, f"{param_str}_{method_name.replace(' ', '_')}")

    def add(self, reward, loss, p_init, z_var, k_gain, ep_len=0):
        self.rewards.append(reward); self.losses.append(loss); self.p_inits.append(p_init)
        self.z_vars.append(z_var); self.k_gains.append(k_gain); self.ep_lengths.append(ep_len)

    def refresh(self):
        ep = range(len(self.rewards)); self.line_r_raw.set_data(ep, self.rewards)
        if len(self.rewards) >= 20:
            self.line_r_ma.set_data(range(19, len(self.rewards)), np.convolve(self.rewards, np.ones(20) / 20, 'valid'))
        self.line_l.set_data(ep, self.losses); self.line_p.set_data(ep, self.p_inits)
        self.line_z.set_data(ep, self.z_vars); self.line_k.set_data(ep, self.k_gains)
        for ax in self.axes: ax.relim(); ax.autoscale_view()
        self.axes[0].set_ylim(-20, self.max_reward * 1.1); plt.savefig(f'{self.filename}_live.png', dpi=100)

    def close(self): plt.close(self.fig)


# =========================================================================
# 4. 4-Context Q-Landscape (forward_single 기반 — RHUKF/Adam 공통)
# =========================================================================
def plot_comparison_4context(theta_rhukf, info_rhukf, adam_net, cfg, param_str, resolution=40):
    print(f"\n[Comparison] 4-Context Q-Landscape 생성 중...")
    device = cfg.device
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, resolution), np.linspace(0.0, 1.0, resolution))
    N = [0.05, 0.05, 0.0]; A = [0.25, 0.35, 1.0]
    contexts = [("Peaceful (N-N-N)", [N, N, N], 0.0), ("Under Attack (A-A-A)", [A, A, A], 1.0),
                ("Early Attack (N-N-A)", [N, N, A], 1.0), ("Early Recovery (A-A-N)", [A, A, N], 0.0)]
    fig = plt.figure(figsize=(24, 12))
    for row, (mname, qfn) in enumerate([
        ("RHUKF-FV", lambda s: forward_single(theta_rhukf.squeeze(), info_rhukf, s)),
        ("Adam DDQN", lambda s: adam_net(s.float()))]):
        for col, (title, hist, prev_a) in enumerate(contexts):
            states = np.zeros((resolution * resolution, cfg.window_size * 3))
            states[:, 0:3] = hist[0]; states[:, 3:6] = hist[1]; states[:, 6:9] = hist[2]
            states[:, 9] = X.flatten(); states[:, 10] = Y.flatten(); states[:, 11] = prev_a
            st = torch.tensor(states, dtype=DTYPE, device=device)
            with torch.no_grad():
                if row == 0:
                    mq = qfn(st.t()).max(dim=0).values.cpu().numpy()
                else:
                    mq = qfn(st).max(dim=-1).values.cpu().numpy()
            Z = mq.reshape(resolution, resolution)
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
            zf = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.15
            ax.contourf(X, Y, Z, zdir='z', offset=zf, cmap='plasma', alpha=0.5)
            ax.set_zlim(zf, np.max(Z)); ax.view_init(elev=25, azim=230)
            ax.set_title(f'{mname}\n{title}', fontsize=10)
    fig.suptitle(f'Q-Landscape Comparison\n({param_str})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, f"{param_str}_comparison_4context.png"), dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 5. 학습 루프
# =========================================================================
def train_agent(agent, env, cfg, method_name, logger, comp_logger=None):
    is_rhukf = isinstance(agent, OnlineRHUKFAgent)
    ep_times = []
    for episode in range(1, cfg.max_episodes + 1):
        ep_start = time.time(); state, _ = env.reset()
        done = False; episode_reward = 0.0; step_count = 0
        losses, p_inits, z_vars, k_gains = [], [], [], []
        q_track_list, q_hover_list = [], []
        while not done:
            eps = agent.get_epsilon()
            qv = agent.get_q_values(state); q_track_list.append(qv[0]); q_hover_list.append(qv[1])
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push(state, action, reward, next_state, terminated)

            theta_before = agent.theta.clone()
            result = agent.learn()
            loss, z_var = result[0], result[2]
            p_init_val = result[3] if len(result) > 3 else cfg.p_init
            k_gain = 0.0
            if loss > 0:
                k_gain = torch.norm(theta_before.squeeze() - agent.theta.squeeze()).item()
                losses.append(loss); p_inits.append(p_init_val); z_vars.append(z_var); k_gains.append(k_gain)
            state = next_state; episode_reward += reward; step_count += 1
        agent.end_episode(episode_reward, step_count)
        ep_times.append(time.time() - ep_start)

        avg_loss = np.mean(losses) if losses else 0.0
        avg_p = np.mean(p_inits) if p_inits else cfg.p_init
        avg_z = np.mean(z_vars) if z_vars else 0.0
        avg_k = np.mean(k_gains) if k_gains else 0.0
        logger.add(episode_reward, avg_loss, avg_p, avg_z, avg_k, step_count)
        if comp_logger: comp_logger.add(method_name, episode_reward, avg_loss, avg_p, avg_z, avg_k)

        if episode % 10 == 0:
            logger.refresh()
            if comp_logger: comp_logger.refresh()
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            avg_qt = np.mean(q_track_list); avg_qh = np.mean(q_hover_list)
            avg_t = np.mean(ep_times[-10:]) if len(ep_times) >= 10 else np.mean(ep_times)
            print(f"[{method_name}] Ep {episode:3d} | R: {episode_reward:7.1f} | Avg20: {recent:7.1f} | "
                  f"Steps: {step_count:3d} | eps: {eps:.2f} | Loss: {avg_loss:.4f} | Z_var: {avg_z:.3f} | "
                  f"Q_trk: {avg_qt:5.2f} | Q_hvr: {avg_qh:5.2f} | Time: {avg_t:.2f}s")
    logger.close()


# =========================================================================
# 6. 메인
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--state_form', choices=['error', 'absolute'])
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--p_init', type=float)
    parser.add_argument('--r_init', type=float)
    parser.add_argument('--skip_adam', action='store_true')
    args = parser.parse_args()

    cfg = Config()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from rl.network import apply_tf32_config
    apply_tf32_config(cfg)   # 전역 FP32 + forward만 스코프 TF32
    if args.episodes is not None: cfg.max_episodes = args.episodes
    if args.state_form is not None: cfg.state_form = args.state_form
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.beta is not None: cfg.beta = args.beta
    if args.p_init is not None: cfg.p_init = args.p_init
    if args.r_init is not None:
        cfg.r_init = args.r_init; cfg.r_end = args.r_init
        cfg.r_inv_sqrt = 1.0 / cfg.r_init; cfg.r_inv = 1.0 / (cfg.r_init ** 2)

    cfg.param_str = f"rhukf_{cfg.state_form}_a{cfg.alpha}_b{cfg.beta}_r{cfg.r_init}_p{cfg.p_init}"
    cfg.outdir = f"./results_drone/{cfg.param_str}"; os.makedirs(cfg.outdir, exist_ok=True)

    tp_r = calculate_reward(1, True, rc=cfg.reward)
    tn_r = calculate_reward(0, False, rc=cfg.reward)
    plot_max_reward = cfg.episode_max_steps * max(tp_r, tn_r)

    print(f"\n{'#'*70}\n  🚀 RHUKF-FV ({cfg.state_form}) vs Adam — Realistic Cliff Env")
    print(f"  Setting: {cfg.param_str}\n  Max Theoretical Reward: {plot_max_reward:.1f}\n{'#'*70}")
    comp = ComparisonPlotter(cfg.max_episodes, plot_max_reward, cfg.outdir, cfg.param_str)

    print(f"\n{'='*50}\n  Phase 1: RHUKF-FV\n{'='*50}")
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    env_r = RealisticCliffEnv(cfg, window_size=cfg.window_size)
    agent_r = OnlineRHUKFAgent(cfg); agent_r.warmup_compile()
    log_r = LivePlotter("RHUKF_FV", cfg.max_episodes, plot_max_reward, cfg.outdir, cfg.param_str)
    train_agent(agent_r, env_r, cfg, 'RHUKF', log_r, comp)

    agent_a = None
    if not args.skip_adam:
        print(f"\n{'='*50}\n  Phase 2: Adam DDQN\n{'='*50}")
        np.random.seed(cfg.seed + 1000); torch.manual_seed(cfg.seed + 1000)
        env_a = RealisticCliffEnv(cfg, window_size=cfg.window_size)
        agent_a = AdamDDQNAgent(cfg)
        log_a = LivePlotter("Adam_DDQN", cfg.max_episodes, plot_max_reward, cfg.outdir, cfg.param_str)
        train_agent(agent_a, env_a, cfg, 'Adam', log_a, comp)
    comp.close()

    print(f"\n{'='*50}\n  Phase 3: Q-Landscape\n{'='*50}")
    try:
        if not args.skip_adam and agent_a is not None:
            plot_comparison_4context(agent_r.theta, agent_r.info, agent_a.net, cfg, cfg.param_str)
    except Exception as e:
        print(f"[경고] 지형도 생성 중 오류: {e}")
        import traceback; traceback.print_exc()
    print(f"\n{'#'*70}\n  ✅ 완료! → {cfg.outdir}\n{'#'*70}")


if __name__ == '__main__':
    main()
