"""
rhukf_core.py — Receding-Horizon UKF (Full-Vector, Covariance form)
====================================================================
RHUKF-FV 필터 코어. FP32(TF32 가속). SRRHUIF(정보형) 전부 제거.

제공 함수:
    rhukf_step_fv          : absolute-state, full-vector covariance UKF 1-step
    rhukf_step_fv_error    : error-state(Δμ) full-vector covariance UKF 1-step
    init_error_horizon     : error-state 호라이즌 직전 1회 setup (anchor + Y_cache)
    _resolve_measurement   : q_target / pure_reward 측정값 분기
    compute_per_priorities : 호라이즌 종료 후 |TD|로 PER priority 재계산

PER IS-weight 적용 (per_apply_is_weight=True):
    측정 노이즈 공분산 R 의 샘플별 대각에 곱:  R_eff_i = R_base_i / w_i
    (w_i ∈ (0,1], 과대표집 샘플을 down-weight = R↑ = 칼만이득↓ = 영향↓)

규약:
    theta:        flat (n_x,) 또는 (n_x, 1)
    batch['s']:   [dim_s, B],  batch['s_next']: [dim_s, B]
    forward_bmm:  x 를 [B, dim_s]로 받음 (내부 transpose)
    forward_single: x 가 [dim_s, B]면 그대로, [B, dim_s]면 transpose
"""
import numpy as np
import torch

from .network import forward_single, forward_bmm, DTYPE

JITTER = 1e-6
JITTER_TRIA = 1e-6


# ═════════════════════════════════════════════════════════════
#  Measurement resolver (q_target / pure_reward)
# ═════════════════════════════════════════════════════════════
def _resolve_measurement(cfg, Q_sigma_at_s_a, unified_sigma, info, s_next,
                         a_best_next, Q_tgt_next, reward, term_mask,
                         target_gamma, device, Q_sigma_next_cache=None,
                         fwd_fn=None, q_val_next_override=None):
    """
    q_target:
        Z_sigma_T  = Q(s, a; chi_i)
        z_measured = r + γⁿ (1-term) Q(s', a*; θ_T)
    pure_reward:
        Z_sigma_T  = Q(s, a; chi_i) - γⁿ (1-term) Q(s', a*; chi_i)
        z_measured = r
    """
    B = reward.shape[0]
    dtype_z = Q_sigma_at_s_a.dtype
    not_term = (1.0 - term_mask).to(dtype_z)
    idx = torch.arange(B, device=device)

    if cfg.measurement_mode == 'q_target':
        Z_sigma_T = Q_sigma_at_s_a
        if q_val_next_override is not None:
            q_val_next = q_val_next_override.to(dtype_z)
        else:
            q_val_next = Q_tgt_next[a_best_next, idx].to(dtype_z)
        z_measured = (reward.to(dtype_z) + target_gamma * not_term * q_val_next).view(-1, 1)
        return Z_sigma_T, z_measured, Q_sigma_next_cache

    elif cfg.measurement_mode == 'pure_reward':
        if Q_sigma_next_cache is None:
            if fwd_fn is None:
                raise ValueError("pure_reward 모드는 Q_sigma_next_cache 또는 fwd_fn 필요")
            Q_sigma_next_cache = fwd_fn(unified_sigma, info, s_next)
        q_next_per_sigma = Q_sigma_next_cache[:, a_best_next, idx].to(dtype_z)
        q_next_per_sigma = q_next_per_sigma * not_term.unsqueeze(0)
        Z_sigma_T = Q_sigma_at_s_a - target_gamma * q_next_per_sigma
        z_measured = reward.to(dtype_z).view(-1, 1)
        return Z_sigma_T, z_measured, Q_sigma_next_cache

    else:
        raise ValueError(f"Unknown measurement_mode: {cfg.measurement_mode!r}")


def _apply_is_weight_to_R(R_diag_eff, batch, cfg):
    """PER IS-weight를 R 대각에 적용: R_eff_i = R_base_i / w_i (down-weight only)."""
    if cfg.use_per and cfg.per_apply_is_weight and ('is_weights' in batch):
        w = batch['is_weights'].to(R_diag_eff.dtype).clamp(min=1e-6)
        R_diag_eff = R_diag_eff / w
    return R_diag_eff


# ═════════════════════════════════════════════════════════════
#  Absolute-state RHUKF (Full-Vector, Covariance form)
# ═════════════════════════════════════════════════════════════
def rhukf_step_fv(theta_current_in, theta_target, filter_P_cov, batch, sp,
                  is_first, p_init_val, fv_cache, cfg):
    """Receding-Horizon UKF, full vector, covariance form (full P, no sqrt)."""
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']

    # ── Prior ───────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_pred = sp['theta_init'].clone()
        else:  # 'target'
            theta_pred = theta_target.clone()
    else:
        theta_pred = theta_current_in.clone()
    theta_pred_flat = theta_pred.squeeze()

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    # ── [A] Time update: P_pred = P_prev + Q ────────────────────────
    eye_n = fv_cache.eye_n
    P_prev = (p_init_val * eye_n) if (is_first or filter_P_cov is None) else filter_P_cov
    Q_proc = cfg.q_init * eye_n            # Phase0: 분산 컨벤션(제곱 제거)→P 재팽창 복원
    P_pred = P_prev + Q_proc
    P_pred = 0.5 * (P_pred + P_pred.t())

    # ── [B] Sigma points: chol(P_pred) ──────────────────────────────
    try:
        S_P_pred = torch.linalg.cholesky(P_pred + JITTER_TRIA * eye_n)
    except Exception:
        S_P_pred = torch.linalg.cholesky(P_pred + 1e-4 * eye_n)

    scaled_P = fv_cache.gamma_sigma * S_P_pred
    unified = fv_cache.unified_thetas
    unified[0] = theta_pred_flat.to(torch.float32)
    unified[1:n_x + 1] = (theta_pred_flat.unsqueeze(0) + scaled_P.t()).to(torch.float32)
    unified[n_x + 1:] = (theta_pred_flat.unsqueeze(0) - scaled_P.t()).to(torch.float32)

    # ── [C] Forward sigma points ────────────────────────────────────
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    Z_sigma_T = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)].to(DTYPE)

    # ── [D] DDQN target value ───────────────────────────────────────
    Q_tgt = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)[0]  # [nA, B]
    Q_sigma_next_cache = None
    if is_first:
        if cfg.use_spas:
            Q_sigma_next_cache = forward_bmm(unified, info, s_next)
            a_best_next = Q_sigma_next_cache.mean(dim=0).argmax(dim=0)
        else:
            a_best_next = Q_tgt.argmax(dim=0)
    else:
        a_best_next = forward_single(theta_pred_flat, info, s_next).argmax(dim=0)

    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma

    Z_sigma_T, z_measured, _ = _resolve_measurement(
        cfg, Q_sigma_at_s_a=Z_sigma_T, unified_sigma=unified, info=info, s_next=s_next,
        a_best_next=a_best_next, Q_tgt_next=Q_tgt, reward=batch['r'], term_mask=batch['term'],
        target_gamma=target_gamma, device=device,
        Q_sigma_next_cache=Q_sigma_next_cache, fwd_fn=forward_bmm)
    target_var = torch.var(z_measured).item()

    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [B, 1]
    residual = z_measured - z_hat
    loss = torch.mean(residual ** 2)

    # ── [E] Cross-cov / innovation cov ──────────────────────────────
    Wc_col = fv_cache.Wc.view(-1, 1)
    Z_dev = Z_sigma_T - z_hat.t()
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x + 1] = scaled_P.t()
    X_dev[n_x + 1:] = -scaled_P.t()
    P_zz_sigma = Z_dev.t() @ (Wc_col * Z_dev)   # [B, B]
    P_xz = X_dev.t() @ (Wc_col * Z_dev)         # [n_x, B]

    # ── [F] Huber-adaptive R + PER IS-weight ────────────────────────
    res_abs = torch.abs(residual).squeeze(-1)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    R_diag_eff = current_r_std * adapt_factor   # Phase0: 분산 컨벤션(제곱 제거)
    R_diag_eff = _apply_is_weight_to_R(R_diag_eff, batch, cfg)
    P_zz = P_zz_sigma + torch.diag(R_diag_eff)
    P_zz = 0.5 * (P_zz + P_zz.t())

    # ── [G] Kalman gain ─────────────────────────────────────────────
    eye_batch = torch.eye(batch_sz, dtype=DTYPE, device=device)
    try:
        L_zz = torch.linalg.cholesky(P_zz + JITTER * eye_batch)
    except Exception:
        L_zz = torch.linalg.cholesky(P_zz + 1e-4 * eye_batch)
    tmp = torch.linalg.solve_triangular(L_zz, P_xz.t(), upper=False)
    K_t = torch.linalg.solve_triangular(L_zz.t(), tmp, upper=True)
    K = K_t.t()  # [n_x, B]

    # ── [H] State update ────────────────────────────────────────────
    theta_new_flat = theta_pred_flat + (K @ residual).squeeze(-1)
    if not torch.isfinite(theta_new_flat).all():
        theta_new_flat = theta_pred_flat.clone()
    theta_new = theta_new_flat.view(-1, 1)

    # ── [I] Covariance update: P - K·P_zz·K^T = P - (K·L)(K·L)^T ────
    K_L = K @ L_zz
    P_new = P_pred - K_L @ K_L.t()
    P_new = 0.5 * (P_new + P_new.t())
    if cfg.tikhonov_lambda > 0:
        P_new = P_new + cfg.tikhonov_lambda * eye_n

    # ── [J] Diagnostics ─────────────────────────────────────────────
    P_diag = torch.diagonal(P_new)
    k_gain_norm = torch.norm(K).item()
    innov_abs = torch.abs(residual)
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'avg_P': P_diag.mean().item(),
        'max_P': P_diag.max().item(),
        'ht_norm': torch.norm(P_xz).item(),
        'resid_norm': torch.norm(residual).item(),
        'adapt_ratio': adapt_factor.mean().item(),
    }
    return theta_new, P_new, loss.item(), target_var, k_gain_norm, dbg


# ═════════════════════════════════════════════════════════════
#  Error-state horizon setup
# ═════════════════════════════════════════════════════════════
@torch.no_grad()
def init_error_horizon(theta_active, theta_target, batch_hist, sp, cfg, fv_cache):
    """호라이즌 직전 1회: θ_anchor 결정 + (online_moving이 아니면) Y_cache 일괄 계산."""
    device, info = sp['device'], sp['info']
    B = cfg.batch_size
    N = cfg.N_horizon

    theta_active_flat = theta_active.squeeze().detach().clone()
    theta_target_flat = theta_target.squeeze().detach().clone()
    if cfg.anchor_type == 'target':
        theta_anchor = theta_target_flat.clone()
    elif cfg.anchor_type == 'init':
        theta_anchor = sp['theta_init'].squeeze().detach().clone()
    else:  # 'current'
        theta_anchor = theta_active_flat.clone()

    if cfg.ddqn_argmax == 'online_moving':
        Y_cache = None
        a_best_per_step = None
    else:
        s_next_all = torch.cat([b['s_next'] for b in batch_hist], dim=1)  # [dim_s, N*B]
        if sp.get('normalizer'):
            s_next_all = sp['normalizer'].normalize(s_next_all)
        NB = N * B
        idx_all = torch.arange(NB, device=device)
        Q_tgt_all = forward_single(theta_target_flat, info, s_next_all).to(DTYPE)  # [nA, NB]

        if cfg.ddqn_argmax == 'target':
            a_best_all = Q_tgt_all.argmax(dim=0)
        elif cfg.ddqn_argmax == 'online_frozen':
            Q_online_all = forward_single(theta_active_flat, info, s_next_all).to(DTYPE)
            a_best_all = Q_online_all.argmax(dim=0)
        elif cfg.ddqn_argmax == 'spas':
            n_x = info['total_params']
            num_sigma = 2 * n_x + 1
            spread = fv_cache.gamma_sigma * float(np.sqrt(cfg.p_delta_init))
            eye_n_local = torch.eye(n_x, dtype=DTYPE, device=device)
            sigma_thetas = torch.empty(num_sigma, n_x, dtype=torch.float32, device=device)
            sigma_thetas[0] = theta_anchor.to(torch.float32)
            sigma_thetas[1:n_x + 1] = (theta_anchor.unsqueeze(0) + spread * eye_n_local).to(torch.float32)
            sigma_thetas[n_x + 1:] = (theta_anchor.unsqueeze(0) - spread * eye_n_local).to(torch.float32)
            Q_sigma_all = forward_bmm(sigma_thetas, info, s_next_all.t()).to(DTYPE)
            a_best_all = Q_sigma_all.mean(dim=0).argmax(dim=0)
        else:
            raise RuntimeError(f"Unreachable: ddqn_argmax={cfg.ddqn_argmax}")

        r_all = torch.cat([b['r'] for b in batch_hist], dim=0).to(DTYPE)
        term_all = torch.cat([b['term'] for b in batch_hist], dim=0).to(DTYPE)
        target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
        if cfg.measurement_mode == 'q_target':
            q_val_next_all = Q_tgt_all[a_best_all, idx_all]
            Y_flat = r_all + target_gamma * (1.0 - term_all) * q_val_next_all
        else:  # pure_reward
            Y_flat = r_all
        Y_cache = Y_flat.view(N, B)
        a_best_per_step = a_best_all.view(N, B)

    return {
        'theta_anchor': theta_anchor,
        'theta_target_ref': theta_target_flat,
        'theta_active_ref': theta_active_flat,
        'Y_cache': Y_cache,
        'p_delta_init': cfg.p_delta_init,
        'a_best_per_step': a_best_per_step,
    }


# ═════════════════════════════════════════════════════════════
#  Error-state RHUKF (Full-Vector, Covariance form)
# ═════════════════════════════════════════════════════════════
def rhukf_step_fv_error(filter_state, ctx, batch, h_idx, sp, cfg, fv_cache):
    """error-state(Δμ) covariance UKF 1-step. θ_active = θ_anchor + Δμ."""
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    eye_n = fv_cache.eye_n

    theta_anchor = ctx['theta_anchor']
    Y_cache = ctx['Y_cache']
    p_delta_init_val = ctx['p_delta_init']
    is_first = (h_idx == 0) or (filter_state is None)

    # ── Prior ───────────────────────────────────────────────────────
    if is_first:
        mu_delta_prev = torch.zeros(n_x, dtype=DTYPE, device=device)
        P_delta_prev = p_delta_init_val * eye_n
    else:
        mu_delta_prev = filter_state['mu_delta']
        P_delta_prev = filter_state['P_delta']

    # ── Time update ─────────────────────────────────────────────────
    Q_proc = cfg.q_init * eye_n            # Phase0: 분산 컨벤션(제곱 제거)→P 재팽창 복원
    P_delta_pred = P_delta_prev + Q_proc
    P_delta_pred = 0.5 * (P_delta_pred + P_delta_pred.t())

    # ── Sigma in error space (projected to absolute for forward) ────
    try:
        S_P_pred = torch.linalg.cholesky(P_delta_pred + JITTER_TRIA * eye_n)
    except Exception:
        S_P_pred = torch.linalg.cholesky(P_delta_pred + 1e-4 * eye_n)
    scaled_P = fv_cache.gamma_sigma * S_P_pred
    unified = fv_cache.unified_thetas
    theta_center = theta_anchor + mu_delta_prev
    unified[0] = theta_center.to(torch.float32)
    unified[1:n_x + 1] = (theta_center.unsqueeze(0) + scaled_P.t()).to(torch.float32)
    unified[n_x + 1:] = (theta_center.unsqueeze(0) - scaled_P.t()).to(torch.float32)

    # ── Forward ─────────────────────────────────────────────────────
    s_batch = batch['s'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    Z_sigma_T = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)].to(DTYPE)

    # ── Y target + Z_sigma_T (mode-dispatched) ──────────────────────
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
    s_next_for_y = batch['s_next']
    if sp.get('normalizer'):
        s_next_for_y = sp['normalizer'].normalize(s_next_for_y)
    s_next_bmm = s_next_for_y.t()

    if Y_cache is not None:
        a_best_for_step = ctx['a_best_per_step'][h_idx]
        if cfg.measurement_mode == 'q_target':
            z_measured = Y_cache[h_idx].view(-1, 1).to(DTYPE)
        else:  # pure_reward
            Z_sigma_T, z_measured, _ = _resolve_measurement(
                cfg, Q_sigma_at_s_a=Z_sigma_T, unified_sigma=unified, info=info, s_next=s_next_bmm,
                a_best_next=a_best_for_step, Q_tgt_next=None, reward=batch['r'], term_mask=batch['term'],
                target_gamma=target_gamma, device=device, Q_sigma_next_cache=None, fwd_fn=forward_bmm)
    else:
        # 'online_moving'
        Q_tgt = forward_single(ctx['theta_target_ref'], info, s_next_for_y).to(DTYPE)
        if is_first:
            h0_init = cfg.h0_online_moving_init
            if h0_init == 'theta_target':
                a_best = Q_tgt.argmax(dim=0)
            elif h0_init == 'spas':
                a_best = forward_bmm(unified, info, s_next_bmm).mean(dim=0).argmax(dim=0)
            else:  # 'prev_est'
                a_best = forward_single(ctx['theta_active_ref'], info, s_next_for_y).to(DTYPE).argmax(dim=0)
        else:
            theta_current = theta_anchor + mu_delta_prev
            a_best = forward_single(theta_current, info, s_next_for_y).to(DTYPE).argmax(dim=0)
        Z_sigma_T, z_measured, _ = _resolve_measurement(
            cfg, Q_sigma_at_s_a=Z_sigma_T, unified_sigma=unified, info=info, s_next=s_next_bmm,
            a_best_next=a_best, Q_tgt_next=Q_tgt, reward=batch['r'], term_mask=batch['term'],
            target_gamma=target_gamma, device=device, Q_sigma_next_cache=None, fwd_fn=forward_bmm)

    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()
    target_var = torch.var(z_measured).item()
    residual = z_measured - z_hat
    loss = torch.mean(residual ** 2)

    # ── Cross-cov in error space, P_zz ──────────────────────────────
    Wc_col = fv_cache.Wc.view(-1, 1)
    Z_dev = Z_sigma_T - z_hat.t()
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x + 1] = scaled_P.t()
    X_dev[n_x + 1:] = -scaled_P.t()
    P_zz_sigma = Z_dev.t() @ (Wc_col * Z_dev)
    P_delta_z = X_dev.t() @ (Wc_col * Z_dev)

    res_abs = torch.abs(residual).squeeze(-1)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    R_diag_eff = current_r_std * adapt_factor   # Phase0: 분산 컨벤션(제곱 제거)
    R_diag_eff = _apply_is_weight_to_R(R_diag_eff, batch, cfg)
    P_zz = P_zz_sigma + torch.diag(R_diag_eff)
    P_zz = 0.5 * (P_zz + P_zz.t())

    # ── Kalman gain ─────────────────────────────────────────────────
    eye_batch = torch.eye(batch_sz, dtype=DTYPE, device=device)
    try:
        L_zz = torch.linalg.cholesky(P_zz + JITTER * eye_batch)
    except Exception:
        L_zz = torch.linalg.cholesky(P_zz + 1e-4 * eye_batch)
    tmp = torch.linalg.solve_triangular(L_zz, P_delta_z.t(), upper=False)
    K_t = torch.linalg.solve_triangular(L_zz.t(), tmp, upper=True)
    K = K_t.t()

    # ── State update in error space ─────────────────────────────────
    mu_delta_new = mu_delta_prev + (K @ residual).squeeze(-1)
    if not torch.isfinite(mu_delta_new).all():
        mu_delta_new = mu_delta_prev.clone()

    K_L = K @ L_zz
    P_delta_new = P_delta_pred - K_L @ K_L.t()
    P_delta_new = 0.5 * (P_delta_new + P_delta_new.t())
    if cfg.tikhonov_lambda > 0:
        P_delta_new = P_delta_new + cfg.tikhonov_lambda * eye_n

    theta_active = (theta_anchor + mu_delta_new).view(-1, 1)

    P_diag = torch.diagonal(P_delta_new)
    k_gain_norm = torch.norm(K).item()
    innov_abs = torch.abs(residual)
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'avg_P': P_diag.mean().item(),
        'max_P': P_diag.max().item(),
        'ht_norm': torch.norm(P_delta_z).item(),
        'resid_norm': torch.norm(residual).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        'mu_delta_norm': torch.norm(mu_delta_new).item(),
    }
    filter_state_new = {'mu_delta': mu_delta_new, 'P_delta': P_delta_new}
    return theta_active, filter_state_new, loss.item(), target_var, k_gain_norm, dbg


# ═════════════════════════════════════════════════════════════
#  PER priority recompute (horizon 종료 후)
# ═════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_per_priorities(theta, theta_target, batch_hist, sp, cfg, force=False):
    """최신 theta로 |TD| 재계산 → (indices, |td|). PER off & not force면 (None, None)."""
    if not cfg.use_per and not force:
        return None, None
    device, info = sp['device'], sp['info']
    normalizer = sp.get('normalizer')

    s_all = torch.cat([b['s'] for b in batch_hist], dim=1)
    a_all = torch.cat([b['a'] for b in batch_hist], dim=0)
    r_all = torch.cat([b['r'] for b in batch_hist], dim=0).to(DTYPE)
    s_next_all = torch.cat([b['s_next'] for b in batch_hist], dim=1)
    term_all = torch.cat([b['term'] for b in batch_hist], dim=0).to(DTYPE)
    idx_all = torch.cat([b['indices'] for b in batch_hist], dim=0)

    if normalizer:
        s_all = normalizer.normalize(s_all)
        s_next_all = normalizer.normalize(s_next_all)

    NB = a_all.shape[0]
    arange = torch.arange(NB, device=device)
    target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma

    theta_flat = theta.squeeze().detach()
    theta_target_flat = theta_target.squeeze().detach()
    Q_curr_all = forward_single(theta_flat, info, s_all).to(DTYPE)
    q_sa = Q_curr_all[a_all, arange]
    Q_curr_next = forward_single(theta_flat, info, s_next_all).to(DTYPE)
    a_star = Q_curr_next.argmax(dim=0)

    if cfg.measurement_mode == 'q_target':
        Q_target_next = forward_single(theta_target_flat, info, s_next_all).to(DTYPE)
        q_next_v = Q_target_next[a_star, arange]
        target_y = r_all + target_gamma * (1.0 - term_all) * q_next_v
        td = target_y - q_sa
    else:  # pure_reward
        q_next_h = Q_curr_next[a_star, arange]
        h_w = q_sa - target_gamma * (1.0 - term_all) * q_next_h
        td = r_all - h_w

    return idx_all, td.abs()
