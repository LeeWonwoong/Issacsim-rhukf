"""
network.py — D3QN/DDQN Network for RHUKF-FV (ported from rhukf.py)
===================================================================
FP32 통일 (TF32 가속). flat theta 벡터 + forward_single / forward_bmm.
FV(full-vector) 전용: FilterCacheFV 만 유지 (node/layer 캐시 제거).

state/forward 규약:
  - theta:  flat 파라미터 벡터 (n_x,) 또는 (n_x, 1)
  - x:      관측 (dimS,) 또는 (dimS, B) 또는 (B, dimS)
  - 출력:   Q values (nA,) 또는 (nA, B)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

DTYPE = torch.float32       # 통일 정밀도 (TF32 가속 대상)
DTYPE_FWD = torch.float32


# ═════════════════════════════════════════════════════════════
#  Activation
# ═════════════════════════════════════════════════════════════
def _get_act_fn(name: str):
    """활성화 이름 → callable (float32 forward 호환)."""
    if name == 'tanh':         return F.tanh
    elif name == 'relu':       return F.relu
    elif name == 'leaky_relu': return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == 'mish':       return F.mish
    elif name == 'gelu':       return F.gelu
    elif name == 'silu':       return F.silu
    else:
        raise ValueError(f"Unknown activation_fn: {name}")


# ═════════════════════════════════════════════════════════════
#  Network Info (flat theta 인덱스 매핑)
# ═════════════════════════════════════════════════════════════
def create_network_info(dimS: int, nA: int, config) -> Dict:
    """D3QN(use_dueling=True) 또는 DDQN(False) 구조를 flat theta로 매핑.
    FV 모드라 filter_layers는 비워둠 (FilterCacheFV가 전체 θ를 한 블록으로 처리)."""
    info = {
        'dimS': dimS, 'nA': nA, 'layers': [],
        'use_dueling': config.use_dueling,
        'act_fn': _get_act_fn(config.activation_fn),
        'act_name': config.activation_fn,
        'use_residual': getattr(config, 'use_residual', False),
    }
    idx = 0

    def add_layers(sizes, type_str):
        nonlocal idx
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W_len = fan_out * fan_in
            b_len = fan_out
            layer = {
                'type': type_str, 'layer_idx': i,
                'W_start': idx, 'W_len': W_len, 'W_shape': (fan_out, fan_in),
                'b_start': idx + W_len, 'b_len': b_len,
                'fan_in': fan_in, 'fan_out': fan_out,
            }
            idx += W_len + b_len
            info['layers'].append(layer)

    shared_out = config.shared_layers[-1] if config.shared_layers else dimS
    add_layers([dimS] + list(config.shared_layers), 'shared')
    info['shared_end_idx'] = len(info['layers'])

    if config.use_dueling:
        add_layers([shared_out] + list(config.value_layers) + [1], 'value')
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + list(config.advantage_layers) + [nA], 'advantage')
    else:
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + list(config.q_layers) + [nA], 'q_layer')

    info['total_params'] = idx
    return info


def initialize_theta(info: Dict, device: str, cfg) -> torch.Tensor:
    """init_scheme(he/xavier/orthogonal)에 따라 flat theta 초기화."""
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    TANH_GAIN = 5.0 / 3.0
    for layer in info['layers']:
        fan_in, fan_out = layer['W_shape'][1], layer['W_shape'][0]
        W_len = layer['W_len']
        l_type, l_idx = layer['type'], layer['layer_idx']
        is_final = (
            (l_type == 'value' and l_idx == len(cfg.value_layers)) or
            (l_type == 'advantage' and l_idx == len(cfg.advantage_layers)) or
            (l_type == 'q_layer' and l_idx == len(cfg.q_layers))
        )
        if cfg.init_scheme == 'orthogonal':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            gain = 0.1 if is_final else float(np.sqrt(2.0))
            torch.nn.init.orthogonal_(W_temp, gain=gain)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        elif cfg.init_scheme == 'xavier':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            gain = 0.1 if is_final else TANH_GAIN
            torch.nn.init.xavier_uniform_(W_temp, gain=gain)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        else:  # 'he'
            theta[layer['W_start']:layer['W_start'] + W_len] = (
                torch.randn(W_len, dtype=DTYPE, device=device) * float(np.sqrt(2.0 / fan_in)))
        # bias = 0
    return theta


# ═════════════════════════════════════════════════════════════
#  Input Normalizer (use_input_norm 항상 ON)
# ═════════════════════════════════════════════════════════════
class InputNormalizer:
    """per-dim scale로 관측 정규화. 드론 NIS는 이미 [0,1]이라 scale=1.0이면 identity."""
    def __init__(self, device, scale=None):
        if scale is None:
            scale = [1.0]  # 안전한 no-op 기본
        self.scale = torch.tensor(scale, dtype=DTYPE, device=device)

    def normalize(self, x):
        if x.dim() == 1:
            return x / self.scale
        elif x.shape[-1] == len(self.scale):
            return x / self.scale
        else:
            return x / self.scale.view(-1, 1)


# ═════════════════════════════════════════════════════════════
#  Filter Cache (Full-Vector)
# ═════════════════════════════════════════════════════════════
class FilterCacheFV:
    """Full-Vector 모드 캐시. 전체 θ ∈ R^n_x 를 한 블록으로 UKF 처리."""
    def __init__(self, info: Dict, cfg, device: str):
        n_x = info['total_params']
        self.n_x = n_x
        self.num_sigma = 2 * n_x + 1

        lam = cfg.alpha ** 2 * (n_x + cfg.kappa) - n_x
        self.gamma_sigma = float(np.sqrt(n_x + lam))
        Wm = np.full(self.num_sigma, 0.5 / (n_x + lam))
        Wc = Wm.copy()
        Wm[0] = lam / (n_x + lam)
        Wc[0] = Wm[0] + (1 - cfg.alpha ** 2 + cfg.beta)
        self.Wm = torch.tensor(Wm, dtype=DTYPE, device=device)
        self.Wc = torch.tensor(Wc, dtype=DTYPE, device=device)

        self.eye_n = torch.eye(n_x, dtype=DTYPE, device=device)
        self.unified_thetas = torch.empty(self.num_sigma, n_x, dtype=DTYPE_FWD, device=device)


# ═════════════════════════════════════════════════════════════
#  Forward Functions
# ═════════════════════════════════════════════════════════════
def forward_single(theta, info, x):
    """단일 theta × obs(들) → Q values. dueling이면 V+(A-mean A)."""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2:
        theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']:
        x = x.t()
    use_resid = info.get('use_residual', False)

    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = info['act_fn'](W @ h + b)
        h = h + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z
    shared_out = h

    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ v + b
        if i == info['value_end_idx'] - 1:
            v = z_lin
        else:
            z = info['act_fn'](z_lin)
            v = v + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z

    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ a + b
        if i == len(info['layers']) - 1:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            a = a + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z

    if info['use_dueling']:
        return (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    return a.to(DTYPE)


def forward_single_with_shared(theta, info, x):
    """forward_single + shared 표현 반환 (진단용)."""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2:
        theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']:
        x = x.t()
    use_resid = info.get('use_residual', False)

    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = info['act_fn'](W @ h + b)
        h = h + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z
    shared_out = h.clone()

    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ v + b
        if i == info['value_end_idx'] - 1:
            v = z_lin
        else:
            z = info['act_fn'](z_lin)
            v = v + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z

    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ a + b
        if i == len(info['layers']) - 1:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            a = a + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z

    if info['use_dueling']:
        Q = (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    else:
        Q = a.to(DTYPE)
    return Q, shared_out.to(DTYPE)


def forward_bmm(thetas, info, x):
    """배치 theta (num_sigma, n_x) × obs → 배치 Q values (num_sigma, nA, B)."""
    thetas = thetas.to(DTYPE_FWD)
    x = x.to(DTYPE_FWD)
    num_sigma = thetas.shape[0]
    use_resid = info.get('use_residual', False)
    x_expanded = x.t().unsqueeze(0).expand(num_sigma, -1, -1)

    h = x_expanded
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = info['act_fn'](torch.bmm(W, h) + b)
        h = h + z if (use_resid and out_dim == in_dim) else z
    shared_out = h

    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z_lin = torch.bmm(W, v) + b
        if i == info['value_end_idx'] - 1:
            v = z_lin
        else:
            z = info['act_fn'](z_lin)
            v = v + z if (use_resid and out_dim == in_dim) else z

    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z_lin = torch.bmm(W, a) + b
        if i == len(info['layers']) - 1:
            a = z_lin
        else:
            z = info['act_fn'](z_lin)
            a = a + z if (use_resid and out_dim == in_dim) else z

    if info['use_dueling']:
        return (v + (a - a.mean(dim=1, keepdim=True))).to(DTYPE)
    return a.to(DTYPE)
