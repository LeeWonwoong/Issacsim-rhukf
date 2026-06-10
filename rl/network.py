"""
network.py — DDQN Network for RHUKF-FV (ported from rhukf.py)
===================================================================
정밀도 모델 (rhukf.py와 동일):
  - 전역 기본은 FP32 (allow_tf32=False). DTYPE = DTYPE_FWD = float32.
  - NN forward(matmul/bmm)만 @tf32_forward 데코레이터로 호출 동안 TF32 허용(옵션).
  - 필터 행렬연산(cholesky/qr/solve_triangular)은 이 플래그와 무관하게 항상 FP32.

아키텍처: 순수 DDQN (dueling 제거). shared_layers → q_layers → nA 단일 Q헤드.

state/forward 규약:
  - theta:  flat 파라미터 벡터 (n_x,) 또는 (n_x, 1)
  - x:      관측 (dimS,) 또는 (dimS, B) 또는 (B, dimS)
  - 출력:   Q values (nA,) 또는 (nA, B) / forward_bmm: (num_sigma, nA, B)
"""
import functools
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

torch.set_default_dtype(torch.float32)
DTYPE = torch.float32
DTYPE_FWD = torch.float32


# ═════════════════════════════════════════════════════════════
#  TF32 정책 — 전역 FP32 고정 + forward만 스코프 TF32
# ═════════════════════════════════════════════════════════════
TF32_FORWARD_ENABLED = False  # apply_tf32_config()에서 cfg + 하드웨어 보고 확정


def _tf32_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def apply_tf32_config(cfg):
    """전역 matmul을 FP32로 고정. GPU 지원 + cfg.use_tf32_forward일 때만 forward TF32 활성.
    Returns (enabled, supported)."""
    global TF32_FORWARD_ENABLED
    supported = _tf32_supported()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    TF32_FORWARD_ENABLED = bool(getattr(cfg, 'use_tf32_forward', False) and supported)
    return TF32_FORWARD_ENABLED, supported


def tf32_forward(fn):
    """NN forward 데코레이터: 호출 동안만 TF32 matmul 허용(활성 시), 끝나면 원복.
    비활성/미지원이면 완전 no-op (오버헤드 없음)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not TF32_FORWARD_ENABLED:
            return fn(*args, **kwargs)
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_mm
            torch.backends.cudnn.allow_tf32 = prev_cudnn
    return wrapper


# ═════════════════════════════════════════════════════════════
#  Activation
# ═════════════════════════════════════════════════════════════
def _get_act_fn(name: str):
    if name == 'tanh':         return F.tanh
    elif name == 'relu':       return F.relu
    elif name == 'leaky_relu': return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == 'mish':       return F.mish
    elif name == 'gelu':       return F.gelu
    elif name == 'silu':       return F.silu
    else:
        raise ValueError(f"Unknown activation_fn: {name}")


# ═════════════════════════════════════════════════════════════
#  Network Info (flat theta 인덱스 매핑) — DDQN 전용
# ═════════════════════════════════════════════════════════════
def create_network_info(dimS: int, nA: int, config) -> Dict:
    """순수 DDQN 구조를 flat theta로 매핑: shared_layers → q_layers → nA.
    FV 모드라 filter_layers는 비워둠 (FilterCacheFV가 전체 θ를 한 블록으로 처리)."""
    info = {
        'dimS': dimS, 'nA': nA, 'layers': [],
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
    add_layers([shared_out] + list(config.q_layers) + [nA], 'q_layer')

    info['total_params'] = idx
    return info


def initialize_theta(info: Dict, device: str, cfg) -> torch.Tensor:
    """init_scheme(he/xavier/orthogonal)에 따라 flat theta 초기화. bias=0."""
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    TANH_GAIN = 5.0 / 3.0
    n_layers = len(info['layers'])
    scheme = getattr(cfg, 'init_scheme', 'he')
    for li, layer in enumerate(info['layers']):
        fan_in, fan_out = layer['W_shape'][1], layer['W_shape'][0]
        W_len = layer['W_len']
        is_final = (li == n_layers - 1)   # 전체 스택의 마지막 = Q 출력층
        if scheme == 'orthogonal':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            torch.nn.init.orthogonal_(W_temp, gain=0.1 if is_final else float(np.sqrt(2.0)))
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        elif scheme == 'xavier':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            torch.nn.init.xavier_uniform_(W_temp, gain=0.1 if is_final else TANH_GAIN)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        else:  # 'he'
            theta[layer['W_start']:layer['W_start'] + W_len] = (
                torch.randn(W_len, dtype=DTYPE, device=device) * float(np.sqrt(2.0 / fan_in)))
    return theta


# ═════════════════════════════════════════════════════════════
#  Input Normalizer
# ═════════════════════════════════════════════════════════════
class InputNormalizer:
    """per-dim scale로 관측 정규화. 드론 NIS는 [0,1]이라 scale=1.0이면 identity."""
    def __init__(self, device, scale=None):
        if scale is None:
            scale = [1.0]
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
#  Forward Functions (DDQN, TF32-scoped)
# ═════════════════════════════════════════════════════════════
@tf32_forward
def forward_single(theta, info, x):
    """단일 theta × obs(들) → Q values (nA,) 또는 (nA, B)."""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2:
        theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']:
        x = x.t()
    use_resid = info.get('use_residual', False)
    n_layers = len(info['layers'])

    h = x
    for i in range(n_layers):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ h + b
        if i == n_layers - 1:
            h = z_lin  # 출력층: activation/residual 없음
        else:
            z = info['act_fn'](z_lin)
            h = h + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z
    return h.to(DTYPE)


@tf32_forward
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
    n_layers = len(info['layers'])
    shared_end = info['shared_end_idx']

    h = x
    shared_out = None
    for i in range(n_layers):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z_lin = W @ h + b
        if i == n_layers - 1:
            h = z_lin
        else:
            z = info['act_fn'](z_lin)
            h = h + z if (use_resid and layer['W_shape'][0] == layer['W_shape'][1]) else z
        if i == shared_end - 1:
            shared_out = h.clone()
    if shared_out is None:
        shared_out = x
    return h.to(DTYPE), shared_out.to(DTYPE)


@tf32_forward
def forward_bmm(thetas, info, x):
    """배치 theta (num_sigma, n_x) × obs → 배치 Q values (num_sigma, nA, B)."""
    thetas = thetas.to(DTYPE_FWD)
    x = x.to(DTYPE_FWD)
    num_sigma = thetas.shape[0]
    use_resid = info.get('use_residual', False)
    n_layers = len(info['layers'])
    h = x.t().unsqueeze(0).expand(num_sigma, -1, -1)

    for i in range(n_layers):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z_lin = torch.bmm(W, h) + b
        if i == n_layers - 1:
            h = z_lin
        else:
            z = info['act_fn'](z_lin)
            h = h + z if (use_resid and out_dim == in_dim) else z
    return h.to(DTYPE)
