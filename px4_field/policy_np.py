"""policy_np.py — 실기/SITL 추론용 numpy 정책 (torch 불요). export_policy.py 가 만든 npz 를 읽어 Q(s) 를 계산한다.

    pol = NumpyPolicy('models/swirl_v2_s42.npz'); q = pol.q(obs12); a = int(q.argmax())

rl/network.forward_single 과 같은 연산(층별 W@h+b, 은닉 활성, 출력층 무활성). 검증: px4_field/_test_policy_np.py (torch 와 1e-5 일치).
"""
import json
import numpy as np

_ACT = {
    'relu': lambda z: np.maximum(z, 0.0),
    'tanh': np.tanh,
    'silu': lambda z: z / (1.0 + np.exp(-z)),
    'swish': lambda z: z / (1.0 + np.exp(-z)),
    'gelu': lambda z: 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
    'elu': lambda z: np.where(z > 0, z, np.expm1(z)),
    'leaky_relu': lambda z: np.where(z > 0, z, 0.01 * z),
}


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.theta = np.asarray(z['theta'], dtype=np.float64)
        self.layers = [dict(W_start=int(a), W_len=int(b), W_shape=(int(r), int(c)), b_start=int(d), b_len=int(e))
                       for a, b, r, c, d, e in zip(z['W_start'], z['W_len'], z['W_rows'], z['W_cols'], z['b_start'], z['b_len'])]
        self.act_name = str(z['act_name']); self.act = _ACT[self.act_name]
        self.dimS = int(z['dimS']); self.nA = int(z['nA']); self.use_residual = bool(z['use_residual'])
        self.obs = json.loads(str(z['obs_json']))
        self.meta = {k: str(z[k]) for k in ('src', 'net', 'agent', 'seed', 'steps_done') if k in z}
        self.W = [self.theta[l['W_start']:l['W_start'] + l['W_len']].reshape(l['W_shape']) for l in self.layers]
        self.b = [self.theta[l['b_start']:l['b_start'] + l['b_len']] for l in self.layers]

    def q(self, x):
        h = np.asarray(x, dtype=np.float64).reshape(-1)
        assert h.size == self.dimS, f'obs dim {h.size} != {self.dimS}'
        n = len(self.W)
        for i in range(n):
            zl = self.W[i] @ h + self.b[i]
            if i == n - 1:
                h = zl
            else:
                zz = self.act(zl)
                h = h + zz if (self.use_residual and self.W[i].shape[0] == self.W[i].shape[1]) else zz
        return h

    def act_greedy(self, x):
        return int(np.argmax(self.q(x)))
