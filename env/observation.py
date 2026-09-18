"""env/observation.py — 관측 정의의 단일 출처 (Isaac·surrogate 공용).

환경(Isaac 노드, surrogate)은 **원시 NIS**(ε = rᵀS⁻¹r / n_z, 채널별)만 내보낸다.
정책 입력으로의 변환(압축·클립·정규화·창 쌓기)은 오직 여기서 한다.

    ε̃ = min(f(ε), clip) / div,   f = 'log1p_sqrt' → log(1 + √ε)   (2026-08-20 v5 확정 압축)
    프레임 = [ε̃_vel, ε̃_gyro, prev_action]   (features 로 선택; 'gyro','action' 만 쓰면 GYRO_ONLY)
    상태   = 최근 window 개 프레임을 평탄화 (window 미충전이면 None)

설정 (YAML obs 섹션):
    compress: log1p_sqrt      clip: 4.0      div: 4.0      window: 4      features: [vel, gyro, action]
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

_COMPRESS = {
    'log1p_sqrt': lambda e: math.log1p(math.sqrt(max(e, 0.0))),
    'log1p': lambda e: math.log1p(max(e, 0.0)),
    'none': lambda e: max(e, 0.0),
}


@dataclass
class ObsSpec:
    compress: str = 'log1p_sqrt'
    clip: float = 4.0
    div: float = 4.0
    window: int = 4
    features: List[str] = field(default_factory=lambda: ['vel', 'gyro', 'action'])

    def __post_init__(self):
        if self.compress not in _COMPRESS:
            raise ValueError(f'obs.compress={self.compress!r} (가능: {list(_COMPRESS)})')
        bad = set(self.features) - {'vel', 'gyro', 'action'}
        if bad:
            raise ValueError(f'obs.features 에 모르는 항목 {bad}')
        if self.div <= 0:
            raise ValueError('obs.div 는 양수')

    @property
    def dim(self) -> int:
        return self.window * len(self.features)

    def scale(self, eps_raw: float) -> float:
        """원시 NIS 1개 → 정책 입력 스칼라."""
        return min(_COMPRESS[self.compress](float(eps_raw)), self.clip) / self.div


class ObsBuilder:
    """스텝마다 (원시 NIS vel, 원시 NIS gyro, 직전 행동) → 상태 벡터 (창 미충전이면 None)."""

    def __init__(self, spec: ObsSpec):
        self.spec = spec
        self.buf = collections.deque(maxlen=spec.window)
        self.last_scaled = (0.0, 0.0)

    def reset(self):
        self.buf.clear()
        self.last_scaled = (0.0, 0.0)

    def push(self, eps_vel: float, eps_gyro: float, prev_action: int) -> Optional[np.ndarray]:
        v, g = self.spec.scale(eps_vel), self.spec.scale(eps_gyro)
        self.last_scaled = (v, g)
        vals = {'vel': v, 'gyro': g, 'action': float(prev_action)}
        self.buf.append([vals[f] for f in self.spec.features])
        if len(self.buf) < self.spec.window:
            return None
        return np.asarray(self.buf, dtype=np.float32).reshape(-1)


def compressed_to_raw(x: float, compress: str = 'log1p_sqrt') -> float:
    """구 풀(압축값 저장)을 원시 NIS 로 되돌린다. log1p_sqrt: ε = (eˣ − 1)². 클립에 걸린 값은 복원 불가(하한)."""
    if compress == 'log1p_sqrt':
        return math.expm1(max(x, 0.0)) ** 2
    if compress == 'log1p':
        return math.expm1(max(x, 0.0))
    return max(x, 0.0)
