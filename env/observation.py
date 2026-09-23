"""env/observation.py — 관측 정의의 단일 출처 (Isaac·surrogate 공용).

환경(Isaac 노드, surrogate)은 **원시 NIS**(ε = rᵀS⁻¹r / n_z, 채널별)만 내보낸다.
정책 입력으로의 변환(압축·클립·정규화·창 쌓기)은 오직 여기서 한다.

    ε̃ = min(f(ε), clip) / div,   f = 'log1p_sqrt' → log(1 + √ε)   (2026-08-20 v5 확정 압축)
    프레임 = [ε̃_vel, ε̃_gyro, prev_action]   (features 로 선택; 'gyro','action' 만 쓰면 GYRO_ONLY)
    상태   = 최근 window 개 프레임을 평탄화 (window 미충전이면 None)

설정 (YAML obs 섹션):
    compress: log1p_sqrt      clip: 4.0      div: 4.0      window: 4      features: [vel, gyro, action]
    noise_std: 0.0            noise_seed: 0          ← ★09-23 입력 인위 잡음 시험(LL-6 대응, 부록 스트레스)

입력 잡음(noise_std > 0): 압축 뒤 vel·gyro 스칼라에 N(0, σ²) 가산(행동 칸 제외, 클립 없음).
    프레임마다 1회 뽑아 창에 저장 → s_t 와 s_{t+1} 이 같은 잡음 프레임을 공유(LL-6 의 obs←noisy obs 와 같은 일관성).
    난수는 push 순서가 아니라 key=(흐름, 에피소드, 스텝) 카운터로 뽑는다 → 학습기가 달라도 같은 (ep, step) 에 같은 잡음(공통난수),
    HARD 리셋·결측 에피에도 정렬 유지. last_scaled 는 깨끗한 값(로그 v_obs·g_obs), last_noise 에 잡음값.
    noise_std = 0 이면 이전과 비트 동일.
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
    noise_std: float = 0.0          # ★09-23 입력 인위 잡음 σ (압축 뒤, vel·gyro). 0 = 끔
    noise_seed: int = 0

    def __post_init__(self):
        if self.compress not in _COMPRESS:
            raise ValueError(f'obs.compress={self.compress!r} (가능: {list(_COMPRESS)})')
        bad = set(self.features) - {'vel', 'gyro', 'action'}
        if bad:
            raise ValueError(f'obs.features 에 모르는 항목 {bad}')
        if self.div <= 0:
            raise ValueError('obs.div 는 양수')
        if float(self.noise_std) < 0:
            raise ValueError('obs.noise_std 는 0 이상')

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
        self.last_noise = (0.0, 0.0)
        self._nz_count = 0          # key 없이 호출될 때의 대체 카운터(잡음 켰을 때만 사용)

    def reset(self):
        self.buf.clear()
        self.last_scaled = (0.0, 0.0)
        self.last_noise = (0.0, 0.0)

    def _noise(self, key) -> tuple:
        sd = float(self.spec.noise_std)
        if sd <= 0.0:
            return (0.0, 0.0)
        if key is None:
            self._nz_count += 1
            key = (9, self._nz_count)
        rng = np.random.default_rng([int(self.spec.noise_seed) & 0xFFFFFFFF] + [int(k) & 0xFFFFFFFF for k in key])
        z = rng.normal(0.0, sd, 2)
        return (float(z[0]), float(z[1]))

    def push(self, eps_vel: float, eps_gyro: float, prev_action: int, key=None) -> Optional[np.ndarray]:
        """key = (흐름, 에피소드, 스텝) 정수 튜플 — 입력 잡음(noise_std>0)의 공통난수 카운터. 잡음 끔이면 무시."""
        v, g = self.spec.scale(eps_vel), self.spec.scale(eps_gyro)
        self.last_scaled = (v, g)
        nzv, nzg = self._noise(key)
        self.last_noise = (nzv, nzg)
        vals = {'vel': v + nzv, 'gyro': g + nzg, 'action': float(prev_action)}
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
