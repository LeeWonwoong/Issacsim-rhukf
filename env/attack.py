"""env/attack.py — 공격 가족(시나리오) 샘플러의 단일 출처 (Isaac·surrogate 공용).

에피소드 시작 때 공격 궤적 δ(t) 를 **미리 전부** 만든다. 환경은 이 배열을 주입(Isaac: allocator c[0],
surrogate: 풀 빈 선택)할 뿐이다. 난수는 호출자가 넘기는 numpy Generator 하나만 쓴다
(학습기 간 같은 (시드, 에피소드) → 같은 시나리오 = 짝 비교).

가족 (YAML scenario.attack.family)
  v5      : 버스트 1개, 계단형 일정 δ. δ ~ p_upper·U(split, hi) + (1−p_upper)·U(lo, split),
            plateau ~ U(hold), 온셋 ~ U(start).  (2026-09-12 확정 가족 · 09-18 격자 기준)
  profile : 등급(약·전이·강) × 형태(burst·persistent). 초기 prefix 가 형태와 무관하게 같아
            t≈2–3 에서는 관측이 겹치고, 이후 burst 는 사그라들고 persistent 는 커진다(2026-09-18 논의).
            ⚠ 초안 — Isaac 캡처로 NIS 가 실제로 겹쳤다가 갈라지는지 확인한 뒤 확정.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
import numpy as np


@dataclass
class ProfileClass:
    name: str
    weight: float
    delta: Tuple[float, float]          # 기준 세기 d0 범위
    kind: str                           # burst | persistent
    dur: Tuple[int, int]                # 총 활성 스텝 수 범위 (prefix·꼬리 포함)
    grow: Tuple[float, float] = (1.0, 1.0)   # persistent: prefix 이후 d0 → d0·g 로 선형 증가
    grow_steps: int = 10


@dataclass
class AttackConfig:
    p_attack: float = 0.5
    family: str = 'v5'                  # v5 | profile
    start: Tuple[int, int] = (60, 200)  # 온셋 스텝 범위(양끝 포함)
    # ── v5 ──
    delta_lo: float = 0.10
    split: float = 0.72
    delta_hi: float = 0.84
    p_upper: float = 0.5
    hold: Tuple[int, int] = (25, 40)    # plateau 스텝 (YAML 1.1 에서 'on' 은 불리언이라 이름을 hold 로)
    gap: Tuple[int, int] = (25, 40)     # 다음 버스트까지 간격 (구 surrogate 는 버스트 1개여도 한 번 추첨 — 난수 순서 호환용)
    # ── profile ──
    prefix: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9, 1.0])   # d0 대비 초기 램프(형태 공통)
    tail: List[float] = field(default_factory=lambda: [0.7, 0.35])              # burst 꼬리(d0 대비)
    classes: List[ProfileClass] = field(default_factory=list)
    authority_nm: float = 4.36          # δ → N·m (Isaac 주입용)

    def __post_init__(self):
        if self.family not in ('v5', 'profile'):
            raise ValueError(f'scenario.attack.family={self.family!r} (v5|profile)')
        self.classes = [c if isinstance(c, ProfileClass) else ProfileClass(**c) for c in self.classes]
        if self.family == 'profile' and not self.classes:
            raise ValueError('family=profile 이면 scenario.attack.classes 가 필요')
        for c in self.classes:
            if c.kind not in ('burst', 'persistent'):
                raise ValueError(f'profile class {c.name}: kind={c.kind!r}')


@dataclass
class AttackPlan:
    n: int
    active: np.ndarray                  # bool[n]
    delta: np.ndarray                   # float[n] (비활성 0)
    bstart: np.ndarray                  # int[n] 버스트 시작 스텝 (비활성 −1)
    cls: str = 'none'                   # 'none' | 'v5' | profile class name
    direction: Optional[float] = None   # 틸트 방향 α (Isaac 전용; surrogate 는 뽑지 않음)

    @property
    def has_attack(self) -> bool:
        return bool(self.active.any())

    @property
    def dmax(self) -> float:
        return float(self.delta.max()) if self.has_attack else 0.0

    def delay(self, t: int) -> int:
        t = min(t, self.n - 1)
        return max(0, t - int(self.bstart[t])) if (self.active[t] and self.bstart[t] >= 0) else 0


def _empty(n: int) -> AttackPlan:
    return AttackPlan(n, np.zeros(n, bool), np.zeros(n), np.full(n, -1, int))


def _profile(c: ProfileClass, cfg: AttackConfig, d0: float, dur: int, g: float) -> np.ndarray:
    pre = [d0 * f for f in cfg.prefix]
    if c.kind == 'burst':
        tail = [d0 * f for f in cfg.tail]
        hold = max(0, dur - len(pre) - len(tail))
        prof = pre + [d0] * hold + tail
        return np.asarray(prof[:max(dur, 1)])
    body = []
    for k in range(max(0, dur - len(pre))):
        frac = min(1.0, (k + 1) / max(c.grow_steps, 1))
        body.append(d0 * (1.0 + (g - 1.0) * frac))
    return np.asarray((pre + body)[:max(dur, 1)])


def sample_attack(rng: np.random.Generator, cfg: AttackConfig, n: int, with_direction: bool = False) -> AttackPlan:
    """에피소드 공격 궤적. 난수 소비 순서는 family=v5 에서 구 surrogate_env8 와 동일(회귀 검증)."""
    has = rng.random() < cfg.p_attack
    if not has:
        plan = _empty(n)
    elif cfg.family == 'v5':
        plan = _empty(n); emax = n - 5
        t = int(rng.integers(cfg.start[0], cfg.start[1] + 1))
        if t < emax:                                        # 구 구현: 온셋이 emax 이상이면 이후 추첨 없음
            d = rng.uniform(cfg.split, cfg.delta_hi) if rng.random() < cfg.p_upper else rng.uniform(cfg.delta_lo, cfg.split)
            hold = int(rng.integers(cfg.hold[0], cfg.hold[1] + 1)); e = min(t + hold, emax)
            plan.delta[t:e] = d; plan.active[t:e] = True; plan.bstart[t:e] = t
            rng.integers(cfg.gap[0], cfg.gap[1] + 1)       # 구 구현의 다음 버스트 간격 추첨(버스트 1개라 미사용) — 난수 순서 유지
        plan.cls = 'v5'
    else:
        w = np.array([c.weight for c in cfg.classes], float)
        c = cfg.classes[int(rng.choice(len(w), p=w / w.sum()))]
        d0 = float(rng.uniform(*c.delta)); dur = int(rng.integers(c.dur[0], c.dur[1] + 1)); g = float(rng.uniform(*c.grow))
        t = int(rng.integers(cfg.start[0], cfg.start[1] + 1))
        prof = _profile(c, cfg, d0, dur, g)
        e = min(t + len(prof), n - 5)
        plan = _empty(n)
        if e > t:
            plan.delta[t:e] = np.clip(prof[:e - t], 0.0, 1.0); plan.active[t:e] = True; plan.bstart[t:e] = t
        plan.cls = c.name
    if with_direction:
        plan.direction = float(rng.uniform(0.0, 2.0 * math.pi))
    return plan
