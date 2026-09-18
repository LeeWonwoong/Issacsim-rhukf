"""env/scenario.py — 에피소드 시나리오(바람·비행 패턴·공격) 샘플러 (Isaac·surrogate 공용).

난수는 호출자가 넘기는 numpy Generator 하나. Isaac 은 np.random.default_rng([seed, episode]) 를 넘겨
학습기와 무관하게 같은 시나리오를 본다(구 SCENARIO_SEED 대체).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from env.attack import AttackConfig, sample_attack


@dataclass
class WindConfig:
    mode: str = 'tiers'                                  # tiers(이산 ws, surrogate 풀 티어) | uniform(연속, Isaac)
    tiers: Dict[float, float] = field(default_factory=lambda: {0: 1.0})   # {ws: 확률} (작성 순서가 추첨 순서)
    range: Tuple[float, float] = (0.0, 6.0)              # uniform
    kind: str = 'wind_turbulence'                        # Isaac 외란 종류 (ws=0 이면 none)

    def __post_init__(self):
        if self.mode not in ('tiers', 'uniform'):
            raise ValueError(f'scenario.wind.mode={self.mode!r} (tiers|uniform)')
        self.tiers = {float(k): float(v) for k, v in self.tiers.items()}


def sample_wind(rng: np.random.Generator, cfg: WindConfig) -> float:
    if cfg.mode == 'tiers':
        items = list(cfg.tiers.items()); z = sum(p for _, p in items)
        return float(items[int(rng.choice(len(items), p=[p / z for _, p in items]))][0])
    return float(rng.uniform(*cfg.range))


@dataclass
class ScenarioConfig:
    attack: AttackConfig = field(default_factory=AttackConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    patterns: List[str] = field(default_factory=lambda: ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve'])

    def __post_init__(self):
        if isinstance(self.attack, dict): self.attack = AttackConfig(**self.attack)
        if isinstance(self.wind, dict): self.wind = WindConfig(**self.wind)


def sample_isaac_scenario(rng: np.random.Generator, cfg: ScenarioConfig, n_steps: int) -> dict:
    """Isaac 용: 바람(연속/이산) → 공격 궤적(방향 포함) → 패턴."""
    ws = sample_wind(rng, cfg.wind)
    plan = sample_attack(rng, cfg.attack, n_steps, with_direction=True)
    pattern = cfg.patterns[int(rng.integers(len(cfg.patterns)))]
    return dict(pattern=pattern, wind_speed=ws, disturbance_type=(cfg.wind.kind if ws > 0 else 'none'), plan=plan)
