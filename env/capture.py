"""env/capture.py — Isaac 캡처 (학습 없음, 스크립트 정책). 두 모드:

  pairs : 분기 검증 — burst/persistent 짝 목록(아래). 설계 검증 Step 1.
  pool  : 풀 수집 — 온라인 RL 과 같은 시나리오 샘플러(5 패턴 · 바람 · 6 공격 그룹)로 에피소드를 돌리며, 행동 상태
          (track · hover 진입/유지 · hover→track 복귀 · 평시 오탐 hover)를 스크립트 정책 혼합으로 덮는다 → 새 surrogate 풀 원자료.

목적(2026-09-18 설계 검증 Step 1): 정책이 실제로 받는 관측 — 원시 NIS_vel·NIS_gyro 와 4-스텝 쌓은 관측 — 에서
    o_{t:t+3}^burst ≈ o_{t:t+3}^persistent   (prefix 구간: 겹침)
    o_{t+4:t+k}^burst ≠ o_{t+4:t+k}^persistent (이후: 갈라짐)
가 성립하는지 본다.

짝(pair) 설계: 한 짝의 두 에피소드는 d0(기준 세기)·성장률 g·온셋·틸트 방향 α·비행 패턴·풍속을 **공유**하고
형태(burst | persistent)만 다르다 → 관측 차이를 형태에 귀속. 등급마다 pairs_per_grade 짝 + 무공격 n_none 에피소드.
에피소드는 온셋 + post 스텝에서 끝낸다(분기 확인에 필요한 구간만).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math
import numpy as np

from env.attack import AttackConfig, AttackPlan, ProfileClass, _empty, _profile


@dataclass
class CaptureConfig:
    enabled: bool = False
    mode: str = 'pairs'                   # pairs(분기 검증) | pool(풀 수집)
    policy: str = 'track'                 # pairs 정책: track(무대응: 관측 자연 전개) | hover_at:K (온셋+K 스텝부터 hover 고정)
    episodes: int = 400                   # pool: 에피소드 수 (길이는 run.ep_steps)
    # pool: 에피소드마다 한 정책을 가중 추첨 (이름 문법은 make_policy 참고)
    policies: Dict[str, float] = field(default_factory=lambda: {
        'track': 0.25, 'hold:2': 0.15, 'hold:5': 0.15, 'window:3:8': 0.2, 'dither:0.03': 0.25})
    onset: int = 60                       # 공격 시작 스텝 (UKF·비행 안정 이후)
    post: int = 40                        # 온셋 뒤 기록 스텝 → 에피소드 길이 = onset + post
    pairs_per_grade: int = 12
    n_none: int = 12                      # 무공격 에피소드(평시 NIS 기준선)
    grades: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'weak': (0.15, 0.35), 'trans': (0.35, 0.60), 'strong': (0.72, 0.84)})
    grow: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'weak': (1.8, 2.2), 'trans': (1.3, 1.5), 'strong': (1.0, 1.0)})
    grow_steps: int = 4                   # 예시 persistent: .35 → .45 → .55 → .65 (스텝당 ≈ +0.1)
    burst_dur: int = 6                    # prefix(4) + 꼬리(2) = 예시 burst (.1 .2 .3 .35 .25 .1 0). 겹침 구간이 가장 긴(가장 애매한) 경우
    wind_range: Tuple[float, float] = (0.0, 6.0)
    patterns: List[str] = field(default_factory=lambda: ['waypoint', 'circle', 'figure8', 'aggressive', 'scurve'])
    seed: int = 7

    def __post_init__(self):
        self.grades = {k: tuple(v) for k, v in self.grades.items()}
        self.grow = {k: tuple(v) for k, v in self.grow.items()}
        if self.mode not in ('pairs', 'pool'):
            raise ValueError(f'capture.mode={self.mode!r} (pairs|pool)')
        if not (self.policy == 'track' or self.policy.startswith('hover_at:')):
            raise ValueError(f'capture.policy={self.policy!r} (track | hover_at:K)')
        for nm in self.policies:
            make_policy(nm, np.random.default_rng(0))      # 문법 검사
        if set(self.grades) != set(self.grow):
            raise ValueError('capture.grades 와 capture.grow 의 등급 이름이 같아야 한다')


def build_capture_list(cc: CaptureConfig, acfg: AttackConfig, authority_nm: float) -> List[dict]:
    """에피소드 순서대로 시나리오 dict 목록. 짝은 연속 배치(burst, persistent), 무공격은 등급 사이에 끼운다."""
    rng = np.random.default_rng(cc.seed)
    n = cc.onset + cc.post
    persist_dur = n - cc.onset                     # 기록 구간 끝까지 지속
    out: List[dict] = []
    nones = list(range(cc.n_none))
    per_block = max(1, cc.n_none // max(1, len(cc.grades)))
    for gi, (grade, (lo, hi)) in enumerate(cc.grades.items()):
        for p in range(cc.pairs_per_grade):
            d0 = float(rng.uniform(lo, hi)); g = float(rng.uniform(*cc.grow[grade]))
            alpha = float(rng.uniform(0.0, 2.0 * math.pi)); pat = cc.patterns[int(rng.integers(len(cc.patterns)))]
            ws = float(rng.uniform(*cc.wind_range))
            for kind, dur in (('burst', cc.burst_dur), ('persistent', persist_dur)):
                pc = ProfileClass(name=f'{grade}_{kind}', weight=1.0, delta=(d0, d0), kind=kind, dur=(dur, dur),
                                  grow=(g, g), grow_steps=cc.grow_steps)
                prof = _profile(pc, acfg, d0, dur, g)
                plan = _empty(n); e = min(cc.onset + len(prof), n)
                plan.delta[cc.onset:e] = np.clip(prof[:e - cc.onset], 0.0, 1.0)
                plan.active[cc.onset:e] = True; plan.bstart[cc.onset:e] = cc.onset
                plan.cls = pc.name; plan.direction = alpha
                out.append(dict(pair=f'{grade}{p:02d}', grade=grade, kind=kind, d0=d0, grow=g,
                                pattern=pat, wind_speed=ws, plan=plan))
        for _ in range(per_block if gi < len(cc.grades) - 1 else len(nones)):
            if not nones: break
            nones.pop()
            ws = float(rng.uniform(*cc.wind_range)); pat = cc.patterns[int(rng.integers(len(cc.patterns)))]
            plan = _empty(n); plan.cls = 'none'
            out.append(dict(pair='none', grade='none', kind='none', d0=0.0, grow=1.0, pattern=pat, wind_speed=ws, plan=plan))
    return out


def capture_action(cc: CaptureConfig, step: int) -> int:
    if cc.policy == 'track':
        return 0
    k = int(cc.policy.split(':', 1)[1])
    return int(step >= cc.onset + k)


class Policy:
    """풀 수집용 스크립트 정책. act(step, plan) → 0 track / 1 hover.
       track        : 항상 track (무대응 — 공격 관측 자연 전개, 추락 경로)
       hold:K       : 첫 온셋 + K 스텝부터 hover 고정 (hover 진입 과도·공격 중 hover 유지)
       window:K:R   : 온셋 + K 부터 마지막 공격 스텝 + R 까지 hover, 이후 track (hover→track 복귀 과도·공격 뒤 오탐 구간)
       dither:p     : 매 스텝 확률 p 로 행동 토글 (평시 오탐 hover·짧은 hover·잦은 전환)"""

    def __init__(self, name: str, rng: np.random.Generator):
        self.name = name; self.rng = rng; parts = name.split(':'); self.kind = parts[0]; self.a = 0
        if self.kind == 'track' and len(parts) == 1:
            pass
        elif self.kind == 'hold' and len(parts) == 2:
            self.K = int(parts[1])
        elif self.kind == 'window' and len(parts) == 3:
            self.K, self.R = int(parts[1]), int(parts[2])
        elif self.kind == 'dither' and len(parts) == 2:
            self.p = float(parts[1])
            if not 0.0 <= self.p <= 1.0: raise ValueError(name)
        else:
            raise ValueError(f'capture 정책 이름 {name!r} (track | hold:K | window:K:R | dither:p)')

    def act(self, step: int, plan: AttackPlan) -> int:
        if self.kind == 'track':
            return 0
        if self.kind == 'dither':
            if self.rng.random() < self.p: self.a = 1 - self.a
            return self.a
        on = np.flatnonzero(plan.active)
        if not len(on):
            return 0                                   # 무공격 에피소드: hold/window 는 track 과 같다
        if self.kind == 'hold':
            return int(step >= on[0] + self.K)
        evs = plan.events or [dict(start=int(on[0]), end=int(on[-1]) + 1)]      # window: 사건마다 온셋+K ~ 끝+R
        return int(any(ev['start'] + self.K <= step <= ev['end'] - 1 + self.R for ev in evs))


def make_policy(name: str, rng: np.random.Generator) -> Policy:
    return Policy(name, rng)


def draw_policy(cc: CaptureConfig, rng: np.random.Generator) -> Policy:
    names = list(cc.policies); w = np.array([cc.policies[n] for n in names], float)
    return make_policy(names[int(rng.choice(len(names), p=w / w.sum()))], rng)
