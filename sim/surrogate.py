"""sim/surrogate.py — Isaac 실측 풀을 재생하는 **환경** (관측·보상 계산 없음).

내보내는 것: 원시 NIS (ε_vel, ε_gyro), 공격 참값, 공격 경과 스텝, 추락 여부.
관측 변환은 env/observation.py, 보상은 env/reward.py, 공격 궤적은 env/attack.py 가 담당한다.

풀 형식
  raw        : 키 'g_<빈>' / 'v_<빈>' 에 원시 NIS 표본 (새 Isaac 캡처, 2026-09-18~)
  compressed : 구 풀 v3–v5e — log1p(√ε) 로 압축·SURR_CLIP 에서 잘린 값. 읽을 때 원시로 역변환
               (압축 정의가 같으면 관측은 구 파이프라인과 수치 동일; 클립에 걸린 0.2% 는 하한으로 복원)
빈: {track,hover}_atk_b0..7 (δ 0.1 간격), track_clean, hover_entry, hover_settled, post_{track,hover}, 접미 _ws<n> = 바람 티어.

동역학(구 surrogate_env8 v5·v8 경로를 그대로 옮김 — 난수 소비 순서 동일)
  · 평시/공격 NIS 는 AR(1) 가우시안 코퓰러(ρ) 로 풀 분위수를 뽑는다(시간 상관).
  · 절벽: 공격 궤적 최대 δ≥0.70 이고 온셋부터 데드라인 D 스텝 동안 hover 가 한 번도 없으면
    온셋+lag 에 추락. 종료 확률 p_loc(δ, 바람 티어) × plateau 곡선, D = 데드라인 곡선(δ, 티어) + N(0,2).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.attack import AttackConfig, AttackPlan, sample_attack
from env.observation import compressed_to_raw
from env.scenario import WindConfig, sample_wind


@dataclass
class CrashConfig:
    ploc_delta: List[float] = field(default_factory=lambda: [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.90])
    ploc_prob: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.15, 0.30, 0.95, 1.0, 1.0])
    ploc_tiers: Dict[int, List[float]] = field(default_factory=dict)    # {ws: [δ,p,δ,p,…]} (구 SURR_PLOC_TIERS)
    plateau_curve: bool = True                                          # 구 SURR_PLATEAU_CURVE
    dead_mode: str = 'curve'                                            # curve | uniform (구 SURR_DEAD_MODE)
    dead_uniform: Tuple[int, int] = (10, 20)
    dead_tiers: Dict[int, List[float]] = field(default_factory=dict)    # {ws: [δ,스텝,…]} (구 SURR_DEAD_TIERS)
    lag: Tuple[int, int] = (26, 31)                                     # 온셋→추락 스텝 (양끝 포함)
    lethal_delta: float = 0.70


@dataclass
class SurrogateConfig:
    pool: str = ''
    pool_format: str = 'auto'          # auto | raw | compressed
    pool_compress: str = 'log1p_sqrt'  # compressed 풀의 압축 정의
    pool_clip: float = 4.0             # compressed 풀 값 상한 (구 SURR_CLIP)
    rho_g_atk: float = 0.96
    rho_g_cln: float = 0.5
    rho_v_atk: float = 0.61
    rho_v_cln: float = 0.47
    entry_dwell: int = 1
    reseed_per_episode: bool = True    # 구 SURR_EP_RNG=1: (시드, 에피소드)로 시나리오 난수 재생성 = 짝 무결성
    # ── k-NN 재생(풀 format=knn_v1, Isaac 실측 원시 NIS): 조건 = δ 이력·공격 종료 후 경과·직전 행동과 유지·풍속 ──
    knn_k: int = 32
    knn_rho_atk: float = 0.7           # 공격 중 조건부 잔차의 시간 상관(추세는 조건 특징이 설명) — 검증으로 보정
    crash: CrashConfig = field(default_factory=CrashConfig)

    def __post_init__(self):
        if isinstance(self.crash, dict): self.crash = CrashConfig(**self.crash)
        self.crash.ploc_tiers = {int(k): list(v) for k, v in self.crash.ploc_tiers.items()}
        self.crash.dead_tiers = {int(k): list(v) for k, v in self.crash.dead_tiers.items()}


_BASE = ['track_clean', 'hover_entry', 'hover_settled', 'post_track', 'post_hover'] + \
        [f'{h}_atk_b{i}' for h in ('track', 'hover') for i in range(8)] + [f'{h}_atk_s{i}' for h in ('track', 'hover') for i in range(3)]
_POOL_CACHE: Dict[str, dict] = {}


def load_pool(cfg: SurrogateConfig) -> dict:
    key = (cfg.pool, cfg.pool_format, cfg.pool_compress, cfg.pool_clip)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    d = np.load(cfg.pool)
    if 'format' in d.files and str(d['format']) == 'knn_v2':      # Isaac 실측 k-NN 풀
        X = np.asarray(d['X'], float); rho = json.loads(str(d['rho']))
        Q = dict(_knn=True, X=X, rho=rho, _tiers=[], _has_s=False, _format='knn_v2')
        _POOL_CACHE[key] = Q
        return Q
    fmt = cfg.pool_format
    if fmt == 'auto':
        fmt = str(d['format']) if 'format' in d.files else 'compressed'   # 새 풀은 'format'='raw' 를 기록한다
    if fmt not in ('raw', 'compressed'):
        raise ValueError(f'pool_format={fmt!r}')
    Q = {}
    for f in d.files:
        if f.startswith('g_'):
            g, v = np.asarray(d[f], float), np.asarray(d['v_' + f[2:]], float)
            if fmt == 'compressed':   # 정렬 유지: 단조 변환이라 분위수 순서 보존
                g = np.array([compressed_to_raw(min(max(x, 0.0), cfg.pool_clip), cfg.pool_compress) for x in g])
                v = np.array([compressed_to_raw(min(max(x, 0.0), cfg.pool_clip), cfg.pool_compress) for x in v])
            Q[f[2:]] = (g, v)
    tiers = sorted({int(m.group(1)) for k in Q for m in [re.search(r'_ws(\d+)$', k)] if m})
    E = (np.array([]), np.array([]))
    n = lambda k: len(Q.get(k, E)[0])

    def fix(sfx):   # 구 surrogate_env8 폴백 규칙 그대로
        for k in _BASE: Q.setdefault(k + sfx, E)
        if n('hover_entry' + sfx) < 50 and n('hover_settled' + sfx) >= 50: Q['hover_entry' + sfx] = Q['hover_settled' + sfx]
        if n('hover_settled' + sfx) < 50 and n('hover_entry' + sfx) >= 50: Q['hover_settled' + sfx] = Q['hover_entry' + sfx]
        for pre, m in (('atk_b', 8), ('atk_s', 3)):
            for i in range(m):
                if n(f'track_{pre}{i}{sfx}') < 50:
                    cand = [j for j in range(m) if n(f'track_{pre}{j}{sfx}') >= 50]
                    if cand:
                        j = min(cand, key=lambda j: abs(j - i)); Q[f'track_{pre}{i}{sfx}'] = Q[f'track_{pre}{j}{sfx}']
            for i in range(m):
                if n(f'hover_{pre}{i}{sfx}') < 50: Q[f'hover_{pre}{i}{sfx}'] = Q[f'track_{pre}{i}{sfx}']
        for h in ('track', 'hover'):
            for i in range(3):
                if n(f'{h}_atk_s{i}{sfx}') < 50: Q[f'{h}_atk_s{i}{sfx}'] = Q[f'{h}_atk_b0{sfx}']
        if sfx:
            for k in _BASE:
                if n(k + sfx) < 50: Q[k + sfx] = Q[k]
    fix('')
    for w in tiers: fix(f'_ws{w}')
    Q['_tiers'] = tiers
    Q['_has_s'] = n('track_atk_s1') >= 50
    Q['_format'] = fmt
    _POOL_CACHE[key] = Q
    return Q


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _iq(S, u):
    i = int(u * (len(S) - 1))
    return float(S[min(max(i, 0), len(S) - 1)])


def _tier_curve(tiers: Dict[int, List[float]], ws: int, outer: bool):
    if ws <= 0 or ws not in tiers:
        return None
    f = tiers[ws]
    if outer:   # 구 _sched_tier_curve: 양끝 (0,0)·(0.95,1) 추가
        return [0.0] + f[0::2] + [0.95], [0.0] + f[1::2] + [1.0]
    return f[0::2], f[1::2]


class SurrogateEnv:
    """reset(episode) → 에피소드 시작.  반복: eps_v, eps_g, attack = nis(prev_action); … ; done = step()."""

    def __init__(self, cfg: SurrogateConfig, attack_cfg: AttackConfig, wind_cfg: WindConfig,
                 ep_steps: int, seed: int, reseed_per_episode: Optional[bool] = None):
        self.cfg, self.acfg, self.wcfg = cfg, attack_cfg, wind_cfg
        self.ep_steps = int(ep_steps)
        self.seed0 = int(seed)
        self.rng = np.random.default_rng(seed)
        self.reseed = cfg.reseed_per_episode if reseed_per_episode is None else reseed_per_episode
        self.Q = load_pool(cfg)
        self.ep_idx = -1
        self.knn = bool(self.Q.get('_knn'))
        if self.knn:
            X = self.Q['X']                                # [d0,d1,d3,d6,since_end,act,dwell,ws,dlast,nis_v,nis_g]
            self._F = self._feat(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8])
            self._V, self._G = X[:, 9], X[:, 10]
            self._nn_cache = {}

    @staticmethod
    def _feat(d0, d1, d3, d6, since, act, dwell, ws, dlast):
        """k-NN 거리 척도: δ 0.05 = 1, 공격 종료 후 1 스텝 = 1(15 에서 포화), 행동 불일치 = 100(사실상 정확 일치),
        유지 1 스텝 = 1(5 에서 포화), 풍속 2 m/s = 1, 직전 사건 최대 세기 0.05 = 1(공격 종료 후 15 스텝 동안만 의미)."""
        since = np.asarray(since, float); dl = np.where(since <= 15, np.asarray(dlast, float), 0.0)
        return np.c_[np.asarray(d0) / 0.05, np.asarray(d1) / 0.05, np.asarray(d3) / 0.05, np.asarray(d6) / 0.05,
                     np.minimum(since, 15), np.asarray(act) * 100.0, np.minimum(dwell, 5), np.asarray(ws) / 2.0, dl / 0.05]

    # ── 에피소드 ─────────────────────────────────────────────────────────────
    def reset(self, plan: Optional[AttackPlan] = None, ws: Optional[float] = None):
        """plan/ws 를 주면 그 시나리오를 강제(프로브·캡처 재현용). 난수 순서: 바람 → 코퓰러 초기값 → 공격 → 절벽."""
        if self.reseed:
            self.rng = np.random.default_rng([self.seed0 & 0xFFFFFFFF, self.ep_idx + 1])
        self.ep_idx += 1
        n, rng, cc = self.ep_steps, self.rng, self.cfg.crash
        self.t = 0; self.crashed = False
        w = sample_wind(rng, self.wcfg, self.ep_idx) if ws is None else ws   # ep_idx: 0 기준(축 B 스케줄)
        av = self.Q['_tiers']
        wi = int(w)
        if wi > 0 and av: wi = min(av, key=lambda x: abs(x - wi))
        self.ws = wi; self.sfx = f'_ws{wi}' if wi > 0 else ''
        if self.knn:
            self.ws = float(w); self._act_prev = None; self._act_dwell = 0; self._last_atk = None; self._dlast = 0.0
        self.sg = rng.normal(); self.sv = rng.normal()
        self.plan = sample_attack(rng, self.acfg, n) if plan is None else plan
        dm = self.plan.dmax
        self.lethal = self.plan.has_attack and dm >= cc.lethal_delta
        u = rng.random()                                      # 구 구현(v7 절벽)의 lag 추첨 — v5 에선 아래에서 덮어쓰지만 난수 순서 유지
        rng.integers(4, 11) if u < 0.42 else (rng.integers(11, 26) if u < 0.50 else rng.integers(26, 34))
        p_loc = float(np.interp(dm, cc.ploc_delta, cc.ploc_prob))
        tc = _tier_curve(cc.ploc_tiers, self.ws, outer=True)
        if tc is not None: p_loc = float(np.interp(dm, tc[0], tc[1]))
        self._lag = int(rng.integers(cc.lag[0], cc.lag[1] + 1))
        self._dead = int(rng.integers(cc.dead_uniform[0], cc.dead_uniform[1] + 1)); self._doom = None
        if cc.plateau_curve:
            hold = int(self.plan.active.sum()); pc = float(np.interp(hold, [5, 10, 15, 20, 25], [0.0, 0.4, 0.85, 0.97, 1.0]))
            if dm >= 0.84: pc = float(np.interp(hold, [5, 10, 15], [0.0, 0.6, 1.0]))
            p_loc *= pc
        if cc.dead_mode == 'curve':
            dc = float(np.interp(dm, [0.78, 0.80, 0.82, 0.84, 0.90], [30.0, 20.0, 12.0, 6.0, 4.0]))
            dt = _tier_curve(cc.dead_tiers, self.ws, outer=False)
            if dt is not None: dc = float(np.interp(dm, dt[0], dt[1]))
            self._dead = max(3, int(round(dc + rng.normal(0, 2.0))))
        self._absorb = (rng.random() > p_loc) if self.lethal else True
        self._expose = 0
        self._dwell = -1; self._since_end = 99
        self._t_shift = None; self._ws2 = None                # ★09-20 에피소드 중간 풍속 전환 (난수는 마지막에: 앞 스트림 불변)
        sh = getattr(self.wcfg, 'shift', None)
        if self.knn and sh and rng.random() < float(sh.get('p', 1.0)):
            self._t_shift = int(rng.integers(int(sh['t'][0]), int(sh['t'][1]) + 1)); self._ws2 = float(rng.uniform(*sh['range']))
        return self

    def _bin(self, d): return min(7, max(0, int((d - 0.1) / 0.1)))
    def _sbin(self, d): return int(np.argmin([abs(d - x) for x in (0.03, 0.05, 0.08)]))
    def _key(self, k):
        kk = k + self.sfx
        return kk if kk in self.Q else k

    def attack_delay(self) -> int:
        if self.knn:                                       # Isaac 규약: 스텝 t 라벨 = active[t−1]
            return self.plan.delay(self.t - 1) if self.t >= 1 else 0
        return self.plan.delay(self.t)

    def _nis_knn(self, prev_action: int):
        """Isaac 실측 k-NN 재생. 스텝 t 의 NIS·공격 라벨은 δ[t−1]·active[t−1] (주입이 스텝 끝 발행되는 Isaac 규약)."""
        t, rng, P, c = self.t, self.rng, self.plan, self.cfg
        tt = t - 1
        dl = lambda k: float(P.delta[tt - k]) if tt - k >= 0 else 0.0
        a = bool(tt >= 0 and P.active[tt])
        if a:
            self._dlast = float(P.delta[tt]) if (tt == 0 or not P.active[tt - 1]) else max(self._dlast, float(P.delta[tt]))
            self._last_atk = tt
        since = 0 if a else (tt - self._last_atk if self._last_atk is not None else 99)
        self._act_dwell = self._act_dwell + 1 if prev_action == self._act_prev else 0
        self._act_prev = prev_action
        hov = prev_action == 1
        if self.lethal:                                    # 절벽(구 모델, 시점만 Isaac 규약)
            if a and not hov:
                self._expose += 1
                if self._expose >= self._dead and not self._absorb and self._doom is None:
                    self._doom = int(P.bstart[tt]) + self._lag
            if self._doom is not None and t >= self._doom:
                self.crashed = True
        ws_now = self._ws2 if (self._t_shift is not None and tt >= self._t_shift) else self.ws   # ★09-20 중간 전환 반영
        q = self._feat(dl(0), dl(1), dl(3), dl(6), since, prev_action, self._act_dwell, ws_now, self._dlast)[0]
        key = tuple(np.round(q, 1))
        nn = self._nn_cache.get(key)
        if nn is None:
            dist = ((self._F - q) ** 2).sum(1)
            idx = np.argpartition(dist, c.knn_k)[:c.knn_k]
            nn = (np.sort(self._V[idx]), np.sort(self._G[idx]))
            if len(self._nn_cache) < 200000: self._nn_cache[key] = nn
        rho = self.Q['rho']
        rg = c.knn_rho_atk if a else rho.get('rho_g_cln', 0.5)
        rv = c.knn_rho_atk if a else max(rho.get('rho_v_cln', 0.0), 0.0)
        self.sg = rg * self.sg + math.sqrt(1 - rg * rg) * rng.normal()
        self.sv = rv * self.sv + math.sqrt(1 - rv * rv) * rng.normal()
        return _iq(nn[0], _phi(self.sv)), _iq(nn[1], _phi(self.sg)), a

    def nis(self, prev_action: int):
        """이번 스텝의 (원시 NIS vel, 원시 NIS gyro, 공격 여부). prev_action = 지금까지 실행 중인 행동."""
        if self.knn:
            return self._nis_knn(prev_action)
        t, rng, P = self.t, self.rng, self.plan
        hov = prev_action == 1
        a = bool(t < self.ep_steps and P.active[t])
        d = float(P.delta[t]) if a else 0.0
        self._dwell = (self._dwell + 1 if self._dwell >= 0 else 0) if hov else -1
        self._since_end = 0 if a else min(self._since_end + 1, 99)
        if self.lethal:
            if a and not hov:
                self._expose += 1
                if self._expose >= self._dead and not self._absorb and self._doom is None:
                    self._doom = int(P.bstart[t]) + self._lag
            if self._doom is not None and t >= self._doom:
                self.crashed = True
        hn = 'hover' if hov else 'track'
        entry = 'hover_entry' if self._dwell <= self.cfg.entry_dwell else 'hover_settled'
        if a and d >= 0.1:
            key = f'{hn}_atk_b{self._bin(d)}'
        elif a and d >= 0.02 and self.Q['_has_s']:
            key = f'{hn}_atk_s{self._sbin(d)}'
        elif a and d >= 0.08:
            key = f'{hn}_atk_b0'
        elif a:
            key = 'track_clean' if not hov else entry
        elif 1 <= self._since_end <= 2:
            key = 'post_hover' if hov else 'post_track'
        elif hov:
            key = entry
        else:
            key = 'track_clean'
        c = self.cfg
        rg = c.rho_g_atk if a else c.rho_g_cln
        rv = c.rho_v_atk if a else c.rho_v_cln
        self.sg = rg * self.sg + math.sqrt(1 - rg * rg) * rng.normal()
        self.sv = rv * self.sv + math.sqrt(1 - rv * rv) * rng.normal()
        Sg, Sv = self.Q[self._key(key)]
        return _iq(Sv, _phi(self.sv)), _iq(Sg, _phi(self.sg)), a

    def step(self) -> bool:
        self.t += 1
        return self.crashed or self.t >= self.ep_steps
