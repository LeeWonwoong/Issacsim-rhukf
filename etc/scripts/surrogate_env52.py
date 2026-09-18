#!/usr/bin/env python3
"""surrogate_env5.2 (2026-09-05) — env5 + **시간 자기상관** (가우시안 코퓰러).

env5 의 결손: 매 스텝 풀에서 iid 로 뽑아 Isaac 의 시간 상관을 통째로 없앴다.
Isaac 실측 (학습로그 17런, 5스텝 간격 lag-1):
    공격 gyro ρ(5)=0.813 → 1스텝 ρ≈0.96 |  평시 gyro ρ(5)=0.442 → ρ≈0.85
    vel      ρ(5)=0.02~0.09 (거의 백색)  → ρ≈0.47/0.61 (AR(1) 가정 하 환산)
온셋: age 0~4 에서 g_med 1.72 → age 5+ 1.90 평탄 (약한 램프) → age<5 전용 풀로 반영.

방식: 잠재 z_t ~ AR(1)(ρ) 유지 → Φ(z) 로 균등화 → 해당 클래스 풀의 **경험 분위수**로 역변환.
      ⇒ 주변분포는 env5 와 **정확히 동일**(= Isaac 정합 유지), 상관만 추가.
      클래스가 바뀌어도 z 는 연속 유지 → 온셋/오프셋 전이가 매끄럽다.
"""
import os
import numpy as np

_Q = None
def _quantiles():
    """클래스별 (gyro, vel) 정렬 배열 = 경험 분위수 함수."""
    global _Q
    if _Q is None:
        d = np.load(os.path.join(os.path.dirname(__file__), 'train_pool_raw.npz'))
        atk, dl, hov, v, g = d['atk'], d['delta'], d['hover'], d['v'], d['g']
        # 버스트 age 는 원 로그에 없으므로 δ 구간만 사용 + age<5 전용은 δ 무관 온셋 풀로 근사
        Q = {}
        edges = np.linspace(0.1, 0.7, 9)
        for h, hn in [(False, 'track'), (True, 'hover')]:
            sel = (hov == h)
            for nm, s in [(f'{hn}_clean', sel & ~atk)]:
                Q[nm] = (np.sort(g[s]), np.sort(v[s]))
            for i in range(8):
                lo, hi = edges[i], edges[i + 1]
                s = sel & atk & (dl >= lo) & (dl < hi + (1e-9 if i < 7 else 1.0))
                if s.sum() < 50:
                    s = sel & atk & (dl >= max(0.1, lo - 0.075)) & (dl < hi + 0.075)
                Q[f'{hn}_atk{i}'] = (np.sort(g[s]), np.sort(v[s]))
        _Q = Q
    return _Q


def _iq(sorted_arr, u):
    """경험 분위수 역변환."""
    i = int(u * (len(sorted_arr) - 1))
    return float(sorted_arr[min(max(i, 0), len(sorted_arr) - 1)])


class SurrogateEnv:
    CLIP = 3.0
    # 1스텝 AR(1) 계수 (Isaac 실측 5스텝 상관에서 환산)
    RHO_G_ATK = float(os.environ.get('SURR_RHO_G_ATK', '0.96'))
    RHO_G_CLN = float(os.environ.get('SURR_RHO_G_CLN', '0.85'))
    RHO_V_ATK = float(os.environ.get('SURR_RHO_V_ATK', '0.61'))
    RHO_V_CLN = float(os.environ.get('SURR_RHO_V_CLN', '0.47'))

    def __init__(self, ep_steps=300, crash=False, seed=0,
                 on_range=(40, 81), off_range=(10, 31)):
        self.ep_steps = ep_steps; self.crash = crash
        self.rng = np.random.default_rng(seed)
        self.on_range = on_range; self.off_range = off_range
        self.Q = _quantiles()
        self.atk_prob = float(os.environ.get('SURR_ATK_PROB', '0.70'))
        self.p_const = float(os.environ.get('SURR_P_CONST', '0.30'))
        # ★2026-09-05 과제 난이도 축: δ 상한을 낮추면 공격이 약해져 d′ 가 내려간다.
        #   오염(alias)과 달리 **클래스 혼동을 만들지 않고** 신호 자체를 약하게 한다 =
        #   'TD 이상치 노이즈'가 아니라 '학습 난이도' 축.
        self.dlo = float(os.environ.get('SURR_DELTA_LO', '0.1'))
        self.dhi = float(os.environ.get('SURR_DELTA_HI', '0.7'))
        # ★2026-09-05 인스턴스 시점에 읽는다 — 클래스 속성이면 import 1회로 고정되어
        #   런마다 환경변수를 바꿔도 반영되지 않는다(실제 사고: alias 스윕 72런이 전부 rate=0).
        self.SLOW_W     = float(os.environ.get('SURR_SLOW_W', '1.0'))
        self.ALIAS_RATE = float(os.environ.get('SURR_ALIAS_RATE', '0'))
        self.ALIAS_LEN  = tuple(int(x) for x in os.environ.get('SURR_ALIAS_LEN', '1,3').split(','))
        self.ALIAS_BIN  = int(os.environ.get('SURR_ALIAS_BIN', '4'))

    def _bin(self, d):
        return int(min(7, max(0, (d - 0.1) / 0.075)))

    # ── 2성분 잠재: z = √w·(느린 AR1) + √(1−w)·(빠른 백색) ────────────────
    #   AR(1) 단일 성분은 ρ₅=0.813 을 맞추면 ρ₁=0.96 이 되어 연속프레임이 거의 같아지고
    #   **창(히스토리)의 가치가 0** 이 된다 (실측 d′(W8)/d′(W1)=1.01).
    #   Isaac 은 프레임당 d′ 2.12 → 작동 d′ ~2.97 로 창이 1.4배를 벌어준다.
    #   백색 성분 비율 (1−w) 이 곧 **POMDP 강도 노브**: 클수록 한 프레임으로는 못 풀고
    #   시간 적분이 필요해진다. w 는 SURR_SLOW_W (기본 1.0 = 기존 env5.1 동작).
    # ── POMDP 기제 ②: 기동 aliasing ──────────────────────────────────
    #   평시 구간에 **공격 풀에서 뽑은 짧은 스파이크**(길이 L)를 삽입한다.
    #   한 프레임으로는 공격과 구별 불가(같은 분포에서 나옴). 구별 정보는 **지속시간**뿐:
    #   공격은 40~80스텝 지속, alias 는 L(2~4)스텝 만에 끝난다.
    #   ⇒ W=1 은 원리적으로 못 풀고 W≥4 는 풀 수 있는 = 정의 그대로의 부분관측성.
    #   ALIAS_RATE = 100스텝당 기대 발생 횟수 (0 = 비활성 = env5.1 동작)
    #   ★길이는 창(W=4)보다 짧아야 한다. 창 전체가 alias 로 덮이면 원리적으로 구별 불가
    #   (= irreducible error) 이지 부분관측성이 아니다. env 기본 (1,3) = 1~2스텝.

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
        self.sg = rng.normal(); self.sv = rng.normal()      # 느린 성분
        self.zg = self.sg; self.zv = self.sv
        self.has_atk = rng.random() < self.atk_prob
        self.atk = np.zeros(n, bool); self.dl = np.zeros(n); self.bstart = np.full(n, -1, int)
        if self.has_atk:
            t0 = int(rng.integers(20, 41)); emax = n - 5
            if rng.random() < self.p_const:
                d = rng.uniform(self.dlo, self.dhi)
                self.atk[t0:emax] = True; self.dl[t0:emax] = d; self.bstart[t0:emax] = t0
            else:
                t = t0
                while t < emax:
                    on = int(rng.integers(*self.on_range)); d = rng.uniform(self.dlo, self.dhi)
                    e = min(t + on, emax)
                    self.atk[t:e] = True; self.dl[t:e] = d; self.bstart[t:e] = t
                    t = e + int(rng.integers(*self.off_range))
        # alias 스케줄 (평시 구간에만)
        self.alias = np.zeros(n, bool)
        if self.ALIAS_RATE > 0:
            k = rng.poisson(self.ALIAS_RATE * n / 100.0)
            for _ in range(int(k)):
                t0 = int(rng.integers(0, n - 6)); L = int(rng.integers(*self.ALIAS_LEN))
                if not self.atk[t0:t0 + L].any():
                    self.alias[t0:t0 + L] = True

    @staticmethod
    def _phi(z):
        from math import erf, sqrt
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def nis(self, prev_action):
        t = self.t; rng = self.rng
        hov = 'hover' if prev_action == 1 else 'track'
        a = bool(t < self.ep_steps and self.atk[t])
        aliased = (not a) and t < self.ep_steps and bool(getattr(self, 'alias', np.zeros(1, bool))[t] if self.ALIAS_RATE > 0 else False)
        if a:
            key = f'{hov}_atk{self._bin(self.dl[t])}'
        elif aliased:
            key = f'{hov}_atk{self.ALIAS_BIN}'      # 공격처럼 보이지만 라벨은 정상
        else:
            key = f'{hov}_clean'
        rg = self.RHO_G_ATK if (a or aliased) else self.RHO_G_CLN
        rv = self.RHO_V_ATK if (a or aliased) else self.RHO_V_CLN
        w = self.SLOW_W
        # 느린 성분의 극을 높여 ρ(5) 는 유지하고 ρ(1) 만 낮춘다: ρ_z(k) = w·ρ_s^k
        rgs = rg ** 0.2 if w >= 1.0 else min(0.9999, (rg / w) ** 0.2) if rg / w < 1 else 0.9999
        rvs = rv ** 0.2 if w >= 1.0 else min(0.9999, (rv / w) ** 0.2) if rv / w < 1 else 0.9999
        rgs, rvs = (rg, rv) if w >= 1.0 else (rgs ** 5 if False else rgs, rvs)
        self.sg = rgs * self.sg + np.sqrt(max(1e-12, 1 - rgs * rgs)) * rng.normal()
        self.sv = rvs * self.sv + np.sqrt(max(1e-12, 1 - rvs * rvs)) * rng.normal()
        if w >= 1.0:
            self.zg, self.zv = self.sg, self.sv
        else:
            self.zg = np.sqrt(w) * self.sg + np.sqrt(1 - w) * rng.normal()
            self.zv = np.sqrt(w) * self.sv + np.sqrt(1 - w) * rng.normal()
        Sg, Sv = self.Q[key]
        g = float(np.clip(_iq(Sg, self._phi(self.zg)), 0.0, self.CLIP))
        v = float(np.clip(_iq(Sv, self._phi(self.zv)), 0.0, self.CLIP))
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        return max(0, t - self.bstart[t]) if (self.atk[t] and self.bstart[t] >= 0) else 0

    def step(self):
        self.t += 1
        return self.t >= self.ep_steps
