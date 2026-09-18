#!/usr/bin/env python3
"""surrogate_env5.1 (2026-09-04) — env5 + **시간 자기상관** (가우시안 코퓰러).

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
        self.dlo, self.dhi = 0.1, 0.7

    def _bin(self, d):
        return int(min(7, max(0, (d - 0.1) / 0.075)))

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
        self.zg = rng.normal(); self.zv = rng.normal()      # 잠재 상태
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

    @staticmethod
    def _phi(z):
        from math import erf, sqrt
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def nis(self, prev_action):
        t = self.t; rng = self.rng
        hov = 'hover' if prev_action == 1 else 'track'
        a = bool(t < self.ep_steps and self.atk[t])
        key = f'{hov}_atk{self._bin(self.dl[t])}' if a else f'{hov}_clean'
        rg = self.RHO_G_ATK if a else self.RHO_G_CLN
        rv = self.RHO_V_ATK if a else self.RHO_V_CLN
        self.zg = rg * self.zg + np.sqrt(max(1e-12, 1 - rg * rg)) * rng.normal()
        self.zv = rv * self.zv + np.sqrt(max(1e-12, 1 - rv * rv)) * rng.normal()
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
