#!/usr/bin/env python3
"""surrogate_env5 (2026-09-04) — FROZEN-ENV v2 **학습 분포** 그대로 이식한 surrogate.

v4 와의 차이 (v4 는 고정정책 스윕 CSV 에서 풀을 뽑아 Isaac 대비 4~5배 쉬웠고,
인공 가우시안 노이즈 σ_v1.04/σ_g0.86 로 억지 정합해야 했다):
  · 풀 소스 = Isaac **학습 로그 17런 192,970 샘플** (FROZEN-v2, TRAIN only)
  · δ 연속(0.101~0.700) 전 구간 — v4 는 {0.1,0.2,0.6,0.7} 4점만 있고 중간은 보간이었다
  · 정책(track/hover)·바람(2-tier)이 학습 그대로 섞여 있음
  · ⇒ **인공 노이즈 불필요.** d′ 이 구성상 Isaac 과 같다 (v 0.28 / g 2.12)

공격 스케줄은 swrl_config FROZEN-v2 확정값을 그대로 따른다:
  burst ON U(40,80) / OFF U(10,30) · 상수공격 30% · δ~U(0.1,0.7) · 에피 300스텝
"""
import os
import numpy as np

_P = None
def _pools():
    global _P
    if _P is None:
        d = np.load(os.path.join(os.path.dirname(__file__), 'train_pool_raw.npz'))
        atk, dl, hov, v, g = d['atk'], d['delta'], d['hover'], d['v'], d['g']
        P = {}
        for h, hname in [(False, 'track'), (True, 'hover')]:
            sel = (hov == h)
            P[f'{hname}_clean'] = np.stack([g[sel & ~atk], v[sel & ~atk]], 1)
            # δ 8구간 (0.1~0.7 을 0.075 폭으로) — 구간 내 샘플이 적으면 이웃과 병합
            edges = np.linspace(0.1, 0.7, 9)
            for i in range(8):
                lo, hi = edges[i], edges[i + 1]
                s = sel & atk & (dl >= lo) & (dl < hi + (1e-9 if i < 7 else 1.0))
                if s.sum() < 50:                      # 희소 구간은 ±1 구간까지 확장
                    s = sel & atk & (dl >= max(0.1, lo - 0.075)) & (dl < hi + 0.075)
                P[f'{hname}_atk{i}'] = np.stack([g[s], v[s]], 1)
        _P = P
    return _P


class SurrogateEnv:
    """v4 와 동일 인터페이스 (reset / nis / step / attack_delay / crashed / atk / dl / bstart)."""
    CLIP = 3.0

    def __init__(self, ep_steps=300, crash=False, seed=0,
                 on_range=(40, 81), off_range=(10, 31)):
        self.ep_steps = ep_steps; self.crash = crash
        self.rng = np.random.default_rng(seed)
        self.on_range = on_range; self.off_range = off_range
        self.P = _pools()
        self.atk_prob = float(os.environ.get('SURR_ATK_PROB', '0.70'))
        self.p_const  = float(os.environ.get('SURR_P_CONST', '0.30'))
        self.dlo, self.dhi = 0.1, 0.7

    def _bin(self, d):
        return int(min(7, max(0, (d - 0.1) / 0.075)))

    def _samp(self, key):
        A = self.P[key]
        return A[self.rng.integers(0, len(A))]

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
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

    def nis(self, prev_action):
        t = self.t
        hov = 'hover' if prev_action == 1 else 'track'
        if t < self.ep_steps and self.atk[t]:
            gv = self._samp(f'{hov}_atk{self._bin(self.dl[t])}'); a = True
        else:
            gv = self._samp(f'{hov}_clean'); a = False
        g = float(np.clip(gv[0], 0.0, self.CLIP)); v = float(np.clip(gv[1], 0.0, self.CLIP))
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        return max(0, t - self.bstart[t]) if (self.atk[t] and self.bstart[t] >= 0) else 0

    def step(self):
        self.t += 1
        return self.t >= self.ep_steps
