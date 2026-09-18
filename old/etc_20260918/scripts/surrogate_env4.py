# -*- coding: utf-8 -*-
"""surrogate v4 — FROZEN-ENV v2 (2026-09-02) 실측 empirical pool 기반.

   v3(파라메트릭) → v4: 당일 Isaac 실측(weak_overlap·boundary_ws)에서 뽑은
   클래스별 (gyro,vel) obs 풀을 직접 샘플. 시간구조·행동의존을 동결환경 그대로:
     · 바람 2-tier: 약풍 65%(≈clean) / 강풍 35% ws U(4,8) — **윈도우** onset U(20,100)·len U(150,300)
     · 공격 70%: 상수 30% / burst ON U(40,80)·OFF U(10,30), δ~U(0.1,0.7) per-burst
     · δ→obs: 앵커 풀 {0.1,0.2,0.6,0.7} 최근접 2개 선형혼합 샘플
     · 행동의존: prev_action=1(hover) → hover_* 풀 (기동성분 제거·앵커)
     · 온셋랙: 버스트 age 0/1 은 g 상한 (실측 온셋 프로파일)
     · 무추락(동결 원칙) — crashed 항상 False
   인터페이스: surrogate_run.run_config 의 SurrogateEnv 와 동일.
"""
import os
import os
import numpy as np

_POOLS = None
def _pools():
    global _POOLS
    if _POOLS is None:
        p = os.path.join(os.path.dirname(__file__), 'surr_pools_v4.npz')
        _POOLS = {k: v for k, v in np.load(p).items()}
    return _POOLS

_ANCH = [0.1, 0.2, 0.6, 0.7]
_ONSET_CAP = [1.05, 1.9]     # age0/1 g 상한 (실측 온셋 프로파일 근사)

class SurrogateEnv:
    def __init__(self, ep_steps=400, clip=3.0, crash=False, seed=None,
                 on_range=(40, 81), off_range=(10, 31)):
        self.ep_steps = ep_steps; self.clip = clip
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.on_range = on_range; self.off_range = off_range
        _pools()

    def _samp(self, key):
        P = _pools()[key]
        return P[self.rng.integers(0, len(P))]

    def _samp_atk(self, delta, hover):
        # 최근접 2앵커 선형혼합
        pre = 'hover_atk_d' if hover else 'atk_d'
        a = _ANCH
        if delta <= a[0]: lo = hi = a[0]; w = 0.0
        elif delta >= a[-1]: lo = hi = a[-1]; w = 0.0
        else:
            for i in range(len(a) - 1):
                if a[i] <= delta <= a[i + 1]:
                    lo, hi = a[i], a[i + 1]; w = (delta - lo) / (hi - lo); break
        k1 = f'{pre}{str(lo).replace("0.","0")}'; k2 = f'{pre}{str(hi).replace("0.","0")}'
        s1 = self._samp(k1); s2 = self._samp(k2)
        return (1 - w) * s1 + w * s2

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0
        # ── 바람 (2-tier + 강풍 윈도우) ──
        self.strong = rng.random() < 0.35
        if self.strong:
            self.ws = rng.uniform(4.0, 8.0)
            w0 = int(rng.integers(20, 101))
            self.wind_win = (w0, min(w0 + int(rng.integers(150, 301)), n - 5))
            self.wblend = (self.ws / 8.0) ** 2          # 모멘트 ∝ ws² → 풀 혼합확률
        else:
            self.wind_win = None; self.wblend = 0.0
        # ── 공격 ──
        # 2026-09-03 Isaac 정합: 공격 에피 비율 노브 (기본 0.70 = 기존 동작)
        self.has_atk = rng.random() < float(os.environ.get('SURR_ATK_PROB', '0.70'))
        self.atk = np.zeros(n, bool); self.dl = np.zeros(n); self.bstart = np.full(n, -1, int)
        if self.has_atk:
            t0 = int(rng.integers(20, 41))
            if rng.random() < 0.30:                      # 상수 공격
                d = rng.uniform(0.1, 0.7)
                self.atk[t0:n - 5] = True; self.dl[t0:n - 5] = d; self.bstart[t0:n - 5] = t0
            else:
                t = t0
                while t < n - 20:
                    on = int(rng.integers(*self.on_range)); d = rng.uniform(0.1, 0.7)
                    e = min(t + on, n - 5)
                    self.atk[t:e] = True; self.dl[t:e] = d; self.bstart[t:e] = t
                    t = e + int(rng.integers(*self.off_range))
        self.crashed = False

    def _wind_on(self, t):
        return self.wind_win is not None and self.wind_win[0] <= t < self.wind_win[1]

    def nis(self, prev_action):
        t = self.t; rng = self.rng
        hover = (prev_action == 1)
        if t < self.ep_steps and self.atk[t]:
            gv = self._samp_atk(self.dl[t], hover)
            age = t - self.bstart[t]
            if age < len(_ONSET_CAP):                    # 온셋랙
                gv = gv.copy(); gv[0] = min(gv[0], _ONSET_CAP[age] * (0.9 + 0.2 * rng.random()))
            a = True
        else:
            if self._wind_on(t) and rng.random() < self.wblend:
                gv = self._samp('hover_wind' if hover else 'wind')
            else:
                gv = self._samp('hover_clean' if hover else 'clean')
            a = False
        g = float(np.clip(gv[0], 0.0, self.clip)); v = float(np.clip(gv[1], 0.0, self.clip))
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        if self.atk[t] and self.bstart[t] >= 0:
            return max(0, t - self.bstart[t])
        return 0

    def step(self):
        self.t += 1
        return self.t >= self.ep_steps
