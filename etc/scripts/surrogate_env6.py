#!/usr/bin/env python3
"""surrogate_env6 (2026-09-07) — FROZEN-v3.1 무대 재현.
env5.2 코퓰러 계승 + ① v3 실데이터 풀(행동 조건부: 호버 바닥·진입 과도·종료 감쇠)
② v3.1 공격 스케줄: 버스트 70%(ON 10~20, δ0.1~0.7) + ramp 30%(상승 15~30 → δ0.6~0.8, hold 20~40)
③ 물리 절벽: ramp 고δ 구간에서 track 유지 누적 ≥ lag(13~54) → crashed (에피 절단, done=True)
   호버는 항상 구조(밴드 실측: δ0.8 호버 생존, 0.9는 생성 안 함). 패턴 흡수 12.5% 재현.
설계 근거 = 아티팩트 「SWIRL 강건성 비교 무대」 §2~5. 보상은 surrogate_run(4쌍)이 담당."""
import os
import numpy as np

_Q = None
def _quantiles():
    global _Q
    if _Q is None:
        d = np.load(os.path.join(os.path.dirname(__file__), 'train_pool_v3.npz'))
        Q = {}
        keys = ['track_clean','hover_entry','hover_settled','post_track','post_hover'] + \
               [f'{h}_atk_b{i}' for h in ('track','hover') for i in range(8)]
        for k in keys:
            g, v = d[f'g_{k}'], d[f'v_{k}']
            if len(g) < 50:   # 소표본 폴백 (hover_settled n=7 → entry 와 병합)
                if k == 'hover_settled':
                    g = np.sort(np.r_[g, d['g_hover_entry']]); v = np.sort(np.r_[v, d['v_hover_entry']])
            Q[k] = (g, v)
        _Q = Q
    return _Q

def _iq(S, u):
    i = int(u * (len(S) - 1))
    return float(S[min(max(i, 0), len(S) - 1)])

class SurrogateEnv:
    CLIP = 3.0
    RHO_G_ATK = float(os.environ.get('SURR_RHO_G_ATK', '0.96'))
    RHO_G_CLN = float(os.environ.get('SURR_RHO_G_CLN', '0.5'))   # ★v6: 0.85→0.5 — 평시 초과 run 을 Isaac(최대2)에 정합 (형태학 우선)
    RHO_V_ATK = float(os.environ.get('SURR_RHO_V_ATK', '0.61'))
    RHO_V_CLN = float(os.environ.get('SURR_RHO_V_CLN', '0.47'))

    def __init__(self, ep_steps=400, crash=True, seed=0, on_range=(10, 21), off_range=(25, 41)):
        self.ep_steps = int(os.environ.get('SURR_EP_STEPS', ep_steps))
        self.rng = np.random.default_rng(seed)
        self.on_range, self.off_range = on_range, off_range
        self.Q = _quantiles()
        # ★2026-09-08 단일 사다리꼴 패밀리 (burst/ramp 이산분기 폐기). rise 연속분포가 급작~점진 전 스펙트럼.
        self.atk_prob = float(os.environ.get('SURR_ATK_PROB', '0.70'))
        self.rise_rng = (0, 31)          # U(0,30): rise≈0=급작(옛 burst) … 30=점진(옛 ramp)
        self.hold_rng = (10, 41)         # U(10,40)
        self.off_rng  = (25, 41)         # U(25,40)
        # δ_end 범위 — ⚠ 정밀밴드/지연곡선 실측 후 확정 (밴드하단~δ_max). 잠정 (0.1,0.8).
        self.dend_rng = (float(os.environ.get('SURR_DEND_LO','0.1')), float(os.environ.get('SURR_DEND_HI','0.8')))
        self.band_dl  = float(os.environ.get('SURR_BAND_DL','0.75'))   # 결과성 하단 (crash 임계 δ)
        self.deadline = int(os.environ.get('SURR_DEADLINE','30'))      # 온셋후 무대응 허용 스텝 (지연곡선 실측)

    def _bin(self, d): return min(7, max(0, int((d - 0.1) / 0.1)))

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
        self.sg = rng.normal(); self.sv = rng.normal()
        self.atk = np.zeros(n, bool); self.dl = np.zeros(n); self.bstart = np.full(n, -1, int)
        self.has_atk = rng.random() < self.atk_prob
        if self.has_atk:
            t = int(rng.integers(20, 41)); emax = n - 5
            while t < emax:
                # ── 단일 사다리꼴: rise 연속(0=급작/옛burst … 30=점진/옛ramp), δ_end, hold ──
                rise = int(rng.integers(*self.rise_rng)); hold = int(rng.integers(*self.hold_rng))
                dend = rng.uniform(*self.dend_rng)
                e = min(t + rise + hold, emax)
                for k in range(t, e):
                    self.dl[k] = dend if rise == 0 else dend * min((k - t + 1) / rise, 1.0)
                self.atk[t:e] = True; self.bstart[t:e] = t
                t = e + int(rng.integers(*self.off_rng))
        # ── 절벽 파라미터 (에피소드당 1회 추첨). ⚠ lag/absorb 는 정밀밴드 실측으로 교체 예정.
        self._lag = int(self.rng.integers(13, 55))         # 추락 lag (지속 실측 분포로 교체)
        self._absorb = self.rng.random() < 0.125           # 패턴 흡수 (실측 비율로 교체)
        self._expose = 0                                   # track ∧ 고δ 누적
        # ── 행동 이력 상태
        self._dwell = -1; self._since_end = 99

    @staticmethod
    def _phi(z):
        from math import erf, sqrt
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def nis(self, prev_action):
        t = self.t; rng = self.rng
        hov = prev_action == 1
        a = bool(t < self.ep_steps and self.atk[t])
        d = float(self.dl[t]) if a else 0.0
        # 행동 이력 갱신
        self._dwell = (self._dwell + 1 if self._dwell >= 0 else 0) if hov else -1
        self._since_end = 0 if a else min(self._since_end + 1, 99)
        # ── 절벽: 결과성 δ(≥밴드하단) 구간에서 track 유지 누적 ≥ lag → 추락. rise 무관(사다리꼴 통일).
        #   데드라인: 온셋 후 deadline 스텝 내 무대응이면 이미 늦음(지연곡선 실측). absorb=패턴 흡수.
        if a and (not hov) and d >= self.band_dl:
            self._expose += 1
            if self._expose >= self._lag and not self._absorb:
                self.crashed = True
        # ── 풀 선택 (행동 조건부)
        hn = 'hover' if hov else 'track'
        if a and d >= 0.08:
            key = f'{hn}_atk_b{self._bin(d)}'
        elif a:                                            # ramp 초반 비가시(δ<0.08)
            key = f'{hn}_clean' if not hov else 'hover_entry'
            key = 'track_clean' if not hov else ('hover_entry' if self._dwell <= 2 else 'hover_settled')
        elif 1 <= self._since_end <= 2:
            key = 'post_hover' if hov else 'post_track'
        elif hov:
            key = 'hover_entry' if self._dwell <= 2 else 'hover_settled'
        else:
            key = 'track_clean'
        rg = self.RHO_G_ATK if a else self.RHO_G_CLN
        rv = self.RHO_V_ATK if a else self.RHO_V_CLN
        self.sg = rg * self.sg + np.sqrt(1 - rg * rg) * rng.normal()
        self.sv = rv * self.sv + np.sqrt(1 - rv * rv) * rng.normal()
        Sg, Sv = self.Q[key]
        g = float(np.clip(_iq(Sg, self._phi(self.sg)), 0.0, self.CLIP))
        v = float(np.clip(_iq(Sv, self._phi(self.sv)), 0.0, self.CLIP))
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        return max(0, t - self.bstart[t]) if (self.atk[t] and self.bstart[t] >= 0) else 0

    def step(self):
        self.t += 1
        return self.crashed or self.t >= self.ep_steps
