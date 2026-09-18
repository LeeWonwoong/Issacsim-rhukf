# -*- coding: utf-8 -*-
"""NIS surrogate 환경 v3 — 확정 tilt 프레임워크 정합 (2026-08-28 재작성).

   ★ v2→v3 근본 정정: 채널 구조가 반대였다.
     v2(틀림): gyro 겹침(평시 스파이크 2.2 ↔ 공격 2.15) = gyro-POMDP.
               → 옛 torque 스윕 + 공격종료 잔향 오염 표본에서 나온 것.
     v3(확정): 아티팩트 "Aliasing은 진폭이 아니라 타이밍"(08-26) + tilt 밴드 스윕.
       · gyro TRIVIAL(d′16.9): 순수평시 med0.44/p90 0.75, 공격 온셋3스텝후 2.0~3.0(δ0.6+ 클립).
         aliasing 봉우리 압축 ~0.41 = 탐지장벽 아님, 옵티마이저 학습노이즈.
       · POMDP = ①온셋0-2스텝 블라인드(gyro0.76) ②vel 오프셋드래그(20-40스텝) ③vel 약채널(d′~4)
       · δ~U(0.1,0.8), gyro δ스케일(δ0.1 0.96→δ0.6+ 클립3.0), vel 약스케일.
"""
import os
import numpy as np

GB_MEAN, GB_STD = 0.32, 0.13
GB_CLIP = 1.5
VB_MEAN, VB_STD = 0.37, 0.18
ALIAS_PEAK = 0.22
P_ALIAS_MAX = 0.08
ALIAS_LEN_MEAN = 2.0
def g_steady(delta): return min(0.55 + 3.9 * delta, 3.0)
ONSET_G = [0.76, 1.5, 2.0]   # age0 블라인드 → age1-2 상승
G_STD = 0.22
def v_steady(delta): return 0.37 + 0.5 * delta
V_STD = 0.16
OFF_TAU_G = 8.0
OFF_TAU_V = 30.0
MAN_HOVER_DECAY = 0.55
MAN_TRACK_TAU = 0.4
CRASH_DRIFT_TH = 14.0
CRASH_DRIFT_PER_FN = 1.0


class SurrogateEnv:
    """확정 채널: gyro=온셋랙+trivial, vel=드래그 POMDP."""

    def __init__(self, ep_steps=400, clip=3.0, crash=False, seed=None, on_range=(40,81), off_range=(10,31)):
        self.ep_steps = ep_steps
        self.clip = clip
        self.crash_on = crash
        self.rng = np.random.default_rng(seed) if seed is not None else np.random
        self.on_range = on_range; self.off_range = off_range

    def _rand(self):
        return self.rng.random() if hasattr(self.rng, 'random') else np.random.rand()

    def _randn(self):
        return self.rng.standard_normal() if hasattr(self.rng, 'standard_normal') else np.random.randn()

    def _randint(self, lo, hi):
        return int(self.rng.integers(lo, hi)) if hasattr(self.rng, 'integers') else np.random.randint(lo, hi)

    def _geom(self, p):
        return int(self.rng.geometric(p)) if hasattr(self.rng, 'geometric') else int(np.random.geometric(p))

    def _make_profile(self):
        n = self.ep_steps
        aggr = self._rand() < 0.5
        base = 0.30 if aggr else 0.13
        peak_h = 0.75 if aggr else 0.32
        n_turns = self._randint(4, 9) if aggr else self._randint(2, 5)
        prof = np.full(n, base)
        idx = np.arange(n)
        for _ in range(n_turns):
            c = self._randint(0, n); w = self._randint(6, 20)
            prof += peak_h * np.exp(-0.5 * ((idx - c) / w) ** 2)
        return np.clip(prof, 0.0, 1.0), aggr

    def reset(self):
        n = self.ep_steps
        self.t = 0
        self.m_profile, self.aggr = self._make_profile()
        self.man_energy = float(self.m_profile[0])
        self.delta = 0.1 + 0.7 * self._rand()
        self.has_atk = self._rand() > 0.30
        self.atk = np.zeros(n, dtype=bool)
        self.burst_start_of = np.full(n, -1, dtype=int)
        if self.has_atk:
            t = self._randint(80, 150)
            while t < n - 20:
                on = self._randint(*self.on_range); off = self._randint(*self.off_range)
                e = min(t + on, n)
                self.atk[t:e] = True; self.burst_start_of[t:e] = t
                t = e + off
        self.alias_rem = 0; self.alias_amp = 0.0
        self.off_g = 0.0; self.off_v = 0.0
        self.drift = 0.0; self.crashed = False
        return

    def _update_man(self, prev_action):
        if prev_action == 1:
            self.man_energy *= MAN_HOVER_DECAY
        else:
            tgt = float(self.m_profile[min(self.t, self.ep_steps - 1)])
            self.man_energy += MAN_TRACK_TAU * (tgt - self.man_energy)
        self.man_energy = float(np.clip(self.man_energy, 0.0, 1.0))

    def nis(self, prev_action):
        t = self.t
        a = bool(self.atk[t]) if t < self.ep_steps else False
        self._update_man(prev_action)
        d = self.delta

        if a:
            bs = self.burst_start_of[t]; age = t - bs if bs >= 0 else 0
            if age < 3:
                g = ONSET_G[age] + G_STD * self._randn()
            else:
                g = g_steady(d) + G_STD * self._randn()
            v = v_steady(d) + V_STD * self._randn()
            self.off_g = max(self.off_g, g_steady(d))
            self.off_v = max(self.off_v, v_steady(d))
            if prev_action == 0:
                self.drift += CRASH_DRIFT_PER_FN * (0.3 + d)
            else:
                self.drift = max(0.0, self.drift - 1.5)
        else:
            g = GB_MEAN + GB_STD * self._randn(); g = min(g, GB_CLIP)
            v = VB_MEAN + VB_STD * self._randn()
            if self.off_g > GB_MEAN:
                self.off_g = GB_MEAN + (self.off_g - GB_MEAN) * np.exp(-1.0 / OFF_TAU_G)
                g = max(g, self.off_g + 0.12 * self._randn())
            if self.off_v > VB_MEAN:
                self.off_v = VB_MEAN + (self.off_v - VB_MEAN) * np.exp(-1.0 / OFF_TAU_V)
                v = max(v, self.off_v + 0.10 * self._randn())
            if self.alias_rem > 0:
                self.alias_rem -= 1; g = max(g, GB_MEAN + self.alias_amp + 0.1 * self._randn())
            elif self._rand() < P_ALIAS_MAX * self.man_energy:
                self.alias_rem = max(0, self._geom(1.0 / ALIAS_LEN_MEAN) - 1)
                self.alias_amp = ALIAS_PEAK * (0.6 + 0.6 * self._rand())
                g = max(g, GB_MEAN + self.alias_amp)
            self.drift = max(0.0, self.drift - 2.0)

        v = float(np.clip(v, 0.0, self.clip)); g = float(np.clip(max(g, 0.0), 0.0, self.clip))
        if self.crash_on and self.drift >= CRASH_DRIFT_TH:
            self.crashed = True
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        if t < self.ep_steps and self.atk[t]:
            bs = self.burst_start_of[t]
            return max(0, t - bs) if bs >= 0 else 0
        return 0

    def step(self):
        self.t += 1
        return self.t >= self.ep_steps or self.crashed
