#!/usr/bin/env python3
"""surrogate_env7 (2026-09-08) — FROZEN-v3.1 무대 재현.
env5.2 코퓰러 계승 + ① v3 실데이터 풀(행동 조건부: 호버 바닥·진입 과도·종료 감쇠)
② v3.1 공격 스케줄: 버스트 70%(ON 10~20, δ0.1~0.7) + ramp 30%(상승 15~30 → δ0.6~0.8, hold 20~40)
③ 물리 절벽: ramp 고δ 구간에서 track 유지 누적 ≥ lag(13~54) → crashed (에피 절단, done=True)
   호버는 항상 구조(밴드 실측: δ0.8 호버 생존, 0.9는 생성 안 함). 패턴 흡수 12.5% 재현.
설계 근거 = 아티팩트 「SWIRL 강건성 비교 무대」 §2~5. 보상은 surrogate_run(4쌍)이 담당."""
import os
import numpy as np
_ENTRY_DWELL = int(os.environ.get('SURR_ENTRY_DWELL', '1') or 1)   # ★09-12 밤: hover 진입 스파이크 1스텝(v5b 풀 dwell≤1)

_Q = None
def _quantiles():
    global _Q
    if _Q is None:
        d = np.load(os.environ.get('SURR_POOL', os.path.join(os.path.dirname(__file__), 'train_pool_v3.npz')))   # ★v4: SURR_POOL 로 풀 교체
        Q = {}
        keys = ['track_clean','hover_entry','hover_settled','post_track','post_hover'] + \
               [f'{h}_atk_b{i}' for h in ('track','hover') for i in range(8)]
        for k in keys:
            g, v = d[f'g_{k}'], d[f'v_{k}']
            if len(g) < 50:   # 소표본 폴백 (hover_settled n=7 → entry 와 병합)
                if k == 'hover_settled':
                    g = np.sort(np.r_[g, d['g_hover_entry']]); v = np.sort(np.r_[v, d['v_hover_entry']])
            Q[k] = (g, v)
        # ★2026-09-12 v5 풀 빈 키 대체: hover_entry↔hover_settled, hover_atk_bi → track_atk_bi (hover 중 NIS = track 과 동일 실측 §3),
        #   track_atk_bi 빈 bin → 가장 가까운 비어있지 않은 track bin (δ 연속 가족이라 bin 경계 부동소수점 구멍 방지)
        if len(Q['hover_entry'][0]) < 50 and len(Q['hover_settled'][0]) >= 50: Q['hover_entry'] = Q['hover_settled']
        if len(Q['hover_settled'][0]) < 50 and len(Q['hover_entry'][0]) >= 50: Q['hover_settled'] = Q['hover_entry']
        for i in range(8):
            if len(Q[f'track_atk_b{i}'][0]) < 50:
                cand = [j for j in range(8) if len(Q[f'track_atk_b{j}'][0]) >= 50]
                if cand: j = min(cand, key=lambda j: abs(j - i)); Q[f'track_atk_b{i}'] = Q[f'track_atk_b{j}']
        for i in range(8):
            if len(Q[f'hover_atk_b{i}'][0]) < 50: Q[f'hover_atk_b{i}'] = Q[f'track_atk_b{i}']
        _Q = Q
    return _Q

def _iq(S, u):
    i = int(u * (len(S) - 1))
    return float(S[min(max(i, 0), len(S) - 1)])

class SurrogateEnv:
    CLIP = float(os.environ.get('SURR_CLIP', '3.0'))   # ★v5: 클립 env (확정 4.0)
    RHO_G_ATK = float(os.environ.get('SURR_RHO_G_ATK', '0.96'))
    RHO_G_CLN = float(os.environ.get('SURR_RHO_G_CLN', '0.5'))   # ★v6: 0.85→0.5 — 평시 초과 run 을 Isaac(최대2)에 정합 (형태학 우선)
    RHO_V_ATK = float(os.environ.get('SURR_RHO_V_ATK', '0.61'))
    RHO_V_CLN = float(os.environ.get('SURR_RHO_V_CLN', '0.47'))

    def __init__(self, ep_steps=400, crash=True, seed=0, on_range=(10, 21), off_range=(25, 41)):
        self.ep_steps = int(os.environ.get('SURR_EP_STEPS', ep_steps))
        self.rng = np.random.default_rng(seed)
        self.on_range, self.off_range = on_range, off_range
        self.Q = _quantiles()
        # ★2026-09-08 v7 최종: 2클래스 — 약버스트(탐지) + 급작치명(절벽). Isaac 샘플러와 동일 구조.
        self.atk_prob  = float(os.environ.get('SURR_ATK_PROB', '0.70'))
        self.p_lethal  = float(os.environ.get('SURR_P_LETHAL', '0.30'))
        # ★2026-09-11 v4 (env): 약공격 = step(rise 0) δ U(WEAK_LO,0.7) ON U(ON_LO,ON_HI) / 치명 = δ U(LETH_LO,LETH_HI) hold U(HOLD_LO,HOLD_HI)
        _e = os.environ.get
        self.weak_drng = (float(_e('SURR_WEAK_LO', '0.1')), float(_e('SURR_WEAK_HI', '0.7')))   # ★gapcheck(09-11): δ0.7×3s ws6 에서 2/15 사망 → 상한 0.6 권장
        self.weak_rise = (0, int(_e('SURR_WEAK_RISE_MAX', '30')) + 1)
        self.weak_hold = (int(_e('SURR_WEAK_ON_LO', '10')), int(_e('SURR_WEAK_ON_HI', '40')) + 1)
        self.leth_drng = (float(_e('SURR_LETH_LO','0.74')), float(_e('SURR_LETH_HI','0.80')))
        self.leth_hold = (int(_e('SURR_LETH_HOLD_LO', '40')), int(_e('SURR_LETH_HOLD_HI', '60')) + 1)
        self.v4 = _e('SURR_V4', '') == '1'
        # ★2026-09-12 v5 단일 가족 (Isaac 샘플러 v5 와 동일): δ ~ p·U(split,hi) + (1−p)·U(lo,split), plateau U(ON_LO,ON_HI), 온셋 U(START_LO,START_HI), 1버스트
        #   절벽 = 실측 P(종료|track,δ) 곡선(새 규칙, plateau≥2.5s): 0.70:0 · 0.72:.05 · 0.74:.15 · 0.76:.30 · 0.78:.95 · ≥0.80:1
        #   종료 시각 = 온셋 + lag(U 26~31 스텝, 실측 2.6~3.1 s), hover 가 데드라인 D(U DEAD_LO~DEAD_HI) 안에 들어가면 생존(데드라인 스윕으로 갱신)
        self.v5 = _e('SURR_V5', '') == '1'
        self.v5_lo = float(_e('ATK_DELTA_LO', '0.15')); self.v5_sp = float(_e('ATK_SPLIT', '0.72')); self.v5_hi = float(_e('ATK_DELTA_HI', '0.84')); self.v5_pu = float(_e('ATK_P_UPPER', '0.5'))
        self.v5_on = (int(_e('ATK_ON_LO', '25')), int(_e('ATK_ON_HI', '50')) + 1); self.v5_start = (int(_e('ATK_START_LO', '60')), int(_e('ATK_START_HI', '200')) + 1)
        self.v5_dead = (int(_e('SURR_DEAD_LO', '10')), int(_e('SURR_DEAD_HI', '20')) + 1)
        self.off_rng   = (25, 41)
        self.band_dl   = float(os.environ.get('SURR_BAND_DL','0.74'))  # P(치명) 상승 시작 δ (bandfill 로 갱신)

    def _bin(self, d): return min(7, max(0, int((d - 0.1) / 0.1)))

    def reset(self):
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
        self.sg = rng.normal(); self.sv = rng.normal()
        self.atk = np.zeros(n, bool); self.dl = np.zeros(n); self.bstart = np.full(n, -1, int)
        self.has_atk = rng.random() < self.atk_prob
        self.is_lethal_ep = False
        if self.has_atk and self.v5:
            t = int(rng.integers(*self.v5_start)); emax = n - 5
            dend = rng.uniform(self.v5_sp, self.v5_hi) if rng.random() < self.v5_pu else rng.uniform(self.v5_lo, float(os.environ.get('ATK_WEAK_HI') or self.v5_sp))
            hold = int(rng.integers(*self.v5_on)); e = min(t + hold, emax)
            self.dl[t:e] = dend; self.atk[t:e] = True; self.bstart[t:e] = t
            self.is_lethal_ep = dend >= 0.70
        elif self.has_atk:
            t = int(rng.integers(20, 41)); emax = n - 5
            if rng.random() < self.p_lethal:
                # ── 결과성: 급작 강버스트 1회 (rise 0, δ 치명대, hold 4~6s) ──
                self.is_lethal_ep = True
                dend = rng.uniform(*self.leth_drng); hold = int(rng.integers(*self.leth_hold))
                e = min(t + hold, emax)
                self.dl[t:e] = dend; self.atk[t:e] = True; self.bstart[t:e] = t
            else:
                # ── 탐지 채널: 약버스트 체인 (사다리꼴 rise 연속) ──
                while t < emax:
                    rise = int(rng.integers(*self.weak_rise)); hold = int(rng.integers(*self.weak_hold))
                    dend = rng.uniform(*self.weak_drng)
                    e = min(t + rise + hold, emax)
                    for k in range(t, e):
                        self.dl[k] = dend if rise == 0 else dend * min((k - t + 1) / rise, 1.0)
                    self.atk[t:e] = True; self.bstart[t:e] = t
                    t = e + int(rng.integers(*self.off_rng))
        # ── 절벽 캘리브 (bandfill+flipband 실측 n=177): LOC lag 42% U(4,10) · 8% U(11,25) · 50% U(26,33).
        u = self.rng.random()
        if self.v4:   # ★v4 bandcap 실측(ON30/15/5, FF on): 추락 시각 = 온셋+lag (중앙 2.7~3.3s, 10% 즉사 0.4~0.7s).
            #   추락 여부 = 버스트 중 track 노출량 e 가 임계 e*~U(4,30) 이상 × p_loc(δ)  (실측 δ0.80: ON5 .1 · ON15 .5 · ON30 1.0)
            self._lag = int(self.rng.integers(4, 8)) if u < 0.10 else int(self.rng.integers(26, 35))
            self._estar = float(self.rng.uniform(4.0, 30.0)); self._doom = None   # P(crash|e)=(e−4)/26: ON5 .04 · ON10 .23 · ON15 .42 · ON30 1.0
        elif u < 0.42:   self._lag = int(self.rng.integers(4, 11))
        elif u < 0.50: self._lag = int(self.rng.integers(11, 26))
        else:          self._lag = int(self.rng.integers(26, 34))
        # 치명확률 = 경험 곡선 보간: δ {0.72:0, 0.74:.51, 0.75:.40, 0.76:.72, 0.78:1, 0.80:1}
        import numpy as _np
        dm = float(self.dl.max()) if self.has_atk else 0.0
        if self.v5:   # ★v5 실측(A-1/A-2, 새 규칙): 종료 확률 곡선
            p_loc = float(_np.interp(dm, [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.90], [0.0, 0.05, 0.15, 0.30, 0.95, 1.0, 1.0]))
            self._lag = int(self.rng.integers(26, 32)); self._dead = int(self.rng.integers(*self.v5_dead)); self._doom = None
            if os.environ.get('SURR_PLATEAU_CURVE', '') == '1':   # ★09-12 plateau 의존 종료확률 (B 실측 @δ0.80: 0.5s 0 · 1.5s .85 · 2.0s .97 · ≥2.5s 1; δ0.84 는 1.5s 부터 1)
                _hold = int(self.atk.sum()); _pc = float(_np.interp(_hold, [5, 10, 15, 20, 25], [0.0, 0.4, 0.85, 0.97, 1.0]))
                if dm >= 0.84: _pc = float(_np.interp(_hold, [5, 10, 15], [0.0, 0.6, 1.0]))
                p_loc *= _pc
            if os.environ.get('SURR_DEAD_MODE', '') == 'curve':   # ★09-12 데드라인 실측: δ0.80 d≤0.5s 100%·1–2.5s 65–85% / δ0.84 d0.5s 85%·≥1.0s 0–5%
                dc = float(_np.interp(dm, [0.78, 0.80, 0.82, 0.84, 0.90], [30.0, 20.0, 12.0, 6.0, 4.0]))
                self._dead = max(3, int(round(dc + self.rng.normal(0, 2.0))))
        elif self.v4:   # ★v4 bandcap ON30 track 사망률(무풍/ws6 평균): .74 .2 / .76 .4 / .78 .7 / .80 1 / .82 1
            # ★v4 bandcap ON30 실측 track 사망률 평균(무풍 arm.07 / ws6 arm.05): .74 (.4+.6)/2 / .76 (.6+.8)/2 / .78 (.8+.8)/2 / .80 1 / .82 1
            p_loc = float(_np.interp(dm, [0.72, 0.74, 0.76, 0.78, 0.80, 0.85], [0.0, 0.5, 0.7, 0.8, 1.0, 1.0]))
        else:
            p_loc = float(_np.interp(dm, [0.72, 0.74, 0.75, 0.76, 0.78, 0.85], [0.0, 0.51, 0.45, 0.72, 1.0, 1.0]))
        self._absorb = self.rng.random() > p_loc if self.is_lethal_ep else True
        self._expose = 0                                   # track ∧ 고δ 누적
        # ★09-14 돌풍 창(자연 감독 충돌): SURR_GUST_P 확률로 길이 U(GUST_LO,GUST_HI) 구간의 평시 NIS 를 배율(GUST_G, GUST_V)로 키움. 라벨은 clean 그대로.
        self.gust = np.zeros(n, bool); self.gust_end = -1; self.atk_after_gust = False
        _gp = float(os.environ.get('SURR_GUST_P', '0') or 0)
        if _gp > 0 and rng.random() < _gp:
            gl = int(rng.integers(int(os.environ.get('SURR_GUST_LO', '20')), int(os.environ.get('SURR_GUST_HI', '50')) + 1))
            g0 = int(rng.integers(30, max(31, n - gl - 40))); self.gust[g0:g0 + gl] = True; self.gust_end = g0 + gl
            if self.has_atk and self.v5 and os.environ.get('SURR_GUST_ATK_AFTER', '0') == '1':   # 돌풍 직후 5–20 스텝 뒤 온셋으로 재배치
                hold = int(self.atk.sum()); dend = float(self.dl.max()); t = self.gust_end + int(rng.integers(5, 21)); e = min(t + hold, n - 5)
                self.atk[:] = False; self.dl[:] = 0.0; self.bstart[:] = -1
                if e - t >= 5:
                    self.dl[t:e] = dend; self.atk[t:e] = True; self.bstart[t:e] = t; self.atk_after_gust = True
                else: self.has_atk = False
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
        # ── 절벽 (실측 캘리브): 치명 에피에서 track 유지 누적 ≥ lag → LOC.
        #   호버는 누적을 멈춤(리셋 아님 — 복귀 도박 유지). absorb = P(치명) 곡선의 확률적 생존.
        if self.v5 and self.is_lethal_ep:
            # 온셋부터 데드라인 D 스텝까지 한 번도 hover 하지 않으면(= track 노출 ≥ D) 절벽 확정 → 온셋+lag 에 종료. hover 가 D 안에 들어가면 생존.
            if a and (not hov):
                self._expose += 1
                if self._expose >= self._dead and not self._absorb and self._doom is None:
                    self._doom = int(self.bstart[t]) + self._lag
            if self._doom is not None and t >= self._doom:
                self.crashed = True
        elif self.v4 and self.is_lethal_ep:
            if a and (not hov):
                self._expose += 1
                if self._expose >= self._estar and not self._absorb and self._doom is None:
                    self._doom = int(self.bstart[t]) + self._lag          # 노출 임계 도달 → 온셋+lag 에 추락 확정(호버로 못 되돌림 = flip 진행)
            if self._doom is not None and t >= self._doom:
                self.crashed = True
        elif a and (not hov) and self.is_lethal_ep:
            self._expose += 1
            if self._expose >= self._lag and not self._absorb:
                self.crashed = True
        # ── 풀 선택 (행동 조건부)
        hn = 'hover' if hov else 'track'
        if a and d >= 0.08:
            key = f'{hn}_atk_b{self._bin(d)}'
        elif a:                                            # ramp 초반 비가시(δ<0.08)
            key = f'{hn}_clean' if not hov else 'hover_entry'
            key = 'track_clean' if not hov else ('hover_entry' if self._dwell <= _ENTRY_DWELL else 'hover_settled')
        elif 1 <= self._since_end <= 2:
            key = 'post_hover' if hov else 'post_track'
        elif hov:
            key = 'hover_entry' if self._dwell <= _ENTRY_DWELL else 'hover_settled'
        else:
            key = 'track_clean'
        rg = self.RHO_G_ATK if a else self.RHO_G_CLN
        rv = self.RHO_V_ATK if a else self.RHO_V_CLN
        self.sg = rg * self.sg + np.sqrt(1 - rg * rg) * rng.normal()
        self.sv = rv * self.sv + np.sqrt(1 - rv * rv) * rng.normal()
        Sg, Sv = self.Q[key]
        g = float(np.clip(_iq(Sg, self._phi(self.sg)), 0.0, self.CLIP))
        v = float(np.clip(_iq(Sv, self._phi(self.sv)), 0.0, self.CLIP))
        if (not a) and self.gust[min(self.t, len(self.gust) - 1)]:   # ★돌풍: 평시 관측 배율 (라벨 clean)
            g = float(np.clip(g * float(os.environ.get('SURR_GUST_G', '2.5') or 2.5), 0.0, self.CLIP)); v = float(np.clip(v * float(os.environ.get('SURR_GUST_V', '2.0') or 2.0), 0.0, self.CLIP))
        if a and d < 0.1 and os.environ.get('SURR_WEAK_INTERP', '') == '1':   # ★09-13 스텔스 대역: clean↔b0 분위수 보간 (w=δ/0.1)
            w = max(0.0, min(1.0, d / 0.1)); kc = 'hover_settled' if hov else 'track_clean'; ka = f'{hn}_atk_b0'
            Sgc, Svc = self.Q[kc]; Sga, Sva = self.Q[ka]; ug, uv = self._phi(self.sg), self._phi(self.sv)
            g = float(np.clip((1 - w) * _iq(Sgc, ug) + w * _iq(Sga, ug), 0.0, self.CLIP))
            v = float(np.clip((1 - w) * _iq(Svc, uv) + w * _iq(Sva, uv), 0.0, self.CLIP))
        return v, g, a

    def attack_delay(self):
        t = min(self.t, self.ep_steps - 1)
        return max(0, t - self.bstart[t]) if (self.atk[t] and self.bstart[t] >= 0) else 0

    def step(self):
        self.t += 1
        return self.crashed or self.t >= self.ep_steps
