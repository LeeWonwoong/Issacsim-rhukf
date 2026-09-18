#!/usr/bin/env python3
"""surrogate_env8 (2026-09-14) — env7 + 풀 v5c 키 사용: ① 바람 티어를 에피소드마다 추첨(SURR_TIER_P="0:0.3,6:0.3,7:0.2,10:0.2")
   → 키에 _ws{n} 접미(없으면 기본 키 폴백). 돌풍 배율(SURR_GUST_*)은 그대로 두되 티어 실측이 있으면 P=0 으로 끄는 것을 권장.
   ② 스텔스 실측 빈 {track,hover}_atk_s0..2 (δ 0.03/0.05/0.08, cert_S): δ<0.1 공격은 가장 가까운 s-빈에서 추첨 (v7 의 보간·clean 대체).
   ③ 풀 키는 npz 에 있는 것을 전부 적재, 소표본(<50)은 같은 티어 내 인접 빈 → 기본 티어 순으로 폴백. 아래는 env7 원문.
surrogate_env7 (2026-09-08) — FROZEN-v3.1 무대 재현.
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
        import re as _re
        d = np.load(os.environ.get('SURR_POOL', os.path.join(os.path.dirname(__file__), 'train_pool_v3.npz')))   # ★v4: SURR_POOL 로 풀 교체
        Q = {}
        base = ['track_clean','hover_entry','hover_settled','post_track','post_hover'] + \
               [f'{h}_atk_b{i}' for h in ('track','hover') for i in range(8)] + [f'{h}_atk_s{i}' for h in ('track','hover') for i in range(3)]
        for f in d.files:
            if f.startswith('g_'): Q[f[2:]] = (d[f], d['v_' + f[2:]])
        tiers = sorted({int(m.group(1)) for k in Q for m in [_re.search(r'_ws(\d+)$', k)] if m})
        E = (np.array([]), np.array([]))
        n = lambda k: len(Q.get(k, E)[0])
        def fix(sfx):
            for k in base: Q.setdefault(k + sfx, E)
            # hover_entry↔hover_settled, hover_atk_bi → track_atk_bi (hover 중 NIS = track 과 동일 실측), 빈 bin → 인접 bin
            if n('hover_entry' + sfx) < 50 and n('hover_settled' + sfx) >= 50: Q['hover_entry' + sfx] = Q['hover_settled' + sfx]
            if n('hover_settled' + sfx) < 50 and n('hover_entry' + sfx) >= 50: Q['hover_settled' + sfx] = Q['hover_entry' + sfx]
            for pre, m in (('atk_b', 8), ('atk_s', 3)):
                for i in range(m):
                    if n(f'track_{pre}{i}{sfx}') < 50:
                        cand = [j for j in range(m) if n(f'track_{pre}{j}{sfx}') >= 50]
                        if cand: j = min(cand, key=lambda j: abs(j - i)); Q[f'track_{pre}{i}{sfx}'] = Q[f'track_{pre}{j}{sfx}']
                for i in range(m):
                    if n(f'hover_{pre}{i}{sfx}') < 50: Q[f'hover_{pre}{i}{sfx}'] = Q[f'track_{pre}{i}{sfx}']
            for h in ('track', 'hover'):   # s-빈이 통째로 없으면(구 풀) b0 로 (=env7 의 d≥0.08→b0 거동)
                for i in range(3):
                    if n(f'{h}_atk_s{i}{sfx}') < 50: Q[f'{h}_atk_s{i}{sfx}'] = Q[f'{h}_atk_b0{sfx}']
            if sfx:   # 티어 키가 여전히 비면 기본 티어(ws0) 로
                for k in base:
                    if n(k + sfx) < 50: Q[k + sfx] = Q[k]
        fix('')
        for w in tiers: fix(f'_ws{w}')
        Q['_tiers'] = tiers; Q['_has_s'] = len(d['g_track_atk_s1']) >= 50 if 'g_track_atk_s1' in d.files else False
        _Q = Q
    return _Q

def _parse_tier(s):
    it = [(int(a), float(b)) for a, b in (x.split(':') for x in s.split(','))]; z = sum(p for _, p in it)
    return [(w, p / z) for w, p in it]

def _sched(envname, ep):
    """에피소드 스케줄 "0-74:SPEC;75-124:SPEC;125-:SPEC" → 현재 ep 의 SPEC 문자열(없으면 None). ★v8 급변 비정상성 축(09-14 사용자: FIR 은 강외란·급변에서 유리)."""
    s = os.environ.get(envname, '').strip()
    if not s: return None
    for seg in s.split(';'):
        rng, spec = seg.split(':', 1); a, _, b = rng.partition('-')
        if int(a or 0) <= ep and (b == '' or ep <= int(b)): return spec
    return None

def _tier_probs(ep=0):
    """SURR_TIER_SCHED(에피소드 스케줄) 우선, 없으면 SURR_TIER_P="0:0.3,6:0.3,7:0.2,10:0.2". 비어 있으면 [(0,1)]."""
    s = _sched('SURR_TIER_SCHED', ep) or os.environ.get('SURR_TIER_P', '').strip()
    return _parse_tier(s) if s else [(0, 1.0)]

def _sched_tier_curve(ws, envname='SURR_PLOC_TIERS'):
    """SURR_PLOC_TIERS="7:δ,p,δ,p,…;10:…" → (xs, ys) for this ws, else None (ws0 또는 미지정 → v5 기본 곡선). 같은 형식으로 SURR_DEAD_TIERS(δ,스텝)."""
    s = os.environ.get(envname, '').strip()
    if not s or ws <= 0: return None
    for seg in s.split(';'):
        w, spec = seg.split(':', 1)
        if int(w) == int(ws):
            f = [float(x) for x in spec.split(',')]; xs = [0.0] + f[0::2] + [0.95]; ys = [0.0] + f[1::2] + [1.0]
            return xs, ys
    return None

_MAIN = None      # 첫 생성 인스턴스 = 학습 env. 프로브 env(별도 시드로 매번 생성)는 학습 env 의 에피소드 인덱스를 따라 같은 체제를 본다.
_TRAIN_EP = 0

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
        self.rng = np.random.default_rng(seed); self._seed0 = int(seed)
        global _MAIN
        self.is_main = _MAIN is None
        if self.is_main: _MAIN = self
        self.ep_idx = -1
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
    def _sbin(self, d): return int(np.argmin([abs(d - x) for x in (0.03, 0.05, 0.08)]))   # 스텔스 실측 빈(cert_S δ 0.03/0.05/0.08) 최근접
    def _key(self, k):   # 티어 접미 키가 있으면 그것, 없으면 기본 키
        kk = k + self.sfx
        return kk if kk in self.Q else k

    def reset(self):
        # ★09-15 저녁(PRECHAIN §0.3 A1): SURR_EP_RNG=1 이면 학습 env 의 시나리오 난수를 (시드, 에피소드 번호)로 매 에피소드 새로 만든다.
        #   구: 한 스트림을 이어 써서 한쪽 학습기만 추락하면 이후 시나리오가 갈라졌다(짝 붕괴). 기본 off = 구 동작 재현.
        if self.is_main and os.environ.get('SURR_EP_RNG', '0') == '1':
            self.rng = np.random.default_rng([self._seed0 & 0xFFFFFFFF, self.ep_idx + 1])
        n = self.ep_steps; rng = self.rng
        self.t = 0; self.crashed = False
        # ★v8 바람 티어 추첨(에피소드 단위 정상 혼합, HANDOFF §9). 풀에 없는 티어는 가장 가까운 실측 티어로.
        #   스케줄(SURR_TIER_SCHED / SURR_FAM_SCHED)이 있으면 학습 에피소드 인덱스로 체제를 고른다(프로브 env 는 학습 env 의 인덱스를 따름).
        global _TRAIN_EP
        self.ep_idx += 1
        if self.is_main: _TRAIN_EP = self.ep_idx
        ep = _TRAIN_EP
        tp = _tier_probs(ep); ws = int(tp[int(rng.choice(len(tp), p=[p for _, p in tp]))][0])
        fam = _sched('SURR_FAM_SCHED', ep)           # "lo,split,hi[,p_upper]" 가족 전환(급변 비정상성 축)
        if fam:
            f = [float(x) for x in fam.split(',')]; self.v5_lo, self.v5_sp, self.v5_hi = f[0], f[1], f[2]
            if len(f) > 3: self.v5_pu = f[3]
        av = self.Q.get('_tiers', [])
        if ws > 0 and av: ws = min(av, key=lambda w: abs(w - ws))
        self.ws = ws; self.sfx = f'_ws{ws}' if ws > 0 else ''
        self.sg = rng.normal(); self.sv = rng.normal()
        self.atk = np.zeros(n, bool); self.dl = np.zeros(n); self.bstart = np.full(n, -1, int)
        self.has_atk = rng.random() < self.atk_prob
        self.is_lethal_ep = False
        if self.has_atk and self.v5:
            emax = n - 5
            _K = max(1, int(os.environ.get('ATK_BURSTS', '1') or 1))   # ★09-16 다중 버스트(기본 1 = 기존 동작). K>1 이면 간격 off_rng 로 연속 버스트.
            if _K > 1:   # 버스트 K 개가 판 안에 들어가도록 첫 온셋 상한을 당긴다
                _need = _K * (self.v5_on[1] - 1) + (_K - 1) * (self.off_rng[1] - 1)
                _hi = max(self.v5_start[0] + 1, min(self.v5_start[1], emax - _need))
                t = int(rng.integers(self.v5_start[0], _hi))
            else:
                t = int(rng.integers(*self.v5_start))
            _dmax = 0.0
            for _b in range(_K):
                if t >= emax: break
                dend = rng.uniform(self.v5_sp, self.v5_hi) if rng.random() < self.v5_pu else rng.uniform(self.v5_lo, float(os.environ.get('ATK_WEAK_HI') or self.v5_sp))
                hold = int(rng.integers(*self.v5_on)); e = min(t + hold, emax)
                self.dl[t:e] = dend; self.atk[t:e] = True; self.bstart[t:e] = t
                _dmax = max(_dmax, dend)
                t = e + int(rng.integers(*self.off_rng))
            self.is_lethal_ep = _dmax >= 0.70
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
            # ★v8 티어별 종료 곡선 (cert_WIND 09-14 실측, plateau 3 s, track done=추락∪지오펜스): SURR_PLOC_TIERS="7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1"
            _pt = _sched_tier_curve(self.ws)
            if _pt is not None: p_loc = float(_np.interp(dm, _pt[0], _pt[1]))
            self._lag = int(self.rng.integers(26, 32)); self._dead = int(self.rng.integers(*self.v5_dead)); self._doom = None
            if os.environ.get('SURR_PLATEAU_CURVE', '') == '1':   # ★09-12 plateau 의존 종료확률 (B 실측 @δ0.80: 0.5s 0 · 1.5s .85 · 2.0s .97 · ≥2.5s 1; δ0.84 는 1.5s 부터 1)
                _hold = int(self.atk.sum()); _pc = float(_np.interp(_hold, [5, 10, 15, 20, 25], [0.0, 0.4, 0.85, 0.97, 1.0]))
                if dm >= 0.84: _pc = float(_np.interp(_hold, [5, 10, 15], [0.0, 0.6, 1.0]))
                p_loc *= _pc
            if os.environ.get('SURR_DEAD_MODE', '') == 'curve':   # ★09-12 데드라인 실측: δ0.80 d≤0.5s 100%·1–2.5s 65–85% / δ0.84 d0.5s 85%·≥1.0s 0–5%
                dc = float(_np.interp(dm, [0.78, 0.80, 0.82, 0.84, 0.90], [30.0, 20.0, 12.0, 6.0, 4.0]))
                _dt = _sched_tier_curve(self.ws, 'SURR_DEAD_TIERS')   # ★v8 티어별 데드라인 (cert_DEAD 09-14: ws10 δ0.84 d5 생존 60% → ~4 스텝)
                if _dt is not None: dc = float(_np.interp(dm, _dt[0][1:-1], _dt[1][1:-1]))
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
        if a and d >= 0.1:
            key = f'{hn}_atk_b{self._bin(d)}'
        elif a and d >= 0.02 and self.Q.get('_has_s', False):   # ★v8 스텔스 실측 빈 (δ 0.03/0.05/0.08 최근접)
            key = f'{hn}_atk_s{self._sbin(d)}'
        elif a and d >= 0.08:                              # 구 풀(s-빈 없음): env7 거동
            key = f'{hn}_atk_b0'
        elif a:                                            # ramp 초반 비가시(δ<0.08)
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
        key = self._key(key)                                # ★v8 티어 접미
        Sg, Sv = self.Q[key]
        g = float(np.clip(_iq(Sg, self._phi(self.sg)), 0.0, self.CLIP))
        v = float(np.clip(_iq(Sv, self._phi(self.sv)), 0.0, self.CLIP))
        if (not a) and self.gust[min(self.t, len(self.gust) - 1)]:   # ★돌풍: 평시 관측 배율 (라벨 clean)
            g = float(np.clip(g * float(os.environ.get('SURR_GUST_G', '2.5') or 2.5), 0.0, self.CLIP)); v = float(np.clip(v * float(os.environ.get('SURR_GUST_V', '2.0') or 2.0), 0.0, self.CLIP))
        if a and d < 0.1 and (not self.Q.get('_has_s', False)) and os.environ.get('SURR_WEAK_INTERP', '') == '1':   # ★09-13 스텔스 보간 (구 풀 전용)
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
