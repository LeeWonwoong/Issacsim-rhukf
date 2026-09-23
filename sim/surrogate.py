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
    # ★09-23 정합수정 F3 — 에피소드 임의효과. Isaac 평시 gyro 분산의 42%(z 단위 55%)가 에피소드 단위
    #   잠재변수(COM 편향 추첨·바람 방향·초기 트림)인데 k=32 이웃이 32개 서로 다른 에피에서 와 평균으로 날아갔다.
    #   증거: 에피 오프셋 십분위별 이웃범위 이탈률 0.017→0.492 완전 단조, 오프셋을 알면 0.245→0.042.
    #   z = m + a 로 두고 m 은 에피 상수, a 는 AR(1). 주변분포 N(0,1) 을 보존하도록 a 의 정상 sd = √(1−sd_m²).
    knn_ep_off_g: float = 0.0          # 실측 0.745 (0 = 끔)
    knn_ep_off_v: float = 0.0          # 실측 0.345
    knn_ep_off_corr: float = 0.461     # corr(m_g, m_v) 실측
    # ★09-23 정합수정 F6 — 채널별 ρ. 구 코드는 평시 ρ 를 풀 파일에서 읽어 설정 불가였고(52–54행이 죽은 값),
    #   공격 중은 두 채널에 같은 knn_rho_atk 를 썼다. 실측 목표 ACF1: 공격 gyro 0.952 / vel 0.703.
    knn_rho_g_atk: float = 0.0         # 0 = knn_rho_atk 사용
    knn_rho_v_atk: float = 0.0
    knn_rho_g_cln: float = 0.0         # 0 = 풀 파일 값 사용
    knn_rho_v_cln: float = 0.0
    # ★09-23 정합수정 F7 — 사건 수준 랜덤효과. 강제 재생에서 공격 구간 실현 sd 가 Isaac 대비
    #   전이 0.78 · 강 0.57 배로 너무 좁았다(이웃이 희소해 코퓰러가 좁은 구간만 훑는다).
    #   실측 사건 ICC ≈ 0.45, 사건 총 z-SD 1.3–1.4. 사건 온셋에서 ζ 를 뽑아 사건 내내 유지한다.
    knn_atk_sd: float = 1.0            # 공격 중 코퓰러 z 의 총 sd (1.0 = 끔). 채널별 값이 0 이면 이 값을 쓴다
    knn_atk_sd_g: float = 0.0          # ★09-23 P0-c: 채널 분리(실측 사건 피크 sd gyro 0.654 / vel 1.078 — 공유는 틀렸다)
    knn_atk_sd_v: float = 0.0
    knn_ev_icc: float = 0.45           # 그중 사건 상수 성분의 분산 비중
    # ★09-23 P0-c 평시 코퓰러: 단일 AR(1) 로는 lag1 과 lag10 을 동시에 못 맞춘다.
    #   실측 적합(5 lag 오차 ≤0.01): gyro = 너겟 0.15 + AR(0.73)·0.44 + AR(0.961)·0.41
    #                                vel  = 너겟 0.27 + AR(0.930)·0.73
    #   모든 가중이 0 이면 구 단일 AR(1) 경로(비트 동일)를 쓴다.
    knn_c2_g: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)   # (nugget, w1, rho1, w2, rho2)
    knn_c2_v: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # ★09-23 θ 채널(P2′ 자세 퍼텐셜 성형용, 기본 끔). 참조 = P2S p6_joint.py 인자 32a (Isaac 홀드아웃 관문 통과판).
    #   1단: 조건 특징(_feat) + 직전 6 행동 비트×theta_abits_w 로 별도 KD-트리에서 theta_k1 행
    #   2단: 그 행들의 풀 NIS 창(압축 vel·gyro 4스텝)이 **생성된** NIS 창과 가까운 theta_m 행 → θ 경험분위
    #   코퓰러 z = 에피 효과 + AR(ρ1) + AR(ρ2) + 너겟 (s_m, w1, ρ1, w2, ρ2 — 분산 비중, 너겟 = 1−합), 전용 난수 스트림
    #   → θ 를 켜도 NIS 는 비트 동일. 풀에 'theta' 열이 있어야 한다(scripts/build_pool_knn.py → knn_v4).
    theta: bool = False
    theta_k1: int = 1024
    theta_m: int = 32
    theta_abits_w: float = 4.0         # 행동 비트 1개 불일치 = 거리 4 (δ 0.2 차와 같음). 0 = 비트 조건 끔
    theta_copula: Tuple[float, float, float, float, float] = (0.054, 0.159, 0.201, 0.787, 0.868)   # P2S J32a 적합값
    crash: CrashConfig = field(default_factory=CrashConfig)

    def __post_init__(self):
        if isinstance(self.crash, dict): self.crash = CrashConfig(**self.crash)
        self.crash.ploc_tiers = {int(k): list(v) for k, v in self.crash.ploc_tiers.items()}
        self.crash.dead_tiers = {int(k): list(v) for k, v in self.crash.dead_tiers.items()}


_BASE = ['track_clean', 'hover_entry', 'hover_settled', 'post_track', 'post_hover'] + \
        [f'{h}_atk_b{i}' for h in ('track', 'hover') for i in range(8)] + [f'{h}_atk_s{i}' for h in ('track', 'hover') for i in range(3)]
_POOL_CACHE: Dict[str, dict] = {}

# ★09-23 정합수정 F4/F5 — 조건 특징의 포화 지점.
#   SINCE_CAP 15→60: 공격 종료 16 스텝 뒤부터 '한 번도 공격 없던 평시'와 같은 질의점이 되던 결함.
#     Isaac 실측 사후 잔류 gyro 는 since 61–98 에서도 평시의 2.16 배이고 에피 끝까지 안 돌아온다.
#     홀드아웃 행의 15.3%(16,769)가 진짜 평시와 뒤섞여 양쪽을 동시에 망치고 있었다.
#   DWELL_CAP 5→12: 행동→관측 되먹임이 5 스텝에서 절단돼 정책이 상태분포를 못 움직였다
#     (hover 비율 0→0.2 일 때 평시 관측 이동 Isaac +0.228 vs surrogate +0.045).
#   거리 스케일은 나누기로 유지(각각 최대 15·6 기여) — 이웃 희석 없이 해상도만 올린다.
SINCE_CAP = 60.0
DWELL_CAP = 12.0


def load_pool(cfg: SurrogateConfig) -> dict:
    key = (cfg.pool, cfg.pool_format, cfg.pool_compress, cfg.pool_clip)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    d = np.load(cfg.pool)
    if 'format' in d.files and str(d['format']) in ('knn_v2', 'knn_v3', 'knn_v4'):   # Isaac 실측 k-NN 풀 (v3 = +패턴 열, v4 = +θ·roll·pitch 열)
        X = np.asarray(d['X'], float); rho = json.loads(str(d['rho']))
        Q = dict(_knn=True, X=X, rho=rho, _tiers=[], _has_s=False, _format=str(d['format']),
                 _has_pat=(str(d['format']) in ('knn_v3', 'knn_v4') and X.shape[1] >= 12))
        names = [str(n) for n in d['names']] if 'names' in d.files else []
        if 'theta' in names:                                   # ★09-23 θ 열(knn_v4, 또는 P2S 스크래치 knn_v3+theta)
            Q['_theta_col'] = X[:, names.index('theta')].copy()
            Q['_ep'] = np.asarray(d['ep']) if 'ep' in d.files else None
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


_IQ_GRID: Dict[int, np.ndarray] = {}


def _iq(S, u):
    """★09-23 정합수정 F1: 구 `i = int(u*(k-1))` 는 최상위 순서통계량 S[k-1] 을 영원히 못 뽑아
    (u=1 필요) 재생분포가 '하위 k-1 개 위의 이산 균등'이 됐다. 실측 왜곡: 평균 −30% · 분산 −73% ·
    >p99 꼬리 6.7배 과소. 표준 plug-in 경험분위(중점 보간)로 교체 — 무편향, 분산 (k−1)/k."""
    k = len(S)
    if k <= 1:
        return float(S[0]) if k else 0.0
    g = _IQ_GRID.get(k)
    if g is None:
        g = (np.arange(k) + 0.5) / k
        _IQ_GRID[k] = g
    return float(np.interp(u, g, S))


def _comp_theta(x):
    """θ 2단 조건용 NIS 압축 (P2S p6 와 동일): min(log1p√x, 4)/4."""
    return np.minimum(np.log1p(np.sqrt(np.maximum(x, 0.0))), 4.0) / 4.0


def _pool_windows(v, g, ep):
    """풀 행별 NIS 창 [c(v)_{t−3..t}, c(g)_{t−3..t}] — 에피 시작부는 그 에피 첫 값으로 채운다."""
    cv, cg = _comp_theta(v), _comp_theta(g); n = len(v); W = np.zeros((n, 8))
    start = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
    first = np.repeat(start, np.diff(np.r_[start, n]))
    for j in range(4):
        idx = np.maximum(np.arange(n) - (3 - j), first)
        W[:, j] = cv[idx]; W[:, 4 + j] = cg[idx]
    return W


def _pool_abits(act, ep, n_bits=6):
    """풀 행별 직전 n 행의 행동(같은 에피 안, 밖이면 0). 열 0 = 1 행 전."""
    B = np.zeros((len(act), n_bits))
    for L in range(1, n_bits + 1):
        sh = np.r_[np.zeros(L), act[:-L]]; ok = np.r_[np.zeros(L, bool), ep[L:] == ep[:-L]]
        B[:, L - 1] = np.where(ok, sh, 0.0)
    return B


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
            X = self.Q['X']                                # [d0,d1,d3,d6,since_end,act,dwell,ws,dlast,nis_v,nis_g(,pat)]
            self._has_pat = bool(self.Q.get('_has_pat'))
            self._F = self._feat(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 11] if self._has_pat else None)
            self.pat = 0
            self._V, self._G = X[:, 9], X[:, 10]
            self._nn_cache = {}
            self._kdt = None                                   # ★09-23 풀 v2(758k)에서 전수탐색이 3.5 ms/스텝 병목 → KD-트리(같은 유클리드 거리 = 같은 이웃)
            if len(self._F) > 200000:
                try:
                    from scipy.spatial import cKDTree
                    self._kdt = cKDTree(np.ascontiguousarray(self._F), balanced_tree=False, compact_nodes=False)
                except Exception:
                    self._kdt = None
        self.theta = None
        if cfg.theta:
            self._theta_setup()

    def _theta_setup(self):
        """θ 채널 자료(풀마다 1회, 풀 캐시에 보관): θ 전용 KD-트리 · 풀 NIS 창 · 정규화 척도."""
        if not self.knn or '_theta_col' not in self.Q:
            raise ValueError('env.surrogate.theta=true 는 θ 열이 있는 k-NN 풀(knn_v4)이 필요하다 — scripts/build_pool_knn.py 로 다시 만들 것')
        if self.Q.get('_ep') is None:
            raise ValueError('θ 채널: 풀에 ep 배열이 없다(창·행동 비트를 만들 수 없음)')
        T = self.Q.get('_theta')
        key = float(self.cfg.theta_abits_w)
        if T is None or T['abw'] != key:
            from scipy.spatial import cKDTree
            X, ep = self.Q['X'], self.Q['_ep']
            F = self._F
            if key > 0:
                F = np.c_[F, _pool_abits(X[:, 5], ep) * key]
            WP = _pool_windows(X[:, 9], X[:, 10], ep)
            T = dict(abw=key, tree=cKDTree(np.ascontiguousarray(F), balanced_tree=False, compact_nodes=False),
                     WP=WP, WS=WP.std(0) + 1e-9, TH=self.Q['_theta_col'], c1={})
            self.Q['_theta'] = T
        self._TH = T

    def _theta_reset(self):
        """θ 전용 난수 스트림(주 rng 는 건드리지 않는다 → NIS 비트 동일)과 에피 상태."""
        self._rt = np.random.default_rng([self.seed0 & 0xFFFFFFFF, self.ep_idx + 1, 0x7E7A])
        sm2 = float(self.cfg.theta_copula[0])
        self._tm = math.sqrt(max(sm2, 0.0)) * self._rt.normal(); self._t1 = self._rt.normal(); self._t2 = self._rt.normal()
        self._twv = []; self._twg = []; self._tah = [0] * 6
        self.theta = None

    def _theta_step(self, q, prev_action, v, g):
        """이번 스텝 θ (rad). q = NIS 조건 특징(_feat 한 행). v, g = 방금 생성한 원시 NIS."""
        c, T = self.cfg, self._TH
        if c.theta_abits_w > 0:
            q = np.r_[q, np.array(self._tah[::-1], float) * float(c.theta_abits_w)]
        self._tah = self._tah[1:] + [int(prev_action)]
        key = tuple(np.round(q, 1))
        nb = T['c1'].get(key)
        if nb is None:
            nb = T['tree'].query(q, k=min(int(c.theta_k1), len(T['TH'])), workers=1)[1]
            if len(T['c1']) < 20000: T['c1'][key] = nb
        self._twv.append(float(_comp_theta(v))); self._twg.append(float(_comp_theta(g)))
        self._twv = self._twv[-4:]; self._twg = self._twg[-4:]
        pad = lambda L: [L[0]] * (4 - len(L)) + L
        w = np.array(pad(self._twv) + pad(self._twg))
        d = (((T['WP'][nb] - w) / T['WS']) ** 2).sum(1)
        M = int(c.theta_m)
        sel = nb[np.argpartition(d, M)[:M]] if M < len(nb) else nb
        S = np.sort(T['TH'][sel])
        sm2, w1, r1, w2, r2 = (float(x) for x in c.theta_copula)
        nug2 = max(0.0, 1.0 - sm2 - w1 - w2)
        rt = self._rt
        self._t1 = r1 * self._t1 + math.sqrt(1 - r1 * r1) * rt.normal()
        self._t2 = r2 * self._t2 + math.sqrt(1 - r2 * r2) * rt.normal()
        z = self._tm + math.sqrt(nug2) * rt.normal() + math.sqrt(w1) * self._t1 + math.sqrt(w2) * self._t2
        self.theta = _iq(S, _phi(z))
        return self.theta

    @staticmethod
    def _feat(d0, d1, d3, d6, since, act, dwell, ws, dlast, pat=None):
        """k-NN 거리 척도: δ 0.05 = 1, 공격 종료 후 1 스텝 = 1(15 에서 포화), 행동 불일치 = 100(사실상 정확 일치),
        유지 1 스텝 = 1(5 에서 포화), 풍속 2 m/s = 1, 직전 사건 최대 세기 0.05 = 1(공격 종료 후 15 스텝 동안만 의미)."""
        since = np.asarray(since, float); dl = np.where(since <= SINCE_CAP, np.asarray(dlast, float), 0.0)
        F = np.c_[np.asarray(d0) / 0.05, np.asarray(d1) / 0.05, np.asarray(d3) / 0.05, np.asarray(d6) / 0.05,
                  np.minimum(since, SINCE_CAP) / (SINCE_CAP / 15.0), np.asarray(act) * 100.0,
                  np.minimum(dwell, DWELL_CAP) / (DWELL_CAP / 6.0), np.asarray(ws) / 2.0, dl / 0.05]
        if pat is not None:                                # ★09-22 v3 풀: 기동 패턴 불일치 = 1000(사실상 같은 패턴 안에서만 이웃)
            F = np.c_[F, np.asarray(pat, float) * 1000.0]
        return F

    # ── 에피소드 ─────────────────────────────────────────────────────────────
    def reset(self, plan: Optional[AttackPlan] = None, ws: Optional[float] = None):
        """plan/ws 를 주면 그 시나리오를 강제(프로브·캡처 재현용). 난수 순서: 바람 → 코퓰러 초기값 → 공격 → 절벽."""
        if self.reseed:
            self.rng = np.random.default_rng([self.seed0 & 0xFFFFFFFF, self.ep_idx + 1])
        self.ep_idx += 1
        n, rng, cc, c_ = self.ep_steps, self.rng, self.cfg.crash, self.cfg
        self.t = 0; self.crashed = False
        w = sample_wind(rng, self.wcfg, self.ep_idx) if ws is None else ws   # ep_idx: 0 기준(축 B 스케줄)
        av = self.Q['_tiers']
        wi = int(w)
        if wi > 0 and av: wi = min(av, key=lambda x: abs(x - wi))
        self.ws = wi; self.sfx = f'_ws{wi}' if wi > 0 else ''
        if self.knn:
            self.ws = float(w); self._act_prev = None; self._act_dwell = 0; self._last_atk = None; self._dlast = 0.0
        # ★09-23 F3: 에피소드 임의효과 m 을 먼저 뽑고, AR(1) 잔차 a 는 √(1−sd_m²) 로 줄여 주변분포를 보존한다.
        og, ov, rc = float(c_.knn_ep_off_g), float(c_.knn_ep_off_v), float(c_.knn_ep_off_corr)
        if og > 0 or ov > 0:
            z1, z2 = rng.normal(), rng.normal()
            self.mg = og * z1
            self.mv = ov * (rc * z1 + math.sqrt(max(0.0, 1.0 - rc * rc)) * z2)
            # ★09-23 P0-b: 가산형. AR 잔차를 깎지 않는다(구 축소형은 에피내 분산을 25% 잃었다).
            self._sag = self._sav = 1.0
        else:
            self.mg = self.mv = 0.0; self._sag = self._sav = 1.0
        self.sg = self._sag * rng.normal(); self.sv = self._sav * rng.normal()
        self.sg2 = rng.normal(); self.sv2 = rng.normal()     # ★09-23 P0-c 2성분 코퓰러의 느린 성분
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
        self._undecl = 0                                   # ★09-21 선언 마감 카운터(연속 미선언 스텝)
        self._t_shift = None; self._ws2 = None                # ★09-20 에피소드 중간 풍속 전환 (난수는 마지막에: 앞 스트림 불변)
        sh = getattr(self.wcfg, 'shift', None)
        if self.knn and sh and rng.random() < float(sh.get('p', 1.0)):
            self._t_shift = int(rng.integers(int(sh['t'][0]), int(sh['t'][1]) + 1)); self._ws2 = float(rng.uniform(*sh['range']))
        if self.knn and getattr(self, '_has_pat', False):   # ★09-22 v3 풀: 에피마다 기동 패턴 추첨(Isaac 과 같이 5패턴 균등) — 맨 끝이라 앞 난수 불변
            self.pat = int(rng.integers(0, 5))
        if self.cfg.theta:                                   # ★09-23 θ 채널: 전용 스트림(주 rng 소비 없음)
            self._theta_reset()
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
        q = self._feat(dl(0), dl(1), dl(3), dl(6), since, prev_action, self._act_dwell, ws_now, self._dlast, self.pat if getattr(self, '_has_pat', False) else None)[0]
        key = (c.knn_k,) + tuple(np.round(q, 1))   # ★09-23 P0-a 버그수정: k 가 키에 없어 k 변경이 조용히 무시됐다
        nn = self._nn_cache.get(key)
        if nn is None:
            if getattr(self, '_kdt', None) is not None:
                idx = self._kdt.query(q, k=c.knn_k, workers=1)[1]
            else:
                dist = ((self._F - q) ** 2).sum(1)
                idx = np.argpartition(dist, c.knn_k)[:c.knn_k]
            nn = (np.sort(self._V[idx]), np.sort(self._G[idx]))
            if len(self._nn_cache) < 200000: self._nn_cache[key] = nn
        rho = self.Q['rho']
        if a:                                                   # ★09-23 F6: 공격 중도 채널별
            rg = c.knn_rho_g_atk or c.knn_rho_atk
            rv = c.knn_rho_v_atk or c.knn_rho_atk
        else:
            rg = c.knn_rho_g_cln or rho.get('rho_g_cln', 0.5)
            rv = c.knn_rho_v_cln or max(rho.get('rho_v_cln', 0.0), 0.0)
        # ★09-23 F7: 사건 온셋에서 사건 상수 ζ 추첨, 사건 내내 유지(종료 시 해제)
        T = float(c.knn_atk_sd)
        if a:
            if not getattr(self, '_in_ev', False):
                self._in_ev = True
                ic = math.sqrt(max(0.0, c.knn_ev_icc))
                self._evg = (c.knn_atk_sd_g or T) * ic * rng.normal()
                self._evv = (c.knn_atk_sd_v or T) * ic * rng.normal()
            vm = getattr(self, 'mg', 0.0) ** 2
            sar = math.sqrt(max(0.0, (1.0 - c.knn_ev_icc) * T * T - vm))
        else:
            self._in_ev = False; self._evg = self._evv = 0.0
            sar = None
        c2g, c2v = c.knn_c2_g, c.knn_c2_v
        if (not a) and (c2g[1] or c2g[3]):            # ★09-23 P0-c 평시 2성분 (공격 중은 기존 경로)
            self.sg = c2g[2] * self.sg + math.sqrt(1 - c2g[2] ** 2) * rng.normal()
            self.sg2 = c2g[4] * self.sg2 + math.sqrt(1 - c2g[4] ** 2) * rng.normal()
            zg = c2g[0] * rng.normal() + c2g[1] * self.sg + c2g[3] * self.sg2
        else:
            sag = sar if sar is not None else getattr(self, '_sag', 1.0)
            if a and (c.knn_atk_sd_g or c.knn_atk_sd):
                T_g = c.knn_atk_sd_g or c.knn_atk_sd
                sag = math.sqrt(max(0.0, (1.0 - c.knn_ev_icc) * T_g * T_g - getattr(self, 'mg', 0.0) ** 2))
            self.sg = rg * self.sg + sag * math.sqrt(1 - rg * rg) * rng.normal()
            zg = self.sg
        if (not a) and (c2v[1] or c2v[3]):
            self.sv = c2v[2] * self.sv + math.sqrt(1 - c2v[2] ** 2) * rng.normal()
            self.sv2 = c2v[4] * self.sv2 + math.sqrt(1 - c2v[4] ** 2) * rng.normal()
            zv = c2v[0] * rng.normal() + c2v[1] * self.sv + c2v[3] * self.sv2
        else:
            sav = sar if sar is not None else getattr(self, '_sav', 1.0)
            if a and (c.knn_atk_sd_v or c.knn_atk_sd):
                T_v = c.knn_atk_sd_v or c.knn_atk_sd
                sav = math.sqrt(max(0.0, (1.0 - c.knn_ev_icc) * T_v * T_v - getattr(self, 'mv', 0.0) ** 2))
            self.sv = rv * self.sv + sav * math.sqrt(1 - rv * rv) * rng.normal()
            zv = self.sv
        self._deadline(a, prev_action == 1)
        # ★09-23 F3/F7: 코퓰러 위치 = 에피 오프셋 + 사건 효과 + AR(1) 잔차
        out = (_iq(nn[0], _phi(zv + getattr(self, 'mv', 0.0) + getattr(self, '_evv', 0.0))),
               _iq(nn[1], _phi(zg + getattr(self, 'mg', 0.0) + getattr(self, '_evg', 0.0))), a)
        if c.theta:                                             # ★09-23 θ 채널(전용 난수, NIS 뒤) → self.theta
            self._theta_step(q, prev_action, out[0], out[1])
        return out

    def _deadline(self, a: bool, hov: bool):
        """★09-21 선언 마감: 공격 활성 스텝에서 track 이면 연속 미선언 +1, hover 면 0; 비활성이면 0. L 에 이르면 종료(crashed 경로 재사용, 기본 L=0 끔)."""
        L = int(getattr(self.acfg, 'deadline_steps', 0) or 0)
        if L <= 0: return
        self._undecl = (self._undecl + 1 if not hov else 0) if a else 0
        if self._undecl >= L: self.crashed = True

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
        self._deadline(a, hov)
        return _iq(Sv, _phi(self.sv)), _iq(Sg, _phi(self.sg)), a

    def step(self) -> bool:
        self.t += 1
        return self.crashed or self.t >= self.ep_steps
