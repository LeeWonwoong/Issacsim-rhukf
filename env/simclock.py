"""env/simclock.py — sim 스탬프 기준 UKF 드레인 (SIMCLOCK_UKF=1 전용, ROS 무관 순수 로직).

원칙: 예측 1회 = GT 스탬프 1개 = sim 0.02 s.  GPS 격자(스탬프 % 100000 µs == 0)에서만 GPS 업데이트와 RL 스텝.
벽시계·RTF·학습 부하와 무관하게 GPS 당 예측 5회, RL 간격 sim 0.100 s 가 구성상 고정된다.

노드는 GT 콜백에서 push_gt(스탬프, 스냅샷), GPS 콜백에서 push_gps(스탬프, 데이터) 를 부르고
pop() 이 None 을 줄 때까지 항목을 처리한다. None = 대기(큐 빔 · 격자에서 같은 스탬프 GPS 대기).

★rev2(09-23 리뷰 반영): 드레인은 **행동 결정을 기다리지 않는다.** 학습기가 늦어 a_k 가 격자 t_{k+1} 을 넘겨도
격자 k+1 의 UKF 업데이트·δ 발행·setpoint 는 sim 시각대로 진행되고(노드가 RL 앞부분만 a_k 뒤로 미룬다),
학습기 지연은 '행동 적용 지연(overrun)' 하나로만 나타난다.

행동 적용(ACT_LAT_SIM): RL 스텝 k(격자 t_k) 뒤 request(k, t_k) → 결정 나면 decide(payload) →
스탬프 ≥ t_k + L(GT 격자로 올림, L=0 이면 t_k 다음 스탬프) 인 항목에 apply 로 실려 나온다.
요청은 FIFO 로 여러 개 쌓일 수 있고(결정 지연 중), 적용도 요청 순서대로만 일어난다.
결정이 그 기한 스탬프를 처리한 뒤에 나면 다음 스탬프에 실리고 overrun=True (L=0 이면 기한 = t_k 다음 GT).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

GT_US = 20000          # GT 50 Hz (run_sim step_counter % 5, 물리 4 ms)
GRID_US = 100000       # GPS·RL 10 Hz (step_counter % 25)


def stamp_us(sec: int, nanosec: int) -> int:
    """ROS header.stamp → 정수 µs (run_sim SIMCLOCK_UKF 스탬프는 4000 µs 배수라 손실 없음)."""
    return int(sec) * 1_000_000 + int(nanosec) // 1000


@dataclass
class Applied:
    payload: Any
    k: int
    t_k: int
    lat_us: int                      # 적용 스탬프 − t_k
    overrun: bool


@dataclass
class Item:
    stamp: int
    snap: Any
    grid: bool = False               # 이 항목에서 GPS 업데이트 + RL 스텝
    grid_stamp: int = -1             # 격자 시각 t_k (격자 GT 결번 시 stamp 와 다를 수 있음)
    gps: Any = None                  # 같은 격자 스탬프 GPS (없으면 None → fresh=False)
    gps_gap: bool = False
    n_pred: int = 0                  # 직전 격자 이후 예측 수(이번 포함, 결번 보간 포함) — 격자 항목에서만 의미
    gt_gap: int = 0                  # 직전 격자 이후 결번 GT 수 — 격자 항목에서만 의미
    dt_gps_us: int = -1              # 직전 격자와의 간격 (첫 격자 −1)
    missing: int = 0                 # 직전 처리 스탬프와 이 스탬프 사이 결번 GT 수(노드가 직전 스냅샷으로 예측 보간)
    applies: List[Applied] = field(default_factory=list)   # 이번 스탬프에 적용할 행동(요청 순서)

    # 구 인터페이스 호환(마지막 적용 1개)
    @property
    def apply(self):
        return self.applies[-1].payload if self.applies else None

    @property
    def apply_lat_us(self):
        return self.applies[-1].lat_us if self.applies else -1

    @property
    def overrun(self):
        return bool(self.applies and self.applies[-1].overrun)


class SimClockDrain:
    def __init__(self, gps_wait_gt: int = 2, act_lat_s: float = 0.0, gt_us: int = GT_US, grid_us: int = GRID_US):
        if not (0.0 <= float(act_lat_s) < grid_us * 1e-6):
            raise ValueError(f'ACT_LAT_SIM={act_lat_s} 는 [0, {grid_us * 1e-6}) 이어야 한다 (다음 RL 스텝 전에 적용)')
        self.gt_us, self.grid_us = int(gt_us), int(grid_us)
        self.gps_wait = max(0, int(gps_wait_gt))
        # 지연은 GT 격자로 올림(적용은 GT 스탬프에서만 일어나므로): 0.04 → 40000, 0.05 → 60000
        lat = int(round(float(act_lat_s) * 1e6))
        self.act_lat_us = -(-lat // self.gt_us) * self.gt_us
        if self.act_lat_us >= self.grid_us:
            raise ValueError(f'ACT_LAT_SIM={act_lat_s} 가 GT 격자로 올리면 {self.act_lat_us} µs ≥ RL 주기')
        self.target_us = max(self.act_lat_us, self.gt_us)   # 적용 목표 = t_k + L (L=0 이면 다음 GT) — 넘기면 overrun
        self.n_regress = 0; self.n_dup = 0; self.n_gps_late = 0; self.n_orphan_decide = 0
        self.reset()

    # ── 상태 ──────────────────────────────────────────────────────────────
    def reset(self):
        self.q = deque()              # (stamp, snap)
        self.gps = {}                 # stamp → data
        self.last_in: Optional[int] = None     # 마지막으로 받은 GT 스탬프
        self.last_done: Optional[int] = None   # 마지막으로 처리한 GT 스탬프
        self.last_grid: Optional[int] = None
        self._n_pred = 0; self._gt_gap = 0
        self.pend = deque()           # 행동 요청 FIFO: dict(k, t_k, decided, payload)

    @property
    def pending(self):
        """가장 오래된 행동 요청(없으면 None) — 구 인터페이스 호환."""
        return self.pend[0] if self.pend else None

    def push_gt(self, stamp: int, snap) -> str:
        """반환: 'ok' | 'dup'(같은 스탬프, 무시) | 'regress'(역행 → 전부 비우고 이 스탬프부터)."""
        stamp = int(stamp)
        ret = 'ok'
        if self.last_in is not None and stamp <= self.last_in:
            if stamp == self.last_in:
                self.n_dup += 1
                return 'dup'
            self.reset(); self.n_regress += 1; ret = 'regress'
        self.q.append((stamp, snap)); self.last_in = stamp
        return ret

    def push_gps(self, stamp: int, data) -> bool:
        """이미 처리한 격자 이하의 GPS 는 버린다(False)."""
        stamp = int(stamp)
        if self.last_done is not None and stamp <= self.last_done:
            self.n_gps_late += 1
            return False
        self.gps[stamp] = data
        return True

    # ── 행동 적용 ──────────────────────────────────────────────────────────
    def request(self, k: int, t_k: int):
        self.pend.append(dict(k=int(k), t_k=int(t_k), decided=False, payload=None))

    def decide(self, payload) -> bool:
        """가장 오래된 미결정 요청에 결정을 싣는다. 요청이 없으면(리셋으로 지워짐) 무시하고 False."""
        for p in self.pend:
            if not p['decided']:
                p['decided'] = True; p['payload'] = payload
                return True
        self.n_orphan_decide += 1
        return False

    def outstanding(self) -> bool:
        return any(not p['decided'] for p in self.pend)

    def oldest_undecided_t(self) -> Optional[int]:
        for p in self.pend:
            if not p['decided']:
                return p['t_k']
        return None

    # ── 드레인 ────────────────────────────────────────────────────────────
    def _grid_of(self, s: int) -> Optional[int]:
        if s % self.grid_us == 0:
            return s
        g = (s // self.grid_us) * self.grid_us
        if self.last_done is not None and g > self.last_done:
            return g                    # 격자 GT 결번 → 그 뒤 첫 스탬프에서 격자 처리(RL 스텝 수 보존)
        return None

    def pop(self) -> Optional[Item]:
        if not self.q:
            return None
        s, snap = self.q[0]
        g = self._grid_of(s)
        gps = None; gap = False
        if g is not None:
            gps = self.gps.get(g)
            if gps is None:
                if self.last_in < s + self.gps_wait * self.gt_us:
                    return None         # GPS 대기
                gap = True
        self.q.popleft()
        it = Item(stamp=s, snap=snap)
        if self.last_done is not None:
            it.missing = max(0, (s - self.last_done) // self.gt_us - 1)
            self._gt_gap += it.missing
        self._n_pred += 1 + it.missing               # 노드가 결번 칸을 직전 스냅샷으로 예측 보간(Σdt 보존)
        while self.pend and self.pend[0]['decided']:
            p = self.pend[0]
            if not (s > p['t_k'] and s >= p['t_k'] + self.act_lat_us):
                break
            it.applies.append(Applied(payload=p['payload'], k=p['k'], t_k=p['t_k'], lat_us=s - p['t_k'],
                                      overrun=bool(s > p['t_k'] + self.target_us)))
            self.pend.popleft()
        if g is not None:
            it.grid = True; it.grid_stamp = g; it.gps = gps; it.gps_gap = gap
            it.n_pred = self._n_pred; it.gt_gap = self._gt_gap
            it.dt_gps_us = (g - self.last_grid) if self.last_grid is not None else -1
            self._n_pred = 0; self._gt_gap = 0; self.last_grid = g
            for k in [k for k in self.gps if k <= g]:
                del self.gps[k]
        self.last_done = s
        return it
