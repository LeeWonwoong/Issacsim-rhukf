"""SIMCLOCK_UKF 드레인 단위 테스트 (ROS·Isaac 없음): python3 -m pytest -q tests/test_simclock.py

합성 스탬프 열로 확인하는 것:
  · GPS 당 예측 5회, RL 간격 100000 µs — 도착이 몰려도(노드 지연)·GPS 가 늦어도 동일
  · 행동 적용 지연 = ACT_LAT_SIM (GT 격자) 고정, 결정이 늦으면 overrun
  · GPS 결번 → gps_gap 후 진행, 늦게 온 GPS 는 버림 · GT 결번 → gt_gap, 격자 GT 결번이어도 RL 스텝 수 보존
  · 스탬프 역행(HARD 재기동)·에피 리셋 → 큐 비움
  · 행동 미결정이어도 다음 격자는 멈추지 않는다(rev2) — 요청은 FIFO 로 쌓이고, 결정 뒤 순서대로 적용(늦으면 overrun)
  · 결정 없는 decide(리셋 뒤)는 무시·계수
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.simclock import SimClockDrain, stamp_us, GT_US, GRID_US  # noqa: E402

T0 = 1_000_000 * 40          # 임의 시작(sim 40 s, 격자 정렬)


def _events(n_gt, t0=T0, gps_delay_gt=0, drop_gps=(), drop_gt=()):
    """run_sim 발행 순서: 매 GT 뒤에(격자면) 같은 스탬프 GPS. gps_delay_gt>0 이면 GPS 가 그만큼 GT 뒤에 도착."""
    ev = []; late = []
    for i in range(n_gt):
        s = t0 + (i + 1) * GT_US
        if s not in drop_gt:
            ev.append(('gt', s))
        for (due, g) in [x for x in late if x[0] == i]:
            ev.append(('gps', g)); late.remove((due, g))
        if s % GRID_US == 0 and s not in drop_gps:
            if gps_delay_gt:
                late.append((i + gps_delay_gt, s))
            else:
                ev.append(('gps', s))
    for (_, g) in late:
        ev.append(('gps', g))
    return ev


class _Node:
    """노드 드레인 루프의 최소 모형. decide_after = 결정이 나기까지 기다리는 '도착 이벤트 수'(학습기 벽시계 지연 흉내).
    결정은 FIFO(학습기 회신 순서) — 앞 결정이 안 났으면 뒤 결정도 안 난다."""
    def __init__(self, lat=0.0, decide_after=0, gps_wait=2):
        self.d = SimClockDrain(gps_wait_gt=gps_wait, act_lat_s=lat)
        self.decide_after = decide_after
        self.grids = []; self.applies = []; self.preds = 0; self.k = 0
        self._waits = []                          # [남은 이벤트 수, k]

    def feed(self, ev_batch):
        """이벤트 묶음을 받은 뒤 한 번 드레인(노드가 멈췄다 몰아서 받는 경우). 이벤트 1개 = 벽시계 한 칸."""
        for kind, s in ev_batch:
            if kind == 'gt':
                self.d.push_gt(s, None)
            else:
                self.d.push_gps(s, ('gps', s))
            for w in self._waits:                 # 학습기 회신 도착(드레인 시작의 poll 과 같은 위치)
                w[0] -= 1
            while self._waits and self._waits[0][0] <= 0:
                self.d.decide({'k': self._waits.pop(0)[1]})
        self.drain()

    def stream(self, events):
        for e in events:
            self.feed([e])
        return self

    def drain(self):
        while True:
            it = self.d.pop()
            if it is None:
                return
            self.preds += 1
            for a in it.applies:
                assert a.payload['k'] == a.k
                self.applies.append((a.k, it.stamp, a.lat_us, a.overrun))
            if it.grid:
                self.grids.append(it)
                self.d.request(self.k, it.grid_stamp); self.k += 1
                if self.decide_after == 0 and not self._waits:
                    self.d.decide({'k': self.k - 1})
                else:
                    self._waits.append([self.decide_after, self.k - 1])


def _check_regular(node, lat_us, skip_first=1):
    g = node.grids[skip_first:]
    assert g, 'no grids'
    assert all(it.n_pred == 5 for it in g), [it.n_pred for it in g]
    assert all(it.dt_gps_us == GRID_US for it in g), [it.dt_gps_us for it in g]
    assert all(not it.gps_gap and it.gt_gap == 0 and it.gps is not None and it.gps[1] == it.grid_stamp for it in g)
    lats = [a[2] for a in node.applies]
    want = lat_us if lat_us > 0 else GT_US
    assert lats and all(x == want for x in lats), lats
    assert not any(a[3] for a in node.applies)


def test_stamp_us():
    assert stamp_us(40, 20_000_000) == 40_020_000 and stamp_us(0, 0) == 0


def test_normal_stream_and_fixed_latency():
    for lat, lat_us in ((0.0, 0), (0.04, 40000), (0.05, 60000), (0.08, 80000)):
        n = _Node(lat=lat).stream(_events(500))
        assert len(n.grids) == 100
        _check_regular(n, lat_us)


def test_bunched_arrival_same_as_stream():
    """노드가 멈췄다가 몰아서 받아도(RELIABLE 큐) 처리 결과가 스트림과 같다 = 벽시계와 무관."""
    ev = _events(1000)
    ref = _Node(lat=0.04).stream(ev)
    rng = random.Random(3)
    for trial in range(20):
        n = _Node(lat=0.04); i = 0
        while i < len(ev):
            m = rng.choice([1, 1, 2, 7, 25, 60]); n.feed(ev[i:i + m]); i += m
        assert [(g.grid_stamp, g.n_pred) for g in n.grids] == [(g.grid_stamp, g.n_pred) for g in ref.grids]
        assert n.applies == ref.applies
        _check_regular(n, 40000)


def test_gps_late_within_wait():
    n = _Node(lat=0.04, gps_wait=2).stream(_events(500, gps_delay_gt=1))
    _check_regular(n, 40000)


def test_gps_late_beyond_wait_gap_and_discard():
    n = _Node(lat=0.0, gps_wait=2).stream(_events(200, gps_delay_gt=3))
    g = n.grids
    assert all(it.gps_gap and it.gps is None for it in g[:-1])          # 끝까지 늦음 → 전부 gap (RL 스텝 수는 유지)
    assert all(it.n_pred == 5 and it.dt_gps_us == GRID_US for it in g[1:])
    assert n.d.n_gps_late >= len(g) - 2                                  # 처리 뒤 도착한 GPS 는 버림


def test_gps_dropped():
    drop = {T0 + 30 * GT_US, T0 + 55 * GT_US}                          # 격자 두 곳 GPS 결번
    n = _Node(lat=0.04).stream(_events(400, drop_gps=drop))
    gaps = [it.grid_stamp for it in n.grids if it.gps_gap]
    assert sorted(gaps) == sorted(drop)
    assert all(it.n_pred == 5 for it in n.grids[1:])


def test_gt_gap_nongrid_and_grid():
    drop = {T0 + 12 * GT_US,                                            # 비격자 결번
            T0 + 20 * GT_US}                                            # 격자 GT 결번 (GPS 는 옴)
    n = _Node(lat=0.0).stream(_events(100, drop_gt=drop))
    by = {it.grid_stamp: it for it in n.grids}
    assert by[T0 + 15 * GT_US].gt_gap == 1 and by[T0 + 15 * GT_US].n_pred == 5     # 결번 칸은 노드가 직전 스냅샷으로 보간 예측
    assert [it.missing for it in n.grids if it.missing] == [1]  # 격자 항목 중 결번 뒤 첫 항목은 T0+21GT 하나
    g = by[T0 + 20 * GT_US]                                             # 격자 GT 가 없어도 다음 스탬프에서 격자 처리
    assert g.stamp == T0 + 21 * GT_US and g.gps is not None and g.gt_gap == 1
    assert len(n.grids) == 20                                           # RL 스텝 수 보존


def test_regress_and_reset():
    n = _Node(lat=0.0).stream(_events(100))
    k0 = len(n.grids)
    n.stream(_events(50, t0=0))                                           # HARD 재기동: 스탬프 0 부터
    assert n.d.n_regress == 1
    later = n.grids[k0:]
    assert len(later) == 10 and all(it.grid_stamp <= 1_000_000 for it in later)
    assert all(it.n_pred == 5 for it in later[1:])
    n.d.reset(); assert not n.d.q and n.d.pending is None and n.d.last_done is None


def test_undecided_does_not_block_grid():
    """rev2: 결정 대기 중에도 격자(GPS 업데이트·RL 스텝 발화)는 sim 시각대로 나오고, 요청은 쌓였다가 결정 뒤 순서대로 적용."""
    d = SimClockDrain(act_lat_s=0.04)
    for kind, s in _events(10):                                         # 첫 격자 T0+100000 까지 처리
        (d.push_gt(s, None) if kind == 'gt' else d.push_gps(s, 0))
    items = []
    while (it := d.pop()) is not None:
        items.append(it)
    grid = [it for it in items if it.grid]
    assert len(grid) == 2
    d.request(0, grid[-1].grid_stamp)                                   # 결정 대기
    for kind, s in _events(15, t0=grid[-1].grid_stamp):
        (d.push_gt(s, None) if kind == 'gt' else d.push_gps(s, 0))
    more = []
    while (it := d.pop()) is not None:
        more.append(it)
        if it.grid:
            d.request(len(more), it.grid_stamp)                         # 다음 RL 스텝도 요청(보류된 앞부분)
    assert [it.grid for it in more][:5] == [False] * 4 + [True]         # 멈추지 않는다
    assert all(it.n_pred == 5 and it.gps is not None for it in more if it.grid)
    assert not any(it.applies for it in more) and d.outstanding() and len(d.pend) == 4   # 원 요청 + 격자 3개
    d.decide({'k': 0}); d.decide({'k': 1})                              # 두 결정이 한꺼번에 도착(FIFO)
    for kind, s in _events(2, t0=more[-1].stamp):
        (d.push_gt(s, None) if kind == 'gt' else d.push_gps(s, 0))
    it = d.pop()
    assert [a.payload['k'] for a in it.applies] == [0, 1] and all(a.overrun for a in it.applies)
    assert [a.t_k for a in it.applies] == [grid[-1].grid_stamp, grid[-1].grid_stamp + GRID_US]
    assert it.apply == {'k': 1} and d.outstanding()                     # k=2 는 아직 미결정


def test_orphan_decide_ignored():
    d = SimClockDrain()
    assert d.decide({'k': 0}) is False and d.n_orphan_decide == 1
    d.request(0, 100000); d.reset()                                      # 리셋(역행·에피 종료) 뒤 늦게 온 결정
    assert d.decide({'k': 0}) is False and d.n_orphan_decide == 2 and d.pending is None


def test_slow_decision_overrun_counts():
    n = _Node(lat=0.04, decide_after=3).stream(_events(300))            # 결정이 GT 3개 뒤 → 기한(2칸) 초과
    assert all(a[3] for a in n.applies) and all(a[2] == 60000 for a in n.applies)
    assert all(it.n_pred == 5 and it.dt_gps_us == GRID_US for it in n.grids[1:])
    n2 = _Node(lat=0.04, decide_after=1).stream(_events(300))           # 기한 안 → 고정 0.04
    _check_regular(n2, 40000)
    n3 = _Node(lat=0.04, decide_after=8).stream(_events(300))           # 결정이 다음 격자 뒤(이벤트 8개 ≈ GT 7개) → 격자는 그대로, 적용만 늦음
    assert all(it.n_pred == 5 and it.dt_gps_us == GRID_US and not it.gps_gap for it in n3.grids[1:])
    assert len(n3.grids) == 60 and [a[0] for a in n3.applies] == list(range(len(n3.applies)))
    assert all(a[3] and a[2] == 140000 for a in n3.applies)


def test_overrun_with_zero_latency():
    n = _Node(lat=0.0, decide_after=2).stream(_events(200))             # L=0: 기한 = 다음 GT, 늦으면 overrun
    assert all(a[3] and a[2] == 40000 for a in n.applies)


def test_bad_latency_rejected():
    for bad in (-0.01, 0.1, 0.09):
        try:
            SimClockDrain(act_lat_s=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
