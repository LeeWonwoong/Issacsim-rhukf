#!/usr/bin/env python3
"""build_stop_dataset — Isaac track 궤적 → 오프라인 최적정지 데이터셋 (A안, 버퍼 50k).

설계 근거(전부 실측):
  · track 궤적 3,294개 · 842,267 스텝, 64폴더 컬럼 일관
  · 온셋이 98% 170스텝 고정 → s0~U(0,S0_HI) 무작위 절단으로 분산 생성(표준편차 0→41)
  · 무공격 궤적 37개뿐 → 공격 궤적의 온셋 이전 구간(548,110 스텝)을 clean 에피로 사용
  · nis_*_raw 가 CSV 에 있으므로 log1p(sqrt(NIS)) 를 클립 없이 재계산 (09-16 클립 버그 우회)
  · 관측 = [nis_v, nis_g] x 4 프레임 = 8D (prev_action 제외: 정지=종료라 항상 '계속'=상수)

출력 npz: obs(N,8) act_label(N) ep_id(N) atk(N) dmax(N) ws(N) step_in_ep(N) ep_start(M) ep_len(M)
"""
import csv, glob, os, sys
import numpy as np

ROOT = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'
OUT  = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/night/stop_dataset_50k.npz'
N_TARGET = int(os.environ.get('STOP_N', '50000'))   # 목표 전이 수 = 버퍼 크기
P_CLEAN  = float(os.environ.get('STOP_P_CLEAN', '0.5'))   # 무공격 에피 비율
S0_HI    = int(os.environ.get('STOP_S0_HI', '140'))       # 절단 시작 상한
CLEAN_LEN = (int(os.environ.get('STOP_CLEAN_LO', '90')), int(os.environ.get('STOP_CLEAN_HI', '150')))
W = int(os.environ.get('STOP_L', '6'))   # ★final7 과 동일(L=6 → 12D)
SEED = int(os.environ.get('STOP_SEED', '2026'))

def compress(raw):
    """ε̃ = log1p(sqrt(NIS)) — 클립 없음(09-16 OBS_CLIP 버그 우회)"""
    return np.log1p(np.sqrt(np.maximum(np.asarray(raw, float), 0.0)))

def load_tracks():
    """(δ, ws, onset, v_raw[], g_raw[], atk[]) 궤적 목록"""
    out = []
    for f in sorted(glob.glob(f'{ROOT}/*/sweep_detail.csv')):
        tr = {}
        try:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if r.get('policy') != 'track': continue
                    k = (r['bias'], r['pattern'], r['wind_speed'], r['episode'], r['cell_idx'])
                    tr.setdefault(k, []).append(r)
        except Exception:
            continue
        for k, rows in tr.items():
            try:
                rows.sort(key=lambda r: int(r['step']))
                v = compress([r['nis_v_raw'] for r in rows])
                g = compress([r['nis_g_raw'] for r in rows])
                a = np.array([int(r.get('attack_active') or 0) for r in rows], np.int8)
                d = float(k[0]) / 4.36
                ws = float(k[2] or 0)
            except Exception:
                continue
            if len(a) < 60: continue
            on = int(np.argmax(a)) if a.any() else -1
            out.append(dict(d=d, ws=ws, on=on, v=v, g=g, a=a))
    return out

def main():
    rng = np.random.default_rng(SEED)
    T = load_tracks()
    atk_T = [t for t in T if t['on'] >= 0]
    print(f'[build] 궤적 {len(T)} (공격 {len(atk_T)})', flush=True)

    O, A, EP, DM, WS, SI = [], [], [], [], [], []
    ep_start, ep_len = [], []
    ep = 0; n = 0
    while n < N_TARGET:
        t = atk_T[rng.integers(len(atk_T))]
        clean = rng.random() < P_CLEAN
        if clean:
            # 온셋 이전 구간만 사용 → 공격이 끝내 오지 않는 에피소드
            if t['on'] < CLEAN_LEN[0] + W: continue
            L = int(rng.integers(CLEAN_LEN[0], min(CLEAN_LEN[1], t['on']) + 1))
            s0 = int(rng.integers(0, t['on'] - L + 1))
            sl = slice(s0, s0 + L)
        else:
            # 무작위 절단으로 온셋 위치 분산
            s0 = int(rng.integers(0, min(S0_HI, max(1, t['on'] - 10)) + 1))
            sl = slice(s0, len(t['a']))
            if t['on'] - s0 < W + 5: continue
        v, g, a = t['v'][sl], t['g'][sl], t['a'][sl]
        if len(v) < W + 5: continue
        st = ep_start_i = len(O)
        for i in range(W - 1, len(v)):
            O.append(np.concatenate([np.stack([v[i-W+1:i+1], g[i-W+1:i+1]], 1).ravel()]))
            A.append(int(a[i])); EP.append(ep); DM.append(t['d'] if not clean else 0.0)
            WS.append(t['ws']); SI.append(i - (W - 1))
        ep_start.append(ep_start_i); ep_len.append(len(O) - ep_start_i)
        n = len(O); ep += 1

    O = np.array(O, np.float32); A = np.array(A, np.int8)
    np.savez_compressed(OUT, obs=O, atk=A, ep_id=np.array(EP, np.int32),
                        dmax=np.array(DM, np.float32), ws=np.array(WS, np.float32),
                        step_in_ep=np.array(SI, np.int32),
                        ep_start=np.array(ep_start, np.int32), ep_len=np.array(ep_len, np.int32))
    ons = []
    for s, L in zip(ep_start, ep_len):
        seg = A[s:s+L]
        if seg.any(): ons.append(int(np.argmax(seg)))
    print(f'[build] 저장 {OUT}')
    print(f'  전이 {len(O):,} · 에피 {ep} · 관측 {O.shape[1]}D')
    print(f'  공격 에피 {len(ons)} ({100*len(ons)/ep:.0f}%) · 무공격 {ep-len(ons)}')
    if ons: print(f'  온셋 평균 {np.mean(ons):.0f} · 표준편차 {np.std(ons):.0f} · 범위 {min(ons)}~{max(ons)}')
    print(f'  관측 범위 {O.min():.3f}~{O.max():.3f} (평균 {O.mean():.3f})')

main()
