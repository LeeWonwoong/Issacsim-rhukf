#!/usr/bin/env python3
"""surr_cusum_eval (09-13) — surrogate v5b 환경에서 고정 탐지기 베이스라인: CUSUM(gyro NIS) / 단순 임계값, 약속 D 동일 적용.
   h 는 per-step FPR 목표(기본 0.02)에 맞춰 격자 보정(FPR-matched) + 최고 F1 h 도 보고. 인자: D [n_ep] [fpr_target]"""
import sys, os, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
N = 'results/claudecodefortest/night'
os.environ.update({'SURR_V7': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(f'{N}/train_pool_v5b.npz'), 'SURR_ATK_PROB': '0.5',
                   'ATK_DELTA_LO': '0.15', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
                   'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5'})
import surrogate_env7 as SE
D = int(sys.argv[1]) if len(sys.argv) > 1 else 5; NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 300; FPR_T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
def run(kind, h, k=0.35, mu0=0.5, seed=7, n_ep=NEP):
    env = SE.SurrogateEnv(seed=seed); tp = fp = fn = tn = 0; wtp = wfn = stp = sfn = 0; delays = []; crashes = 0; n_atk = 0
    for ep in range(n_ep):
        env.reset(); prev = 0; dwell = 0; S = 0.0; onset = None; det = None; hist_g = []
        for t in range(300):
            v, g, a = env.nis(prev)
            dl = float(env.dl[min(env.t, len(env.dl) - 1)]) if a else 0.0; wk = dl < 0.35
            if a:
                if onset is None: onset = t
                if prev == 1:
                    tp += 1; wtp += wk; stp += (not wk)
                    if det is None: det = t
                else:
                    fn += 1; wfn += wk; sfn += (not wk)
            else:
                if prev == 1: fp += 1
                else: tn += 1
            if dwell > 0: act = 1; dwell -= 1
            else:
                if kind == 'cusum': S = max(0.0, S + g - mu0 - k); alarm = S > h
                else: alarm = g > h
                if alarm and prev == 0: act = 1; dwell = D - 1; S = 0.0
                elif alarm: act = 1
                else: act = 0; S = S if kind == 'cusum' and prev == 0 else 0.0
            prev = act
            if env.step(): break
        if onset is not None: n_atk += 1; delays.append((det - onset) if det is not None else 300 - onset)
        crashes += int(env.crashed)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(kind=kind, h=h, F1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, fpr=fp / (fp + tn) if fp + tn else 0.0, recall=rec, prec=prec,
                delay=float(np.mean(delays)) if delays else -1.0, wrec=wtp / (wtp + wfn) if wtp + wfn else -1.0, srec=stp / (stp + sfn) if stp + sfn else -1.0, crash=crashes, n_atk=n_atk)
out = {}
for kind, grid in (('cusum', [1, 2, 3, 4, 6, 8, 12, 16, 24]), ('thr', [0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0])):
    rows = [run(kind, h) for h in grid]
    matched = min(rows, key=lambda r: abs(r['fpr'] - FPR_T)); bestf1 = max(rows, key=lambda r: r['F1'])
    out[kind] = dict(grid=rows, fpr_matched=matched, best_f1=bestf1)
    for tag, r in (('FPR-matched', matched), ('best-F1', bestf1)):
        print(f"  {kind:6s} {tag:12s} h={r['h']:<5} F1={r['F1']:.3f} fpr={r['fpr']:.3f} rec={r['recall']:.3f} dly={r['delay']:.2f} wrec={r['wrec']:.2f} srec={r['srec']:.2f} crash={r['crash']}/{r['n_atk']}", flush=True)
json.dump(out, open(f'{N}/cusum_D{D}.json', 'w'), default=float); print('saved', f'{N}/cusum_D{D}.json')
