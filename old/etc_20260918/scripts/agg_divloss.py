#!/usr/bin/env python3
"""agg_divloss — 사용자 1차 검증 기준으로 세 프레임워크를 읽는다.
   ①reward 수렴속도·최고성능 유지  ②loss 감소 패턴  ③Qmax(O(10) 수렴 여부)  ④T_Var(K 밴드 근거)
   원본 rhukf.py 정의와 맞춤: loss=mean(residual^2), T_Var=var(z_measured), Qmax=max_a Q(s,a)."""
import json, glob, re, sys
import numpy as np

def curves(path_glob, key_fn):
    H = {}
    for f in sorted(glob.glob(path_glob)):
        tag = key_fn(f)
        if tag is None: continue
        for nm, m in json.load(open(f)).items():
            H.setdefault(tag, {}).setdefault(nm, []).append(m['hist'])
    return H

def block(a, n=5):
    L = a.shape[1]; w = max(1, L // n)
    return [float(np.nanmean(a[:, i*w:(i+1)*w])) for i in range(n)]

def report(title, H, order):
    for tag in order:
        if tag not in H: continue
        print(f"\n{'='*86}\n=== {title} · {tag} ===")
        print(f"  {'학습기':<14}{'보상 최종':>10}{'90%도달':>9}{'최고→최종':>10}"
              f"{'loss 처음':>11}{'loss 최종':>11}{'loss 추세':>10}{'Qmax':>9}{'T_Var':>9}")
        for nm in sorted(H[tag]):
            hs = H[tag][nm]
            g = lambda k: np.array([[ (x.get(k) if x.get(k) is not None else np.nan) for x in h] for h in hs], float)
            R, Lo, Qm, Tv = g('reward'), g('loss'), g('qmax'), g('tvar')
            m = np.nanmean(R, 0); fin = float(np.nanmean(m[int(len(m)*0.8):]))
            base = float(m[0]); tgt = base + 0.9*(fin-base)
            i = int(np.argmax(m >= tgt)) if (m >= tgt).any() else -1
            sm = np.convolve(m, np.ones(20)/20, 'valid'); peak = float(sm.max()) if len(sm) else fin
            lb = block(Lo); q = float(np.nanmean(Qm[:, int(Qm.shape[1]*0.8):])) if np.isfinite(Qm).any() else float('nan')
            tv = float(np.nanmean(Tv[:, int(Tv.shape[1]*0.8):])) if np.isfinite(Tv).any() else float('nan')
            tr = lb[-1]/lb[0] if lb[0] and lb[0]==lb[0] else float('nan')
            print(f"  {nm:<14}{fin:10.2f}{i:9d}{peak-fin:+10.2f}"
                  f"{lb[0]:11.5f}{lb[-1]:11.5f}{tr:9.2f}x{q:9.3f}{tv:9.4f}")

D = 'results/claudecodefortest/divloss'
k = lambda f: ('원본 div=1' if '_d10_' in f else '/4 div=4' if '_d40_' in f else None)
report('3번 온라인 정지', curves(f'{D}/q3_*.json', k), ['원본 div=1', '/4 div=4'])
report('2번 오프라인 정지', curves(f'{D}/f2_*.json', k), ['원본 div=1', '/4 div=4'])
print("\n  · 아티팩트 기준: Qmax 는 O(10) 수렴이 건강, 이탈 상승은 발산 신호")
print("  · T_Var <2 → 저-K(CartPole 레짐) · >5 → 고-K(LunarLander 레짐)")
print("  · CartPole 정상상태 loss ~0.11 · LL ~0.5 (둘 다 감소 후 평탄)")
