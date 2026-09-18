#!/usr/bin/env python3
# speed_ab_compare v2 — speed3(신, 타이머수정) vs speed1(기존 batch3b) per-optimizer 검증.
#   동일 seed42·config·env → 에피별 시나리오 동일. speed 불변이면 per-에피 reward/F1 일치.
import csv, glob, sys, numpy as np, re

import os
# 신 코드 speed1 이 있으면 우선(apples-to-apples), 없으면 기존 batch3b speed1 로 폴백
def base(fresh, fallback): return fresh if os.path.isdir(fresh) else fallback
PAIRS = [
    ('RHUKF', 'results_spdtest_swirl_spd3', base('results_spdtest_swirl_spd1', 'results_g1_B_r15_pd001')),
    ('Adam',  'results_spdtest_adam_spd3',  base('results_spdtest_adam_spd1',  'results_g1_adam_n16')),
]

def load(d):
    f = glob.glob(f'{d}/metrics_*.csv')
    return list(csv.DictReader(open(f[0]))) if f else None

def col(rows, k): return np.array([float(r[k]) for r in rows])

def learn_stats(d):
    """마지막 learn-step 로그의 mean·n (skip 진단: n 이 spd1 대비 줄면 스킵)."""
    try:
        t = open(glob.glob(f'{d}/train.log')[0]).read()
        m = re.findall(r'learn-step: mean=([\d.]+)ms.*?\(n=(\d+)\)', t)
        return (float(m[-1][0]), int(m[-1][1])) if m else (None, None)
    except Exception:
        return (None, None)

def rtf(d):
    try:
        t = open(glob.glob(f'{d}/train.log')[0]).read()
        m = re.findall(r'달성=([\d.]+)x', t)
        return float(np.median([float(x) for x in m])) if m else None
    except Exception:
        return None

for name, d3, d1 in PAIRS:
    a = load(d1); b = load(d3)   # a=speed1 기존, b=speed3 신
    print(f'\n{"="*60}\n {name}:  speed3(신) vs speed1(기존 {d1.split("/")[-1]})\n{"="*60}')
    if not b:
        print(f'  ⏳ speed3 결과 아직 없음 ({d3})'); continue
    if not a:
        print(f'  ⚠ speed1 기준선 없음 ({d1})'); continue
    n = min(len(a), len(b))
    ra, rb = col(a[:n],'reward'), col(b[:n],'reward')
    dr = np.abs(ra - rb); corr = np.corrcoef(ra, rb)[0,1] if n > 2 else float('nan')
    # pooled F1 (첫 n에피)
    def pooled(rows):
        TP=sum(int(r['tp']) for r in rows); FP=sum(int(r['fp']) for r in rows); FN=sum(int(r['fn']) for r in rows)
        P=TP/(TP+FP) if TP+FP else 0; R=TP/(TP+FN) if TP+FN else 0
        return 2*P*R/(P+R) if P+R else 0
    f1a, f1b = pooled(a[:n]), pooled(b[:n])
    lm1,ln1 = learn_stats(d1); lm3,ln3 = learn_stats(d3)
    rtf3 = rtf(d3)
    print(f'  에피수 비교: {n}')
    print(f'  per-에피 reward:  |Δ|mean={dr.mean():.2f}  max={dr.max():.2f}  corr={corr:.4f}')
    print(f'  pooled F1:        spd1={f1a:.3f}  spd3={f1b:.3f}  Δ={abs(f1a-f1b):.3f}')
    print(f'  learn-step:       spd1 mean={lm1}ms n={ln1}  |  spd3 mean={lm3}ms n={ln3}')
    if ln1 and ln3: print(f'    → learn n 비율 spd3/spd1 = {ln3/ln1:.2f} (1.0=스킵없음, <1=learn 못따라옴)')
    print(f'  RTF 달성(speed3): {rtf3}x (요청 3.0)')
    # 판정
    ok_equiv = (corr>0.9 and dr.mean()<15 and abs(f1a-f1b)<0.06)
    ok_rtf = (rtf3 is not None and rtf3 >= 2.7)
    ok_learn = (ln1 is None or ln3 is None or ln3/ln1 > 0.9)
    verdict = 'PASS ✅' if (ok_equiv and ok_rtf and ok_learn) else 'FAIL ❌'
    flags = []
    if not ok_equiv: flags.append('결과불일치')
    if not ok_rtf: flags.append(f'RTF미달({rtf3})')
    if not ok_learn: flags.append('learn스킵')
    print(f'  판정: {verdict} {("— "+", ".join(flags)) if flags else ""}')
print()
