#!/usr/bin/env python3
"""agg_isaac_tonight — Isaac 오늘 밤·연장 런 SW vs Adam 짝 비교 (v30p 분석 방식). 창: 전 40–59 · 진입 60–69 · 블록 60–109 · 후 110–119/149. CSV episode 번호로 정렬.
   무공격 에피 = tp+fn==0. 지표: 무공격 에피 FP 건수 평균, 보상, 공격 에피 F1, 탐지지연, 기록 에피 수. → night/ISAAC_TONIGHT_RESULT.md"""
import csv, glob, os, numpy as np
R = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest'
def load(d):
    f = glob.glob(f'{R}/{d}/metrics_*.csv')
    if not f: return None
    rows = list(csv.DictReader(open(f[0])))
    return {int(float(r['episode'])): r for r in rows}
def win(E, a, b, key, cond=None):
    v = []
    for ep, r in E.items():
        if a <= ep < b and (cond is None or cond(r)):
            try: v.append(float(r[key]))
            except Exception: pass
    return (float(np.mean(v)) if v else np.nan), len(v)
noatk = lambda r: float(r['tp']) + float(r['fn']) == 0
isatk = lambda r: float(r['tp']) + float(r['fn']) > 0
import re
def lstep(d):
    v = [float(m.group(1)) for line in open(f'{R}/{d}/train.log', errors='ignore') for m in [re.search(r'learn-step: mean=([\d.]+)ms', line)] if m] if os.path.exists(f'{R}/{d}/train.log') else []
    return (float(np.median(v)), float(np.mean(np.array(v) > 40))) if v else (np.nan, np.nan)
L = ['# Isaac 오늘 밤 결과 (자동 생성 `etc/scripts/agg_isaac_tonight.py`)', '', '> SWIRL 은 learn-step 이 RL 스텝 예산(40 ms @speed 2.5)을 넘으면 행동이 늦어져 결과가 무효. 표 끝 열에 갱신 시간 중앙값·예산 초과 에피 비율을 싣는다.', '']
pairs = [('isaac_t2_swirl_s42', 'isaac_t_adam_s42', '급변 블록 시드 42 (SWIRL GPU 단독 재실행)'), ('isaac_t2_swirl_s43', 'isaac_t_adam_s43', '급변 블록 시드 43 (SWIRL GPU 단독 재실행)'),
         ('isaac_t_swirl_s42', 'isaac_t_adam_s42', '[무효: SWIRL learn-step 예산 초과] 시드 42'), ('isaac_t_swirl_s43', 'isaac_t_adam_s43', '[무효: SWIRL learn-step 예산 초과] 시드 43'),
         ('isaac_x_blk_swirl_s44', 'isaac_x_blk_adam_s44', '급변 블록 시드 44'), ('isaac_x_per20_swirl_s42', 'isaac_x_per20_adam_s42', '지속 20 시드 42'),
         (None, 'isaac_x_blk_adam1e-3_s42', '급변 블록 시드 42 Adam 1e-3')]
L += ['| 무대 | 학습기 | 기록 ep | 전 40–59 무공격 FP | 진입 60–69 무공격 FP | 블록 60–109 무공격 FP | 블록 보상 | 블록 공격 F1 | 후 110+ 공격 F1 | 블록 지연 | 갱신 ms(초과 비율) |', '|---|---|---|---|---|---|---|---|---|---|---|']
for sw, ad, lab in pairs:
    for d, nm in ((sw, 'SWIRL'), (ad, 'Adam')):
        if d is None: continue
        E = load(d)
        if E is None: L.append(f'| {lab} | {nm} | 자료 없음 | | | | | | | |'); continue
        f = lambda a, b, k, c=None: win(E, a, b, k, c)[0]
        L.append(f"| {lab} | {nm} | {len(E)} | {f(40, 60, 'fp', noatk):.2f} | {f(60, 70, 'fp', noatk):.2f} | {f(60, 110, 'fp', noatk):.2f} | {f(60, 110, 'reward'):.1f} | {f(60, 110, 'f1', isatk):.3f} | {f(110, 151, 'f1', isatk):.3f} | {f(60, 110, 'det_delay', isatk):.2f} | {lstep(d)[0]:.1f} ({lstep(d)[1]:.2f}) |")
out = f'{R}/night/ISAAC_TONIGHT_RESULT.md'; open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
