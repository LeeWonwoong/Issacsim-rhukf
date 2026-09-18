#!/usr/bin/env python3
"""decide_config — bandcap(ON30 arm0.07) + bandcap_arm005/003 분석 요약에서 (arm, filter, 밴드) 자동 결정 → decision.json
규칙(2026-09-11 밤샘, 사용자 선호 arm 0.05):
  · FA proxy = ws6 바람 gyro 초과 run ≥5 per100.  G2 가 arm 0.05 에서 FA ≤ 1.5 면 (G2, 0.05).
    아니면 arm 0.03 에서 G2 FA ≤ 1.5 면 (G2, 0.03). 둘 다 아니면 (E, 0.05).
  · 약공격 하한 δ_lo: weakfloor(ON15) 의 선택 필터 공격 고원(무풍 med) 이 선택 arm·필터의 ws6 gyro p99 를 넘는 최소 δ ∈ {0.10,0.15,0.20,0.25}
  · 치명 클래스: δ U(0.80,0.82) × hold U(30,60)  (ON30 δ≥0.80 track 0/10, dhover3 ≥8/10 실측)
  · 약공격: δ U(δ_lo,0.7) × ON U(5,30), rise 0
사용: python3 decide_config.py → results/claudecodefortest/night/decision.json (+ stdout)"""
import re, os, json, sys
os.chdir('/home/acsl/projects/Issacsim-rhukf')
R = 'results/claudecodefortest'
SRC = {0.07: 'results_bandcap/analysis/summary.txt', 0.05: f'{R}/bandcap_arm005/analysis/summary.txt', 0.03: f'{R}/bandcap_arm003/analysis/summary.txt'}
def parse(path):
    """{filter: {'fa5':..,'fa3':..,'wind_p99':..,'wind_vel_auc':..,'a25_vel':..,'a25_gmin':..}}"""
    out = {}
    if not os.path.exists(path): return out
    cur = None
    for line in open(path, errors='replace'):
        m = re.match(r'### (E|G2)', line.strip())
        if m: cur = m.group(1); out[cur] = {}; continue
        if cur is None: continue
        m = re.search(r'ws6 바람 gyro med/p95(?:/p99)? ([\d.]+)/([\d.]+)(?:/([\d.]+))?.*?초과 run ≥3/≥5 per100: ([\d.]+)/([\d.]+).*?clean→wind AUC gyro ([\d.]+) vel ([\d.]+)', line)
        if m:
            out[cur].update(wind_med=float(m.group(1)), wind_p95=float(m.group(2)), wind_p99=float(m.group(3)) if m.group(3) else None,
                            fa3=float(m.group(4)), fa5=float(m.group(5)), wind_g_auc=float(m.group(6)), wind_vel_auc=float(m.group(7)))
        m = re.search(r'δ0\.25 ws6: 공격중 gyro med/min ([\d.]+)/([\d.]+).*?AUC\(4창\) vs 바람: gyro ([\d.]+) vel ([\d.]+)', line)
        if m: out[cur].update(a25_gmed=float(m.group(1)), a25_gmin=float(m.group(2)), a25_g_auc=float(m.group(3)), a25_vel_auc=float(m.group(4)))
    return out
S = {arm: parse(p) for arm, p in SRC.items()}
# weakfloor 고원 (무풍 med) — E/G2
WF = {}
wp = 'results_weakfloor/analysis/summary.txt'
if os.path.exists(wp):
    cur = None
    for line in open(wp, errors='replace'):
        m = re.match(r'### (E|G2)', line.strip())
        if m: cur = m.group(1); WF[cur] = {}; continue
        m = re.search(r'δ(0\.\d+) ws0: 공격중 gyro med/min ([\d.]+)/([\d.]+)', line)
        if m and cur: WF[cur][float(m.group(1))] = (float(m.group(2)), float(m.group(3)))   # (고원 med, 5pct min)
for arm in (0.07, 0.05, 0.03):
    for f in ('E', 'G2'):
        d = S[arm].get(f, {})
        if d: print(f"arm {arm} {f}: FA≥5 {d.get('fa5')}  wind p95/p99 {d.get('wind_p95')}/{d.get('wind_p99')}  wind vel AUC {d.get('wind_vel_auc')}  δ.25 gyro min {d.get('a25_gmin')} vel AUC {d.get('a25_vel_auc')}")
# ★규칙 수정(01:50): 평시-p95 기준 run 은 필터 스케일에 편향(G2 는 clean p95 가 낮아 바람 run 이 과대). 공격 기준 진폭 여유로 판정:
#   margin = ws6 바람 gyro p95 / δ0.25 공격 gyro min (ws6).  feasible ⇔ margin ≤ 0.65.  G2(vel 보조채널) 우선, arm 0.05 우선.
def margin(arm, f):
    d = S.get(arm, {}).get(f, {})
    return d['wind_p95'] / d['a25_gmin'] if d.get('wind_p95') and d.get('a25_gmin') else 9.9
for arm_, f_ in ((0.05, 'G2'), (0.03, 'G2'), (0.07, 'G2'), (0.05, 'E'), (0.07, 'E')):
    print(f"  margin arm {arm_} {f_}: {margin(arm_, f_):.3f}")
MARGIN_MAX = 0.65
if margin(0.05, 'G2') <= MARGIN_MAX: arm, filt = 0.05, 'G2'
elif margin(0.03, 'G2') <= MARGIN_MAX: arm, filt = 0.03, 'G2'
elif margin(0.05, 'E') <= MARGIN_MAX: arm, filt = 0.05, 'E'
else: arm, filt = 0.07, 'E'
p99 = (S.get(arm, {}).get(filt, {}) or {}).get('wind_p99') or (S.get(arm, {}).get(filt, {}) or {}).get('wind_p95', 1.2) * 1.25
d_lo = 0.25   # 규칙: 공격 고원의 5pct(min) 이 선택 조건의 ws6 바람 p99 이상인 최소 δ (= 학습 불가 밴드 배제)
for dl in (0.10, 0.15, 0.20, 0.25):
    v = WF.get(filt, {}).get(dl)
    if v and v[1] >= p99: d_lo = dl; break
# weak_hi 0.6: gapcheck δ0.7×3s ws6 2/15 사망. ON 공통 (10,30). 강공격 δ(0.76,0.82): 기대 P(crash|track) .3~.9, dhover3 추락 ≤.1 (0.82 는 .4)
dec = dict(arm=arm, filter=filt, wind_p99=round(p99, 3), weak_lo=d_lo, weak_hi=0.6, weak_on=[10, 30],
           leth=[0.76, 0.80], leth_hold=[10, 30], p_lethal=0.5, ep_steps=300, note='auto (decide_config.py)',
           src={str(k): v for k, v in S.items()}, weakfloor=WF)
os.makedirs(f'{R}/night', exist_ok=True); json.dump(dec, open(f'{R}/night/decision.json', 'w'), indent=1, ensure_ascii=False)
print('\nDECISION:', {k: dec[k] for k in ('arm', 'filter', 'wind_p99', 'weak_lo', 'weak_on', 'leth', 'leth_hold')})
