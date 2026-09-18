#!/usr/bin/env python3
"""agg_tonight — 오늘 밤 세 무대 판정표 + 그림 (night/TONIGHT_RESULT.md, night/tonight_*.png)
   mix: 평시 동률/우위 — 후기 110–149 보상·F1·FP, 전 구간 보상 합(초기 효율), SW−Adam 짝 t (n≤5)
   blk: 첫 노출 계단 — 진입 60–69·블록 60–109 FP·보상, 프로브 FP (S1 g90n3 SW s42·s43 + tonight blk SW s44–46, Adam 은 S1)
   hm : 은닉 모드 — 모드 진입 사건(바람 C→S 첫 진입·재진입, 공격자 K→St 첫 진입) 정렬 후 10 ep 초과 FP·보상, 모드별 정상 구간 성능, 전체 ep60–149
   비교군: Adam 3e-4(문헌 기본) · Adam 1e-3(튜닝) · UKF/EKF-TD(hm)"""
import json, glob, os, sys, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf/etc/scripts')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(fp): font_manager.fontManager.addfont(fp); plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25})
N = '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night'; T = f'{N}/tonight'; C = f'{N}/chain_hz'
os.environ.setdefault('CHAIN_DIR', C)
import tonight_hm as TH
S = TH.SEEDS; TCR = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
COL = {'SWp1': '#1b4f9c', 'SWn5': '#5b9bef', 'SW': '#2a6fdb', 'Adam3e-4': '#e0592a', 'Adam1e-3': '#b8860b', 'UKFp03': '#7a9a3a', 'EKFp03': '#9b6bb3'}
NM = {'SWp1': 'SWIRL P₀0.1', 'SWn5': 'SWIRL N5', 'Adam3e-4ams': 'Adam 3e-4 AMSGrad', 'Adam1e-3ams': 'Adam 1e-3 AMSGrad', 'SW': 'SWIRL', 'Adam3e-4': 'Adam 3e-4', 'Adam1e-3': 'Adam 1e-3', 'UKFp03': 'UKF-TD', 'EKFp03': 'EKF-TD'}
H = {}
for f in glob.glob(f'{T}/[TUA]*_w*.json'):
    for k, m in json.load(open(f)).items(): H[(m['cell'], m['learner'], m['seed'])] = m['hist']
for f in glob.glob(f'{T}/q/*.json'):
    for k, m in json.load(open(f)).items():
        H[(m['cell'], m['learner'], m['seed'])] = m['hist']
        if m['cell'].startswith('x'): H[(m['cell'][1:], m['learner'], m['seed'])] = m['hist']   # 연장 셀 이름 x<cell> → <cell> 별칭(AMSGrad on 보강분이 per5·per20·mix 표에 붙도록)
for f in glob.glob(f'{C}/S1*_w*.json'):
    for k, m in json.load(open(f)).items():
        if m['cell'] == 'g90n3' and m['learner'] in ('SW', 'Adam3e-4', 'Adam1e-3'): H[('blk', m['learner'], m['seed'])] = m['hist']
def a(h, key): return np.array([np.nan if x.get(key) is None else float(x[key]) for x in h], float)
def wm(h, key, lo, hi, atk=False):
    v = [float(x[key]) for x in h[lo:hi] if x.get(key) is not None and (not atk or x.get('has_atk'))]
    v = [x for x in v if not np.isnan(x)]; return float(np.mean(v)) if v else np.nan
def pt(d):
    d = np.array([x for x in d if not np.isnan(x)], float); n = len(d)
    if n < 2: return n, (float(d.mean()) if n else np.nan), np.nan, int((d > 0).sum())
    sd = d.std(ddof=1); return n, float(d.mean()), (float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan), int((d > 0).sum())
def fp(p, nd=3): n, mu, t, pos = p; return f"{mu:+.{nd}f} (t {t:+.2f}, {pos}/{n})" if n else '—'
L = ['# 오늘 밤 세 무대 결과 (자동 생성 `etc/scripts/agg_tonight.py`)', '', f'- 새 풀 v5e · 하한 0.10 · γ0.9 · n-step 3 · 짝 무결성 on · 시드 {S[0]}–{S[-1]}. 괄호 = 짝 평균 (t, SW 우세 시드 수/n). 짝 t 는 n≤5 라 탐색 수준.', '- 헤드라인 Adam 비교군은 **AMSGrad on**(사용자 09-16 결정). AMSGrad off 는 PyTorch 기본값이며 같은 표에 함께 싣는다.', '']
def table(stage, rows, comps):
    L.append('| 지표 | ' + ' | '.join(f'{NM[l]} 평균' for l in ['SW'] + comps) + ' | ' + ' | '.join(f'SW − {NM[l]}' for l in comps) + ' |')
    L.append('|' + '---|' * (2 + 2 * len(comps)))
    for label, fn, higher_better in rows:
        vals = {l: [fn(H[(stage, l, s)]) if (stage, l, s) in H else np.nan for s in S] for l in ['SW'] + comps}
        cells = [f"{np.nanmean(vals[l]):.3f}" if np.any(~np.isnan(vals[l])) else '—' for l in ['SW'] + comps]
        diffs = []
        for l in comps:
            d = [x - y for x, y in zip(vals['SW'], vals[l])]
            n, mu, t, pos = pt(d)
            if not higher_better: pos = n - pos - sum(1 for x in d if not np.isnan(x) and x == 0)
            diffs.append(f"{mu:+.3f} (t {t:+.2f}, SW 우세 {pos}/{n})" if n else '—')
        L.append(f'| {label} | ' + ' | '.join(cells) + ' | ' + ' | '.join(diffs) + ' |')
    L.append('')
L += ['## 1. 평시 (정상 티어 혼합 mix)', '']
table('mix', [('후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150), True), ('후기 F1 110–149 (공격 에피)', lambda h: wm(h, 'f1', 110, 150, True), True),
              ('후기 오탐 110–149', lambda h: wm(h, 'fpr', 110, 150), False), ('초기 보상 0–39', lambda h: wm(h, 'reward', 0, 40), True), ('전 구간 보상', lambda h: wm(h, 'reward', 0, 150), True)], ['Adam3e-4ams', 'Adam1e-3ams', 'Adam3e-4', 'Adam1e-3'])
L += ['## 2. 첫 노출 급변 (blk)', '']
table('blk', [('블록 전 오탐 40–59', lambda h: wm(h, 'fpr', 40, 60), False), ('진입 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70), False), ('블록 오탐 60–109', lambda h: wm(h, 'fpr', 60, 110), False),
              ('진입 프로브 오탐 60–69', lambda h: wm(h, 'probe_fpr', 60, 70), False), ('블록 보상 60–109', lambda h: wm(h, 'reward', 60, 110), True), ('후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150), True),
              ('진입 보상 60–69', lambda h: wm(h, 'reward', 60, 70), True)], ['Adam3e-4ams', 'Adam1e-3ams', 'Adam3e-4', 'Adam1e-3', 'UKFp03', 'EKFp03'])
# 가치 시야 축: 사슬 S1(급변 블록) γ·n-step 셀, SW 는 S1 + 대기열 g97n1 보강분
S1H = {}
for f in glob.glob(f'{C}/S1*_w*.json'):
    for k, m in json.load(open(f)).items(): S1H[(m['cell'], m['learner'], m['seed'])] = m['hist']
for k, v in H.items():
    if k[0] == 'g97n1': S1H[k] = v
L += ['## 0b. 가치 시야(γ·n-step) — 급변 블록, 새 풀', '', '| 셀 | SW 시드 | 블록 보상 SW − Adam 3e-4 | SW − Adam 1e-3 | 진입 오탐 SW / Adam 3e-4 / Adam 1e-3 |', '|---|---|---|---|---|']
for cell, lab in (('g90n3', 'γ0.90·n3'), ('g95n3', 'γ0.95·n3'), ('g97n3', 'γ0.97·n3'), ('g90n1', 'γ0.90·n1'), ('g95n1', 'γ0.95·n1'), ('g97n1', 'γ0.97·n1')):
    ss = [sd for sd in S if (cell, 'SW', sd) in S1H]
    if not ss: continue
    d3 = pt([wm(S1H[(cell, 'SW', sd)], 'reward', 60, 110) - wm(S1H[(cell, 'Adam3e-4', sd)], 'reward', 60, 110) for sd in ss if (cell, 'Adam3e-4', sd) in S1H])
    d1 = pt([wm(S1H[(cell, 'SW', sd)], 'reward', 60, 110) - wm(S1H[(cell, 'Adam1e-3', sd)], 'reward', 60, 110) for sd in ss if (cell, 'Adam1e-3', sd) in S1H])
    fe = lambda l: np.nanmean([wm(S1H[(cell, l, sd)], 'fpr', 60, 70) for sd in ss if (cell, l, sd) in S1H])
    L.append(f"| {lab} | {len(ss)} | {fp(d3, 1)} | {fp(d1, 1)} | {fe('SW'):.3f} / {fe('Adam3e-4'):.3f} / {fe('Adam1e-3'):.3f} |")
L.append('')
# 지속시간 스윕: 강풍 전체 비율 0.3 동일, 지속만 다름 (mix = 무작위 1.4 ep, per5, per20)
import tonight_queue as TQ
def storm_seq(cell, sd):
    if cell == 'mix': return None
    return TQ.pers_env(sd, cell)[1]
L += ['## 0. 핵심 — 조건 지속시간 스윕 (강풍 비율 0.3 동일, 지속만 바꿈)', '', '| 셀 | 강풍 평균 지속 | 지표 | SWIRL | Adam 3e-4 AMS | Adam 1e-3 AMS | Adam 3e-4 (off) | Adam 1e-3 (off) | SW − Adam 3e-4 AMS | SW − Adam 1e-3 AMS | SW − Adam 3e-4 (off) |', '|' + '---|' * 11]
def ent_excess(h, seq, key='fpr', w=10):
    ent = [i for i in range(10, 145) if seq[i] and not seq[i - 1]]
    v = [wm(h, key, e, min(e + w, 150)) - wm(h, key, max(e - 10, 0), e) for e in ent]
    v = [x for x in v if not np.isnan(x)]; return float(np.mean(v)) if v else np.nan
for cell, dwell in (('mix', '1.4 ep (무작위)'), ('per5', '5 ep'), ('per20', '20 ep')):
    mets = [('보상 30–149', lambda h, sd: wm(h, 'reward', 30, 150), True), ('오탐 30–149', lambda h, sd: wm(h, 'fpr', 30, 150), False)]
    if cell != 'mix': mets.append(('강풍 진입 뒤 초과 오탐(10 ep)', lambda h, sd: ent_excess(h, storm_seq(cell, sd)), False))
    for lab, fn, hb in mets:
        LL = ('SW', 'Adam3e-4ams', 'Adam1e-3ams', 'Adam3e-4', 'Adam1e-3')
        vals = {l: [fn(H[(cell, l, sd)], sd) if (cell, l, sd) in H else np.nan for sd in S] for l in LL}
        mean = lambda v: f"{np.nanmean(v):.3f}" if np.any(~np.isnan(v)) else '—'
        ds = []
        for l in ('Adam3e-4ams', 'Adam1e-3ams', 'Adam3e-4'):
            n, mu, t, pos = pt([x - y for x, y in zip(vals['SW'], vals[l])]); better = pos if hb else n - pos
            ds.append(f"{mu:+.3f} (t {t:+.2f}, SW 우세 {better}/{n})" if n else '—')
        L.append(f"| {cell} | {dwell} | {lab} | " + ' | '.join(mean(vals[l]) for l in LL) + f' | {ds[0]} | {ds[1]} | {ds[2]} |')
L.append('')
# AMSGrad on/off (같은 lr·시드 짝)
L += ['## 4. Adam AMSGrad off(기본) vs on', '', '| 무대 | lr | 지표 | off 평균 | on 평균 | on − off (t, on 이 나은 시드/n) |', '|---|---|---|---|---|---|']
for stg, mets in (('mix', [('후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150), True), ('후기 오탐', lambda h: wm(h, 'fpr', 110, 150), False)]),
                  ('blk', [('진입 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70), False), ('블록 보상 60–109', lambda h: wm(h, 'reward', 60, 110), True)]),
                  ('hm', [('투입 후 보상 60–149', lambda h: wm(h, 'reward', 60, 150), True), ('투입 후 오탐 60–149', lambda h: wm(h, 'fpr', 60, 150), False)])):
    for lr in ('3e-4', '1e-3'):
        for lab, fn, hb in mets:
            off = [fn(H[(stg, f'Adam{lr}', sd)]) if (stg, f'Adam{lr}', sd) in H else np.nan for sd in S]
            on = [fn(H[(stg, f'Adam{lr}ams', sd)]) if (stg, f'Adam{lr}ams', sd) in H else np.nan for sd in S]
            d = [y - x for x, y in zip(off, on)]; n, mu, t, pos = pt(d)
            better = pos if hb else (n - pos)
            L.append(f"| {stg} | {lr} | {lab} | {np.nanmean(off):.3f} | {np.nanmean(on) if n else np.nan:.3f} | {mu:+.3f} (t {t:+.2f}, {better}/{n}) |" if n else f"| {stg} | {lr} | {lab} | {np.nanmean(off):.3f} | — | — |")
L.append('')
# SWIRL 재튜닝 (시드 42–44, 기본 SW 와 짝)
L += ['## 5. SWIRL 재튜닝 (시드 42–44, 새 풀)', '', '| 무대 | 지표 | SW 기본 | P₀0.1 | N5 | P₀0.1 − 기본 | N5 − 기본 |', '|---|---|---|---|---|---|---|']
for stg, lab, fn in (('blk', '진입 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70)), ('blk', '블록 보상 60–109', lambda h: wm(h, 'reward', 60, 110)), ('hm', '투입 후 보상 60–149', lambda h: wm(h, 'reward', 60, 150)), ('mix', '후기 보상 110–149', lambda h: wm(h, 'reward', 110, 150))):
    base = [fn(H[(stg, 'SW', sd)]) if (stg, 'SW', sd) in H else np.nan for sd in (42, 43, 44)]
    cols = []; diffs = []
    for v in ('SWp1', 'SWn5'):
        vv = [fn(H[(stg, v, sd)]) if (stg, v, sd) in H else np.nan for sd in (42, 43, 44)]
        cols.append(f"{np.nanmean(vv):.3f}" if np.any(~np.isnan(vv)) else '—'); diffs.append(fp(pt([y - x for x, y in zip(base, vv)])))
    L.append(f"| {stg} | {lab} | {np.nanmean(base):.3f} | {cols[0]} | {cols[1]} | {diffs[0]} | {diffs[1]} |")
L.append('')
# hm 사건 정렬
def events(sd):
    _, w, at = TH.hm_env(sd); ev = []
    for name, seq in (('바람 C→S', w), ('공격자 K→St', at)):
        k = 0
        for i in range(len(seq)):
            if seq[i] and (i == 0 or not seq[i - 1]): ev.append((name, k, 60 + i)); k += 1
    return ev
def exc(h, e0, key='fpr', w=10):   # 진입 뒤 w ep 평균 − 진입 전 10 ep 평균 (초과 오탐)
    return wm(h, key, e0, min(e0 + w, 150)) - wm(h, key, max(e0 - 10, 0), e0)
L += ['## 3. 은닉 모드 무대 (hm: 바람 × 공격자 마르코프, ep60 투입)', '']
table('hm', [('투입 전 보상 40–59', lambda h: wm(h, 'reward', 40, 60), True), ('투입 후 보상 60–149', lambda h: wm(h, 'reward', 60, 150), True), ('투입 후 오탐 60–149', lambda h: wm(h, 'fpr', 60, 150), False),
             ('투입 후 F1 60–149 (공격 에피)', lambda h: wm(h, 'f1', 60, 150, True), True), ('투입 직후 오탐 60–69', lambda h: wm(h, 'fpr', 60, 70), False)], ['Adam3e-4', 'Adam1e-3', 'UKFp03', 'EKFp03'])
L += ['### 3.1 모드 전환 사건별 초과 오탐 (진입 뒤 10 ep − 직전 10 ep) · SW − 비교군 짝', '', '| 사건 | n 시드 | SW 평균 | Adam 3e-4 | Adam 1e-3 | SW − Adam 3e-4 | SW − Adam 1e-3 |', '|---|---|---|---|---|---|---|']
for name in ('바람 C→S', '공격자 K→St'):
    for kth, lab in ((0, '첫 진입'), (1, '재진입')):
        vals = {l: [] for l in ('SW', 'Adam3e-4', 'Adam1e-3')}
        for sd in S:
            evs = [e for e in events(sd) if e[0] == name and e[1] == kth and e[2] < 145]
            if not evs: continue
            for l in vals: vals[l].append(exc(H[('hm', l, sd)], evs[0][2]) if ('hm', l, sd) in H else np.nan)
        if not vals['SW']: L.append(f'| {name} {lab} | 0 | — | — | — | — | — |'); continue
        d1 = [x - y for x, y in zip(vals['SW'], vals['Adam3e-4'])]; d2 = [x - y for x, y in zip(vals['SW'], vals['Adam1e-3'])]
        n1 = pt(d1); n2 = pt(d2)
        L.append(f"| {name} {lab} | {len(vals['SW'])} | {np.nanmean(vals['SW']):+.3f} | {np.nanmean(vals['Adam3e-4']):+.3f} | {np.nanmean(vals['Adam1e-3']):+.3f} | {n1[1]:+.3f} (t {n1[2]:+.2f}, SW 낮음 {n1[0] - n1[3]}/{n1[0]}) | {n2[1]:+.3f} (t {n2[2]:+.2f}, SW 낮음 {n2[0] - n2[3]}/{n2[0]}) |")
L.append('')
# 그림: 세 무대 보상·오탐 곡선 (이동평균 5, 평균±SE)
def roll(x, w=5): return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) if np.any(~np.isnan(x[max(0, i - w + 1):i + 1])) else np.nan for i in range(len(x))])
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
for j, (stage, title) in enumerate((('mix', '평시: 정상 티어 혼합'), ('blk', '첫 노출 급변: ep60–109 강풍'), ('hm', '은닉 모드: ep60 투입 뒤 바람·공격자 전환'))):
    for i, key in enumerate(('reward', 'fpr')):
        ax = axs[i, j]
        for l in ('UKFp03', 'EKFp03', 'Adam1e-3', 'Adam3e-4', 'SW'):
            M = [roll(a(H[(stage, l, s)], key)) for s in S if (stage, l, s) in H]
            if not M: continue
            M = np.array(M); mu = np.nanmean(M, 0); se = np.nanstd(M, 0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else 0 * mu
            thin = l in ('UKFp03', 'EKFp03')
            ax.plot(mu, color=COL[l], lw=1.2 if thin else 2.1, label=f'{NM[l]} (n={len(M)})'); ax.fill_between(np.arange(len(mu)), mu - se, mu + se, color=COL[l], alpha=0.08 if thin else 0.18, lw=0)
        if stage == 'blk': ax.axvspan(60, 110, color='#8a8f98', alpha=0.1, lw=0)
        if stage == 'hm': ax.axvline(60, color='#333', lw=1)
        ax.set_title(f"{title} — {'에피소드 보상' if key == 'reward' else '학습 중 오탐률'} (이동평균 5)", loc='left', fontsize=10)
        if key == 'fpr': ax.set_ylim(0, 0.2)
    axs[1, j].set_xlabel('에피소드')
axs[0, 0].legend(fontsize=8, loc='lower right')
fig.suptitle('오늘 밤 세 무대 (surrogate, 새 풀 v5e, 시드 42–46)', fontsize=13, x=0.01, ha='left'); fig.tight_layout(); fig.savefig(f'{N}/tonight_three_stages.png', dpi=105); plt.close(fig)
L.append(f'그림: `night/tonight_three_stages.png` · 완료 런 수 {len(H)}')
open(f'{N}/TONIGHT_RESULT.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

try:   # 09-16 새벽 연장 ext_G 표도 같이 갱신(10:03 최종 종합 cron 이 agg_tonight 을 부른다)
    import subprocess, sys as _s; subprocess.run([_s.executable, '/home/acsl/projects/Issacsim-rhukf/etc/scripts/agg_ext.py'], stdout=subprocess.DEVNULL, timeout=600)
except Exception as _e: print('agg_ext 실패', _e)
