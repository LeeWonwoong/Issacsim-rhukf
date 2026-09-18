#!/usr/bin/env python3
"""agg_v33 — DIRECTION_0915 §7.3 사전등록 규칙 그대로 v33(패널티 k·마르코프 체제) 판정 → night/V33_JUDGE_0915.md
   짝: 시드 42–46, k=1 대조 = v30 blk50k/mix50k 같은 시드. 창(0-based): pre 40–59 · 진입 60–69 · 블록 60–109 · 후기 110–149.
   M1 블록 FP(hist.fpr 60–109 전 에피) · M2 블록 공격에피 F1 · M3 프로브 낙폭(pre − 60–69, s43 제외) · M4 재수렴(MA5 프로브 ≥ pre−tol, 상한 50, s43 제외) · M5 후기 F1(= run 요약 F1, 마지막 공격 에피 20개 — §7.3 기준선 .852/.837 과 같은 정의).
   판정(1차는 §7.3 문언대로 시나리오 짝이 창 끝까지 유지된 시드만, 전 시드는 보조 열): 확대 = SW 유리 방향 ∧ |짝 t| ≥ 2.5 ∧ 부호 전부(5/5, 프로브 4/4) / 축소 = 반대 방향 같은 조건 / 불변 = |평균 ΔG| < 동등 한계 / 그 외 판정불가.
   env: V33_DIR·V30_DIR·OUT(검사용)."""
import glob, json, os, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); N = 'results/claudecodefortest/night'
V33 = os.environ.get('V33_DIR', N); V30 = os.environ.get('V30_DIR', N); OUT = os.environ.get('OUT', f'{N}/V33_JUDGE_0915.md')
S = [42, 43, 44, 45, 46]; SP = [42, 44, 45, 46]
EQ = dict(M1=.005, M2=.015, M3=.04, M4=5.0, M5=.015); LOWER = {'M1', 'M3', 'M4'}   # 작을수록 SW 유리
L = []
def P(s=''): L.append(s); print(s)
def load(d, pre):
    t = {}
    for f in glob.glob(f'{d}/{pre}gpu_w*.json') + glob.glob(f'{d}/{pre}cpu_w*.json'):
        for k, m in json.load(open(f)).items():
            if 'hist' not in m: continue
            cell, ag, sd = k.rsplit('_', 2); t[(cell, ag, int(sd[1:]))] = m
    return t
def arr(h, key): return np.array([x.get(key) if x.get(key) is not None else np.nan for x in h], float)
def roll(a, w=5):
    r = np.full(len(a), np.nan)
    for i in range(len(a)):
        s = a[max(0, i - w + 1):i + 1]; s = s[~np.isnan(s)]; r[i] = s.mean() if len(s) else np.nan
    return r
def recov(p, start, ref, tol=0.03, w=5):
    r = roll(p, w)
    for i in range(start, min(start + 50, len(r))):   # §7.3 기준선(11.5/30.5, t −3.71)을 재현하는 정의: 진입 시점부터 탐색
        if r[i] >= ref - tol: return i - start
    return 50
def atkmean(h, a, b, key):
    v = [x[key] for x in h[a:b] if x.get('has_atk') and x.get(key) is not None and not np.isnan(x[key])]
    return float(np.mean(v)) if v else np.nan
def met(m):
    h = m['hist']; p = arr(h, 'probe_f1'); f = arr(h, 'fpr'); pre = np.nanmean(p[40:60])
    wk = [x['rec'] for x in h[60:110] if x.get('has_atk') and x.get('dmax', 1) < 0.3 and x.get('rec') is not None]
    return dict(M1=np.nanmean(f[60:110]), M2=atkmean(h, 60, 110, 'f1'), M3=pre - np.nanmean(p[60:70]),
                M4=recov(p, 60, pre, .03), M4b=recov(p, 60, pre, .05), M4c=recov(p, 60, pre, .10), M5=m.get('F1', np.nan), F1win=atkmean(h, 110, 150, 'f1'),
                fprL=np.nanmean(f[110:150]), recL=atkmean(h, 110, 150, 'rec'), fprA=np.nanmean(f[20:150]), recA=atkmean(h, 20, 150, 'rec'),
                wrecB=float(np.mean(wk)) if wk else np.nan, fpr=f, atk=np.array([x.get('has_atk', 0) for x in h]), dmax=arr(h, 'dmax'))
def pt(d):
    d = np.array([x for x in d if not np.isnan(x)], float); n = len(d)
    if n < 2: return dict(n=n, mean=float(d.mean()) if n else np.nan, t=np.nan, pos=int((d > 0).sum()), neg=int((d < 0).sum()))
    sd = d.std(ddof=1); return dict(n=n, mean=float(d.mean()), t=float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan, pos=int((d > 0).sum()), neg=int((d < 0).sum()))
def welch(a, b):
    a = np.array([x for x in a if not np.isnan(x)], float); b = np.array([x for x in b if not np.isnan(x)], float)
    if len(a) < 2 or len(b) < 2: return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12))
def f(p, nd=4): return f"{p['mean']:+.{nd}f} (t {p['t']:+.2f}, +{p['pos']}/−{p['neg']}, n={p['n']})"
def paired_ok(a, b, upto=110):   # 시나리오 짝 유지: 두 학습기의 공격 유무·δ 최대가 창 끝까지 같음
    return bool(np.array_equal(a['atk'][:upto], b['atk'][:upto]) and np.allclose(np.nan_to_num(a['dmax'][:upto]), np.nan_to_num(b['dmax'][:upto])))
def verdict(M, p, nseeds):
    if np.isnan(p['mean']): return '판정불가(자료 없음)'
    fav = (p['mean'] < 0) if M in LOWER else (p['mean'] > 0)
    allsign = (p['neg'] if p['mean'] < 0 else p['pos']) == nseeds and p['n'] == nseeds
    if not np.isnan(p['t']) and abs(p['t']) >= 2.5 and allsign: return '확대' if fav else '축소'
    if abs(p['mean']) < EQ[M]: return '불변'
    return '판정불가'

r30, r33 = load(V30, 'v30'), load(V33, 'v33')
M = {k: met(m) for k, m in {**r30, **r33}.items()}
def g(cell, ag, sd): return M.get((cell, ag, sd))
P('# v33 판정 — DIRECTION_0915 §7.3 사전등록 규칙 (자동 생성 `etc/scripts/agg_v33.py`)'); P()
P(f'- 원자료: v33 {len(r33)} 런, v30 {len(r30)} 런. 짝 시드 42–46(프로브 지표 s43 제외). 괄호 = 평균 (짝 t, 양/음 개수, n)'); P()
P('## 1. k=1 기준선 재현 (v30, §7.3 기록값과 대조)'); P()
P('| 지표 | SW | Adam | SW−Adam | §7.3 기록 |'); P('|---|---|---|---|---|')
REC = dict(M1='.033/.050(−.017, t −8.6)', M2='.786/.759(t 3.1)', M3='.105/.232(−.127, t −6.5)', M4='11.5/30.5(t −3.7)', M5='.852/.837 (= summarize 마지막 공격 에피 20개)', F1win='(보조: 공격 에피 110–149 평균)')
for key in ('M1', 'M2', 'M3', 'M4', 'M5', 'F1win'):
    ss = SP if key in ('M3', 'M4') else S
    a = [g('blk50k', 'SW', s)[key] for s in ss if g('blk50k', 'SW', s)]; b = [g('blk50k', 'Adam', s)[key] for s in ss if g('blk50k', 'Adam', s)]
    d = pt([g('blk50k', 'SW', s)[key] - g('blk50k', 'Adam', s)[key] for s in ss if g('blk50k', 'SW', s) and g('blk50k', 'Adam', s)])
    P(f"| {key} | {np.nanmean(a):.3f} | {np.nanmean(b):.3f} | {f(d, 3)} | {REC[key]} |")
ml = pt([g('mix50k', 'SW', s)['M5'] - g('mix50k', 'Adam', s)['M5'] for s in S if g('mix50k', 'SW', s) and g('mix50k', 'Adam', s)])
P(f"| mix50k M5 | {np.nanmean([g('mix50k','SW',s)['M5'] for s in S if g('mix50k','SW',s)]):.3f} | {np.nanmean([g('mix50k','Adam',s)['M5'] for s in S if g('mix50k','Adam',s)]):.3f} | {f(ml,3)} | .841/.831 |"); P()

P('## 2. 패널티 k 셀 판정 (ΔG = (SW−Adam)_k − (SW−Adam)_1)'); P()
P('| 셀 | 지표 | ΔG (짝 유지 시드, 1차) | 판정(1차) | ΔG (전 시드, 보조) | 판정(보조) |'); P('|---|---|---|---|---|---|')
res = {}
for k in (2, 4):
    cell = f'blk50k_k{k}'
    trunc = {ag: sum(1 for s in SP if g(cell, ag, s) and g(cell, ag, s)['M4'] >= 50) for ag in ('SW', 'Adam')}
    for key in ('M1', 'M2', 'M3', 'M4', 'M5'):
        ss = SP if key in ('M3', 'M4') else S; ns = len(ss)
        Gk, G1, dd, ddv = [], [], [], []
        for s in ss:
            a, b, c, e = g(cell, 'SW', s), g(cell, 'Adam', s), g('blk50k', 'SW', s), g('blk50k', 'Adam', s)
            if not (a and b and c and e): continue
            Gk.append(a[key] - b[key]); G1.append(c[key] - e[key]); dd.append(Gk[-1] - G1[-1])
            if paired_ok(a, b) and paired_ok(c, e): ddv.append(dd[-1])
        p, pv = pt(dd), pt(ddv); va = verdict(key, p, ns); v = verdict(key, pv, len(ddv)) if len(ddv) >= 2 else '판정불가(짝 유지 시드 부족)'
        if key == 'M4' and max(trunc.values()) >= 3: v = va = f'해석 안 함(50 절단 SW {trunc["SW"]}·Adam {trunc["Adam"]})'
        res[(cell, key)] = v
        P(f"| {cell} | {key} | {f(pv)} | **{v}** | {f(p)} | {va} |")
    P(f"| {cell} | M4 보조 tol .05/.10(전 시드) | " + ' / '.join(f(pt([g(cell,'SW',s)[kk]-g(cell,'Adam',s)[kk]-(g('blk50k','SW',s)[kk]-g('blk50k','Adam',s)[kk]) for s in SP if g(cell,'SW',s) and g(cell,'Adam',s) and g('blk50k','SW',s) and g('blk50k','Adam',s)]),1) for kk in ('M4b','M4c')) + " | 참고 | — | — |")
P()
P('## 3. mix50k_k4 — 정상 동률과 Huber 보수화 가설'); P()
mxa = pt([g('mix50k_k4', 'SW', s)['M5'] - g('mix50k_k4', 'Adam', s)['M5'] for s in S if g('mix50k_k4', 'SW', s) and g('mix50k_k4', 'Adam', s)])
mx = pt([g('mix50k_k4', 'SW', s)['M5'] - g('mix50k_k4', 'Adam', s)['M5'] for s in S if g('mix50k_k4', 'SW', s) and g('mix50k_k4', 'Adam', s) and paired_ok(g('mix50k_k4', 'SW', s), g('mix50k_k4', 'Adam', s), 150)])
tie = lambda p: '판정불가(자료 없음)' if np.isnan(p['mean']) else ('동률(|Δ| ≤ .02)' if abs(p['mean']) <= .02 else '차이')
P(f"- 후기 F1 SW−Adam(k4), 1차(짝 유지 시드): {f(mx, 3)} → **{tie(mx)}** · 보조(전 시드): {f(mxa, 3)} → {tie(mxa)}"); P()
P('| 학습기 (짝 유지 시드만) | 후기 fpr k4−k1 | 후기 recall k4−k1 | 전체(20–149) fpr | 전체 recall | 둘 다 하락(t ≤ −2) |'); P('|---|---|---|---|---|---|')
sup = rej = 0
for ag in ('SW', 'UKF', 'EKF', 'Adam'):
    okS = [s for s in S if g('mix50k_k4', ag, s) and g('mix50k', ag, s) and paired_ok(g('mix50k_k4', ag, s), g('mix50k', ag, s), 150)]
    pr = {kk: pt([g('mix50k_k4', ag, s)[kk] - g('mix50k', ag, s)[kk] for s in okS]) for kk in ('fprL', 'recL', 'fprA', 'recA')}
    both = pr['fprL']['mean'] < 0 and pr['recL']['mean'] < 0 and pr['fprL']['t'] <= -2 and pr['recL']['t'] <= -2
    if both: sup += 1
    if pr['fprL']['mean'] > 0 and pr['recL']['mean'] > 0: rej += 1
    P(f"| {ag} | {f(pr['fprL'])} | {f(pr['recL'])} | {f(pr['fprA'])} | {f(pr['recA'])} | {'예' if both else '아니오'} |")
hub = '지지' if sup >= 3 else ('기각(recall↑·FP↑ 다수)' if rej >= 3 else '판정불가')
P(); P(f"- **Huber 보수화 가설: {hub}** (4 학습기 중 둘 다 하락 {sup}, 둘 다 상승 {rej}; 1차 창 = 후기 110–149)")
wr = {ag: pt([g('blk50k_k4', ag, s)['wrecB'] - g('blk50k', ag, s)['wrecB'] for s in S if g('blk50k_k4', ag, s) and g('blk50k', ag, s) and paired_ok(g('blk50k_k4', ag, s), g('blk50k', ag, s))]) for ag in ('SW', 'UKF', 'EKF', 'Adam')}
alldown = all((not np.isnan(p['mean'])) and p['mean'] < 0 for p in wr.values())
P('- blk50k_k4 약공격(dmax<0.3) 블록 recall k4−k1 (짝 유지 시드): ' + ' · '.join(f"{ag} {f(p, 3)}" for ag, p in wr.items()) + f" → **{'전원 하락 = 보수화 보조증거' if alldown else '전원 하락 아님 = 보수화 보조증거 없음'}**"); P()

P('## 4. SW vs 칼만-TD (Welch, M1 블록 FP)'); P()
P('| 셀 | vs UKF t | vs EKF t | k=1 대조 t (UKF/EKF) | 변화 판정 |'); P('|---|---|---|---|---|')
for cell in ('blk50k_k2', 'blk50k_k4'):
    row = []
    for kt in ('UKF', 'EKF'):
        tk = welch([g(cell, 'SW', s)['M1'] for s in S if g(cell, 'SW', s)], [g(cell, kt, s)['M1'] for s in S if g(cell, kt, s)])
        t1 = welch([g('blk50k', 'SW', s)['M1'] for s in S if g('blk50k', 'SW', s)], [g('blk50k', kt, s)['M1'] for s in S if g('blk50k', kt, s)])
        row.append((tk, t1))
    ch = ['변화' if not np.isnan(tk) and abs(tk) >= 2.5 and not np.isnan(t1) and np.sign(tk) == np.sign(t1) else '변화 없음' for tk, t1 in row]
    P(f"| {cell} | {row[0][0]:+.2f} | {row[1][0]:+.2f} | {row[0][1]:+.2f} / {row[1][1]:+.2f} | UKF {ch[0]} · EKF {ch[1]} |")
P()

P('## 5. mkv50k — 마르코프 교대 체제 (게이트 전용, 판별 실험 아님)'); P()
def markov(sd):
    rng = np.random.default_rng(1000 + sd); segs = []; ep = 0; calm = True
    while ep < 150:
        e = min(ep + int(rng.integers(20, 41)) - 1, 149); segs.append((ep, e, calm)); ep = e + 1; calm = not calm
    return segs
P('| 시드 | 폭풍 진입 ep | 진입 시 버퍼 ws10 비중(명목) | 1차 진입 10ep FP SW−Adam | 2차 진입 10ep FP SW−Adam | 짝 유지 |'); P('|---|---|---|---|---|---|')
d1, d2 = [], []
for s in S:
    segs = markov(s); storms = [a for a, b, c in segs if not c]
    share = []
    for e0 in storms:
        prior = sum(min(b, e0 - 1) - a + 1 for a, b, c in segs if not c and a < e0); share.append(prior / e0 if e0 else 0)
    a, b = g('mkv50k', 'SW', s), g('mkv50k', 'Adam', s)
    if not (a and b): P(f'| {s} | {storms} | — | 자료 없음 | — | — |'); continue
    x1 = np.nanmean(a['fpr'][storms[0]:storms[0] + 10]) - np.nanmean(b['fpr'][storms[0]:storms[0] + 10]) if storms else np.nan
    x2 = np.nanmean(a['fpr'][storms[1]:storms[1] + 10]) - np.nanmean(b['fpr'][storms[1]:storms[1] + 10]) if len(storms) > 1 else np.nan
    d1.append(x1); d2.append(x2)
    P(f"| {s} | {storms} | {', '.join('%.2f' % x for x in share)} | {x1:+.4f} | {x2:+.4f} | {paired_ok(a, b, 150)} |")
p1, p2 = pt(d1), pt(d2)
mkl = pt([g('mkv50k', 'SW', s)['M5'] - g('mkv50k', 'Adam', s)['M5'] for s in S if g('mkv50k', 'SW', s) and g('mkv50k', 'Adam', s)])
first = (not np.isnan(p1['mean'])) and p1['mean'] < 0 and (not np.isnan(p2['mean'])) and p2['mean'] < 0 and abs(p1['mean']) > abs(p2['mean'])
P(); P(f"- 1차 진입: {f(p1)} · 2차 진입: {f(p2)} → '첫 전환 집중' {'지지' if first else '미지지'} (두 진입 모두 SW 우위 방향 ∧ 1차 격차가 더 큼)")
P(f"- 전체 후기 F1 SW−Adam: {f(mkl, 3)} → {'동률(|Δ| ≤ .02)' if not np.isnan(mkl['mean']) and abs(mkl['mean']) <= .02 else '차이'}"); P()

P('## 6. 해석 규칙 적용 (§7.3 끝)'); P()
for k in (2, 4):
    m1 = res.get((f'blk50k_k{k}', 'M1'), '—')
    if m1 in ('불변', '확대') and k == 2:
        rule = '[미확인] mix50k_k2 셀이 없어 §7.3 의 "mix 동률" 조건을 직접 확인할 수 없음. 참고로 k=1 mix 후기 F1 동률(+0.010, t 0.81)과 k=4 mix(짝 유지 시드) ' + tie(mx) + ' 이 방향 근거' + (' — 확대이므로 β=c=3k 대조 필요' if m1 == '확대' else '')
    elif m1 in ('불변', '확대') and (not np.isnan(mx['mean']) and abs(mx['mean']) <= .02):
        rule = '벌점 크기에 강건한 과도 우위(강건성 문단)' + (' — 확대이므로 β=c=3k 대조 필요' if m1 == '확대' else '')
    elif m1 == '축소' and hub == '지지': rule = "우위는 Huber 비활성 영역(k≤1)에서만(한계 절)"
    elif m1 == '축소': rule = '축소 — 공통 보수화 미지지, 원인 미상'
    else: rule = '판정불가 — 규칙 적용 없음'
    P(f"- k={k}: M1 {m1} → {rule}")
open(OUT, 'w').write('\n'.join(L) + '\n')
