#!/usr/bin/env python3
"""agg_all_0915 — 09-14/15 surrogate·Isaac 결과를 한 파일(night/AGG_0915.md)로 재생성. ultracode 총종합의 단일 수치 원본.
   포함: v28(구 풀·돌풍 잠정), v29(튜닝 격자), v30(+v30b 병합, 급변 블록), v31(다이얼), Isaac v30(비짝)·v30p(짝, 있는 만큼).
   probe = 에피소드별 greedy 프로브 F1(surrogate). Isaac 은 ε-greedy 학습곡선(공격 에피 F1)."""
import glob, json, csv, os, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); N = 'results/claudecodefortest/night'; R = 'results/claudecodefortest'
out = []
def P(s=''): out.append(s)
def load(prefix):
    t = {}
    for f in glob.glob(f'{N}/{prefix}gpu_w*.json') + glob.glob(f'{N}/{prefix}cpu_w*.json'): t.update(json.load(open(f)))
    return t
def pf(h): return np.array([x.get('probe_f1') if x.get('probe_f1') is not None else np.nan for x in h], float)
def roll(a, w=5):
    r = np.full(len(a), np.nan)
    for i in range(len(a)):
        s = a[max(0, i - w + 1):i + 1]; s = s[~np.isnan(s)]; r[i] = s.mean() if len(s) else np.nan
    return r
def recov(p, start, ref, tol=0.03, w=5):
    r = roll(p, w)
    for i in range(start + w, min(start + 50, len(r))):
        if r[i] >= ref - tol: return i - start
    return 50
def welch(a, b):
    a = np.array(a, float); b = np.array(b, float)
    if len(a) < 2 or len(b) < 2: return float('nan')
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)
def ms(v): v = np.array(v, float); return f"{v.mean():.3f}±{v.std(ddof=1) if len(v) > 1 else 0:.3f}"
def split(k):
    cell, ag, sd = k.rsplit('_', 2); return cell, ag, int(sd[1:])
P('# AGG_0915 — 09-14/15 결과 집계(자동 생성, `etc/scripts/agg_all_0915.py`)'); P()
# ── v28
t = load('v28')
if t:
    P('## v28 — 구 v5b 풀, 돌풍 잠정 배율(G2.5/V2.0) 창, 시드 3, 150 ep. 셀 gU=돌풍·uniform50k, nU=무돌풍'); P()
    P('| cell | agent | n | F1 | t vs Adam | fpr | gustFP | rwd | auc100 | probe ep0-4 / 5-9 / 10-19 |'); P('|---|---|---|---|---|---|---|---|---|---|')
    rows = {}
    for k, m in t.items():
        cell, ag, sd = split(k); h = m['hist']; p = pf(h)
        fg = [x['fpr_gust'] for x in h[100:] if x.get('had_gust') and x.get('fpr_gust') is not None]
        rows.setdefault((cell, ag), []).append(dict(F1=m['F1'], fpr=m['fpr'], rwd=m['rwd'], auc=m['auc100'], fg=np.mean(fg) if fg else np.nan, p5=np.nanmean(p[:5]), p10=np.nanmean(p[5:10]), p20=np.nanmean(p[10:20])))
    for cell in ('gU', 'nU'):
        ad = [r['F1'] for r in rows.get((cell, 'Adam'), [])]
        for ag in ('SWa', 'SWb', 'Adam', 'UKF', 'EKF'):
            v = rows.get((cell, ag))
            if not v: continue
            g = lambda k: np.nanmean([r[k] for r in v])
            P(f"| {cell} | {ag} | {len(v)} | {ms([r['F1'] for r in v])} | {welch([r['F1'] for r in v], ad):+.1f} | {g('fpr'):.3f} | {g('fg'):.3f} | {g('rwd'):.1f} | {g('auc'):.0f} | {g('p5'):.2f} / {g('p10'):.2f} / {g('p20'):.2f} |")
    P()
# ── v29
t = load('v29')
if t:
    P('## v29 — v5c 풀·env8 티어 혼합 ws0/7/10=.4/.3/.3·하한 0.05, 시드 3(42–44), 150 ep. SWIRL 격자 + UKF/EKF-TD P₀ + Adam 3e-4'); P()
    P('| config | n | F1 | s42/s43/s44 | Δ vs Adam(짝) | fpr | rwd | auc100 | probe 0-4/5-9/10-19 |'); P('|---|---|---|---|---|---|---|---|---|')
    agg = {}
    for k, m in t.items():
        c = k.rsplit('_s', 1)[0]; sd = int(k.rsplit('_s', 1)[1]); h = m['hist']; p = pf(h)
        agg.setdefault(c, []).append(dict(sd=sd, F1=m['F1'], fpr=m['fpr'], rwd=m['rwd'], auc=m['auc100'], p5=np.nanmean(p[:5]), p10=np.nanmean(p[5:10]), p20=np.nanmean(p[10:20])))
    ad = {r['sd']: r['F1'] for r in agg.get('Adam', [])}
    for c in sorted(agg, key=lambda c: -np.mean([r['F1'] for r in agg[c]])):
        v = sorted(agg[c], key=lambda r: r['sd']); g = lambda k: np.nanmean([r[k] for r in v])
        d = np.mean([r['F1'] - ad.get(r['sd'], np.nan) for r in v]) if ad else np.nan
        seeds = '/'.join('%.3f' % r['F1'] for r in v)
        P(f"| {c} | {len(v)} | {g('F1'):.3f} | {seeds} | {d:+.3f} | {g('fpr'):.3f} | {g('rwd'):.1f} | {g('auc'):.0f} | {g('p5'):.2f}/{g('p10'):.2f}/{g('p20'):.2f} |")
    P()
# ── v30 (+v30b)
t = load('v30'); tb = load('v30b')
for k, m in tb.items(): t['B_' + k] = m
if t:
    nb = len(tb)
    P(f'## v30 — 급변 블록(v5d 풀, SWIRL P0.03·R1·N7), 150 ep. blk = ep0–59 ws0/7=.6/.4 → 60–109 ws10 100 % → 110– 복귀; mix = 정상 혼합. 버퍼 50k/10k. 시드 42–46(+v30b 47–51 blk50k, 병합 {nb} 런)'); P()
    P('| cell | agent | n | F1 late | probe pre(40-59) | 진입직후(60-69) | 낙폭 | 재수렴 ep(진입) | 재수렴 ep(이탈) | FP blk | FP post | rwd | probe 5-9 | t(낙폭 vs SW) | t(재수렴 vs SW) | t(FP blk vs SW) |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = {}
    for k, m in t.items():
        kk = k[2:] if k.startswith('B_') else k
        cell, ag, sd = split(kk); h = m['hist']; p = pf(h); f = np.array([x['fpr'] for x in h], float)
        pre = np.nanmean(p[40:60])
        rows.setdefault((cell, ag), []).append(dict(sd=sd, F1=m['F1'], pre=pre, be=np.nanmean(p[60:70]), dip=pre - np.nanmean(p[60:70]), ri=recov(p, 60, pre), ro=recov(p, 110, pre), fb=np.nanmean(f[60:110]), fp=np.nanmean(f[110:150]), rwd=m['rwd'], p10=np.nanmean(p[5:10])))
    for cell in ('blk50k', 'blk10k', 'mix50k', 'mix10k'):
        sw = rows.get((cell, 'SW'), [])
        for ag in ('SW', 'Adam', 'UKF', 'EKF'):
            v = rows.get((cell, ag))
            if not v: continue
            g = lambda k: np.nanmean([r[k] for r in v]); tt = lambda k: welch([r[k] for r in v], [r[k] for r in sw]) if ag != 'SW' else 0.0
            P(f"| {cell} | {ag} | {len(v)} | {ms([r['F1'] for r in v])} | {g('pre'):.2f} | {g('be'):.2f} | {g('dip'):.3f} | {g('ri'):.1f} | {g('ro'):.1f} | {g('fb'):.3f} | {g('fp'):.3f} | {g('rwd'):.1f} | {g('p10'):.2f} | {tt('dip'):+.1f} | {tt('ri'):+.1f} | {tt('fb'):+.1f} |")
    P(); P('- 재수렴 ep = 이동평균5 프로브 F1 이 pre−0.03 을 회복하기까지 에피소드 수(최소 5=바닥, 최대 50). t 는 Welch, 양수 = 해당 학습기가 SW 보다 큼(낙폭·재수렴·FP 는 작을수록 좋음).'); P()
# ── v31
t = load('v31')
if t:
    for k, m in load('v30').items():
        if k.startswith('mix50k_'): t['mixref_' + k.split('_', 1)[1]] = m
    P('## v31 — 다이얼(v5d 풀·mix 무대, 시드 42–46, 150 ep). mixref = v30 mix50k 대조'); P()
    P('| cell | 정의 | SW | Adam | UKF | EKF | t(SW−Adam) | probe 5-9 SW/Adam/UKF/EKF |'); P('|---|---|---|---|---|---|---|---|')
    defs = {'mixref': '정상 혼합(대조)', 'g4mix': 'gyro-only 8D', 'ws10h': 'ws10 비중 0.5', 'nost': '스텔스 off(하한 .15)', 'pa0.2': 'P(공격)=0.2', 'fam6': '.75·U(.05,.5)+.25·U(.80,.84)'}
    rows = {}
    for k, m in t.items():
        cell, ag, sd = split(k); p = pf(m['hist']); rows.setdefault((cell, ag), []).append(dict(F1=m['F1'], p10=np.nanmean(p[5:10])))
    for cell in ('mixref', 'g4mix', 'ws10h', 'nost', 'pa0.2', 'fam6'):
        c = {ag: rows.get((cell, ag), []) for ag in ('SW', 'Adam', 'UKF', 'EKF')}
        if not c['SW']: continue
        f = lambda ag: ms([r['F1'] for r in c[ag]]) if c[ag] else '–'; q = lambda ag: f"{np.nanmean([r['p10'] for r in c[ag]]):.2f}" if c[ag] else '–'
        P(f"| {cell} | {defs[cell]} | {f('SW')} | {f('Adam')} | {f('UKF')} | {f('EKF')} | {welch([r['F1'] for r in c['SW']], [r['F1'] for r in c['Adam']]):+.1f} | {q('SW')}/{q('Adam')}/{q('UKF')}/{q('EKF')} |")
    P()
# ── Isaac v30 / v30p
for tag, title in (('isaac_v30', 'Isaac v30 (비짝 — 학습기별 시나리오 상이, 비교 무효; ekf 는 compile 버그로 무학습)'), ('isaac_v30p', 'Isaac v30p (짝비교, SCENARIO_SEED=42, 09-15 09:33 착수)')):
    dirs = sorted(glob.glob(f'{R}/{tag}_*'))
    if not dirs: continue
    P(f'## {title}'); P(); P('| agent | n ep | pre F1(41–60) | blk F1(61–110) | post F1(111–150) | FP pre/blk/post | rwd blk/post | 블록 약(δ<.3)/강 공격 수 | 약공격 F1 블록 안/밖 | crash | HARD_RESET | GEOFENCE skip |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in dirs:
        ag = d.rsplit('_', 1)[1]; fs = glob.glob(f'{d}/metrics_*.csv')
        if not fs: continue
        Rw = list(csv.DictReader(open(fs[0])))
        if not Rw: continue
        ep = np.array([int(r['episode']) for r in Rw]); f1 = np.array([float(r['f1']) for r in Rw]); fp = np.array([float(r['fp_rate']) for r in Rw]); b = np.array([float(r['bias_scale']) for r in Rw]); rw = np.array([float(r['reward']) for r in Rw]); cr = np.array([int(r['crashed']) for r in Rw])
        f1a = np.where(b > 0, f1, np.nan); w = lambda lo, hi, a: (np.nanmean(a[(ep >= lo) & (ep <= hi)]) if ((ep >= lo) & (ep <= hi)).any() else np.nan)
        blk = (ep >= 61) & (ep <= 110); nw = int(((b > 0) & (b < 0.3) & blk).sum()); ns = int(((b >= 0.3) & blk).sum())
        wk_in = f1a[(b > 0) & (b < 0.3) & blk]; wk_out = f1a[(b > 0) & (b < 0.3) & ~blk]
        log = open(f'{d}/train.log').read() if os.path.exists(f'{d}/train.log') else ''
        P(f"| {ag} | {len(ep)} | {w(41,60,f1a):.2f} | {w(61,110,f1a):.2f} | {w(111,150,f1a):.2f} | {w(41,60,fp):.3f}/{w(61,110,fp):.3f}/{w(111,150,fp):.3f} | {w(61,110,rw):.1f}/{w(111,150,rw):.1f} | {nw}/{ns} | {np.nanmean(wk_in) if len(wk_in) else np.nan:.2f}/{np.nanmean(wk_out) if len(wk_out) else np.nan:.2f} | {cr.sum()} | {log.count('HARD_RESET')} | {log.count('[GEOFENCE]')} |")
    P()
open(f'{N}/AGG_0915.md', 'w').write('\n'.join(out)); print(f'saved {N}/AGG_0915.md ({len(out)} lines)')
