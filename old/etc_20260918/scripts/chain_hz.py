#!/usr/bin/env python3
"""chain_hz (09-15 저녁, 사용자 요청) — γ 상향·n-step 끄기를 '독립 스캔'이 아니라 **발전된 선상**에서 결정하는 사슬.
   각 단계는 앞 단계의 사전등록 판정(D*.json)을 읽어 자기 무대·학습기를 정한다. 판정 규칙은 결과를 보기 전에 이 파일에 고정했다.

   D0  v33(패널티 k) 판정 → k* ∈ {1,2,4}                      [새 런 없음, v33 + v30 k=1 짝]
   S1  시야 선별: blk50k × γ{0.9,0.95,0.97} × n-step{3,1} × SW(GPU) · Adam lr{3e-4,1e-3,3e-3}(CPU), 시드 42–46, 하한 0.10, 패널티 k*
   D1  → 시야 셀 c* (기준 g90n3 에서 벗어나려면 가드 + 기준 대비 DiD 필요)
   S2  c* 에서 학습기 튜닝: SW{N5,N10,P₀0.1,(γ≠0.9 이면 Huber c 스케일)} · UKF/EKF-TD P₀{0.03,0.1} · (γ≠0.9 이면 Adam β 스케일), 시드 42–46
   D2  → SW*, UKF*, EKF*, Adam*(강건성 행) 선택 + 튜닝 시드 사다리
   S3  확인(새 시드): blk@c* SW*·UKF*·EKF* 시드 101–110 → 하한 0.20 대조 SW* 101–110 → (c*≠기준) 기준 SW 101–105 → mix50k SW* 101–105 ; CPU Adam3e-4·Adam* 짝
   D3  최종 보고 REPORT.md (1차 가설 H1 = 블록 보상 SW*−Adam3e-4 짝 t, n=10) → CHAIN_DONE(프레임워크 확정)
   S4  배치 64 절제(사용자 09-15 19:20): 확정 무대 c*·D2 학습기에서 BATCH_SIZE 64. SW* 는 창 N ∈ {10, 14, 2·N*}(창 표본 수 B·N 을 128·N* 과 맞춘 짝 포함),
       UKF*·EKF*(GPU), Adam3e-4·Adam*(CPU), 시드 101–110 → S3 blk_c*(배치 128)와 같은 시드 짝
   D4  REPORT_B64.md (H6 = 블록 보상 격차 SW*−Adam3e-4 의 배치 DiD, 창 표본 동일 짝) → B64_DONE

   창(0-based ep): pre 40–59 · 진입 60–69 · 블록 60–109 · 후기 110–149. 1차 지표 = 블록 에피소드 보상 평균(FP·recall 비용을 함께 담음, 무할인 합이라 γ·n 셀 사이 비교 가능).
   사용: work STAGE WID NW (env JOBSET=gpu|cpu) | count STAGE | decide D0|D1|D2|D3
   env: CHAIN_DIR(결과 폴더), CHAIN_NEP(스모크 테스트 에피 수), V33_DIR·V30_DIR(판정 원자료 폴더)"""
import sys, os, json, time, importlib, glob, numpy as np
ROOT = '/home/acsl/projects/Issacsim-rhukf'; os.chdir(ROOT); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
N = 'results/claudecodefortest/night'; C = os.environ.get('CHAIN_DIR', f'{N}/chain_hz'); os.makedirs(C, exist_ok=True)
V33 = os.environ.get('V33_DIR', N); V30 = os.environ.get('V30_DIR', N)
POOL = f'{N}/train_pool_v5e.npz' if os.path.exists(f'{C}/POOL_V5E_READY') else f'{N}/train_pool_v5d.npz'   # ★09-15: aggressive 수정 풀 v5e 준비 표식이 있으면 v5e
NEP = int(os.environ.get('CHAIN_NEP', '150'))
nan = float('nan')
ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_CLIP': '4.0', 'SURR_POOL': os.path.abspath(POOL), 'SURR_ATK_PROB': '0.5',
        'SURR_PLOC_TIERS': '7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1', 'SURR_DEAD_TIERS': '10:0.78,26,0.80,16,0.84,4',
        'ATK_DELTA_LO': '0.10', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'EPS_DECAY': '4000', 'TERMINAL_PEN': '0', 'GAMMA': '0.9',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '5', 'PROBE_EVERY': '1', 'PROBE_N': '4', 'REPLAY_MODE': 'uniform', 'SURR_EP_RNG': '1'}   # SURR_EP_RNG: 09-15 짝 무결성(A1)
CLR = ('OBS_MODE', 'SURR_TIER_P', 'SURR_TIER_SCHED', 'SURR_FAM_SCHED', 'BUFFER_SIZE', 'REPLAY_HALFLIFE', 'SURR_GUST_P', 'SURR_GUST_LO', 'SURR_GUST_HI', 'SURR_GUST_G', 'SURR_GUST_V', 'SURR_GUST_ATK_AFTER',
       'ADAM_INIT', 'ADAM_HUBER_BETA', 'RHUKF_MODE', 'SURR_WEAK_LO', 'SURR_WEAK_HI', 'SURR_WEAK_ON_LO', 'SURR_WEAK_ON_HI', 'SURR_WEAK_RISE_MAX', 'SURR_P_LETHAL', 'SURR_LETH_LO', 'SURR_LETH_HI', 'ADAM_AMSGRAD',
       'ATK_WEAK_HI', 'SURR_WEAK_INTERP', 'SURR_SHIFT_EP', 'SURR_SHIFT_NOISE', 'SURR_SHIFT_SCALE', 'REWARD_MODE', 'A2_P', 'A2_C', 'A2_C0', 'A2_F', 'A2_INSTALL', 'R_TP', 'R_TN', 'R_FP', 'FN_BASE', 'FN_PER',
       'SURR_OBS_DIV', 'SURR_OBS_CLIP', 'SURR_OBS_CENTER', 'SURR_CLIP', 'ATK_BURSTS', 'FN_ONSET_MULT', 'HOVER_DWELL', 'BATCH_SIZE', 'RHUKF_PD', 'RHUKF_PINIT', 'RHUKF_FORM', 'RHUKF_ALPHA', 'RHUKF_R', 'RHUKF_N', 'RHUKF_HUBER_C', 'RHUKF_SPAS', 'RHUKF_Q', 'RHUKF_TAU', 'RHUKF_UI', 'ADAM_LR', 'OPT', 'ADAM_LOSS', 'RELAPSE_PEN', 'NSTEP_OFF')
BLOCK = '0-59:0:0.6,7:0.4;60-109:10:1;110-:0:0.6,7:0.4'; MIX = '0:0.4,7:0.3,10:0.3'
BASE = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_HUBER_C='3')
SWB = dict(BASE, RHUKF_PINIT='0.03', RHUKF_R='1', RHUKF_N='7')
KTD = dict(NET_HIDDEN=16, RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute', RHUKF_SPAS=0, RHUKF_ALPHA='0.1', RHUKF_Q='1e-3', RHUKF_N='1', RHUKF_R='1', RHUKF_HUBER_C='3', RHUKF_PINIT='0.03')
ADAMP = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='3', ADAM_LR='3e-4')
HZ = {'g90n3': (0.90, 0), 'g95n3': (0.95, 0), 'g97n3': (0.97, 0), 'g90n1': (0.90, 1), 'g95n1': (0.95, 1), 'g97n1': (0.97, 1)}
HZBASE = 'g90n3'; TUNE = list(range(42, 47)); EVAL = list(range(101, 111)); EVAL5 = list(range(101, 106))   # 09-15 17:25 설계도 권고: v29·v30·v30b 선택 근거 시드(42–51)와 완전 분리
ADAM_LRS = {'Adam3e-4': '3e-4', 'Adam1e-3': '1e-3', 'Adam3e-3': '3e-3'}

def adam_primary(SWs): return 'Adam3e-4hb' if SWs == 'SWhc' else 'Adam3e-4'   # 09-15 PRECHAIN B: SW* 가 Huber c 스케일이면 Adam 도 β 같은 배율로 1차 비교
def pen(k): return {} if k == 1 else dict(R_FP=f'{-0.7*k:.3f}', FN_BASE=f'{-0.4*k:.3f}', FN_PER=f'{-0.35*k:.3f}')
def hz_env(c):
    g, off = HZ[c]; e = {'GAMMA': str(g)}
    if off: e['NSTEP_OFF'] = '1'
    return e
def hscale(c): return 0.1 / (1.0 - HZ[c][0])            # Q 척도 배율 1/(1−γ) 를 γ0.9 기준으로
def dec(i):
    p = f'{C}/D{i}.json'
    if not os.path.exists(p): raise SystemExit(f'{p} 없음 — 앞 단계 판정이 먼저 필요')
    return json.load(open(p))
def stage_blk(c, k, **kw): return dict(SURR_TIER_SCHED=BLOCK, BUFFER_SIZE='50000', **hz_env(c), **pen(k), **kw)
def stage_mix(c, k): return dict(SURR_TIER_P=MIX, BUFFER_SIZE='50000', **hz_env(c), **pen(k))

# ---------------- 잡 구성 (단계별, 앞 판정에 의존) ----------------
def learner_env(lrn, d2=None, c=None):
    """학습기 이름 → env dict. S2/S3 변형 이름도 여기서 해석. 'BASE-N14' = BASE 설정에 창 N 만 덮어씀(S4)."""
    if '-N' in lrn:
        base, n = lrn.rsplit('-N', 1); e = learner_env(base, d2, c); e['RHUKF_N'] = str(int(n)); return e
    if lrn == 'SW': return dict(SWB)
    if lrn == 'SWn5': return dict(SWB, RHUKF_N='5')
    if lrn == 'SWn10': return dict(SWB, RHUKF_N='10')
    if lrn == 'SWp1': return dict(SWB, RHUKF_PINIT='0.1')
    if lrn == 'SWp01': return dict(SWB, RHUKF_PINIT='0.01')
    if lrn == 'SWp3': return dict(SWB, RHUKF_PINIT='0.3')
    if lrn == 'SWn3': return dict(SWB, RHUKF_N='3')
    if lrn == 'SWhc': return dict(SWB, RHUKF_HUBER_C=f'{3*hscale(c):.3g}')
    if lrn in ('UKFp03', 'UKF'): return dict(KTD, RHUKF_MODE='ukf')
    if lrn in ('EKFp03', 'EKF'): return dict(KTD, RHUKF_MODE='ekf')
    if lrn == 'UKFp1': return dict(KTD, RHUKF_MODE='ukf', RHUKF_PINIT='0.1')
    if lrn == 'EKFp1': return dict(KTD, RHUKF_MODE='ekf', RHUKF_PINIT='0.1')
    if lrn.startswith('SGD'):   # ★09-16 1차 비적응 베이스라인: OPT=sgd (momentum 0), 이름 SGD<lr> 예: SGD1e-3
        return dict(ADAMP, ADAM_LR=lrn[3:] or '1e-3', OPT='sgd')
    if lrn in ADAM_LRS: return dict(ADAMP, ADAM_LR=ADAM_LRS[lrn])
    if lrn.endswith('hb') and lrn[:-2] in ADAM_LRS: return dict(ADAMP, ADAM_LR=ADAM_LRS[lrn[:-2]], ADAM_HUBER_BETA=f'{3*hscale(c):.3g}')
    raise ValueError(lrn)

def build(stage, js):
    """반환: [(name, learner_env, agent_type, seed, stage_extra, cell, learner)] — GPU 는 우선순위 순."""
    jobs = []
    def add(cell, lrn, sd, sx):
        ag = 'adam' if lrn.startswith('Adam') else 'rhukf'
        if (js == 'gpu') != (ag == 'rhukf'): return
        c = sx.get('_hz'); sx2 = {k: v for k, v in sx.items() if k != '_hz'}
        jobs.append((f'{cell}_{lrn}_s{sd}', learner_env(lrn, c=c), ag, sd, sx2, cell, lrn))
    if stage == 'S1':
        k = dec(0)['k']
        for sd in TUNE:
            for c in HZ:
                sx = dict(stage_blk(c, k), _hz=c)
                add(c, 'SW', sd, sx)
                for a in ADAM_LRS: add(c, a, sd, sx)
    elif stage == 'S2':
        d1 = dec(1); c, k = d1['cell'], d1['k']; sx = dict(stage_blk(c, k), _hz=c)
        sw = ['SWn5', 'SWn10', 'SWp1'] + (['SWhc'] if HZ[c][0] != 0.9 else [])
        kt = ['UKFp03', 'EKFp03', 'UKFp1', 'EKFp1']
        ad = (['Adam3e-4hb', 'Adam1e-3hb'] if HZ[c][0] != 0.9 else [])
        for sd in TUNE:
            for l in sw + kt + ad: add(c, l, sd, sx)
    elif stage == 'S3':
        d1, d2 = dec(1), dec(2); c, k = d1['cell'], d1['k']
        SWs, UKs, EKs, ADs = d2['SW'], d2['UKF'], d2['EKF'], d2['Adam']
        AP = adam_primary(SWs); adams = list(dict.fromkeys(['Adam3e-4', AP, ADs]))
        blk = dict(stage_blk(c, k), _hz=c)
        for sd in EVAL:                                         # 우선 1: 본 확인
            for l in [SWs, UKs, EKs] + adams: add(f'blk_{c}', l, sd, blk)
        ctl = dict(stage_blk(c, k, ATK_DELTA_LO='0.20'), _hz=c)
        for sd in EVAL:                                         # 우선 2: 하한 0.20 대조(반전 대역 제거)
            for l in [SWs] + adams: add(f'ctl020_{c}', l, sd, ctl)
        if c != HZBASE:                                         # 우선 3: 시야 레버의 새 시드 확인
            bb = dict(stage_blk(HZBASE, k), _hz=HZBASE)
            for sd in EVAL5:
                for l in [SWs] + list(dict.fromkeys(['Adam3e-4', AP])): add(f'blk_{HZBASE}', l, sd, bb)   # 리뷰 수정: 같은 SW* 로 두 셀 비교(튜닝 이득이 시야 효과로 섞이지 않게)
        mx = dict(stage_mix(c, k), _hz=c)
        for sd in EVAL5:                                        # 우선 4: 정상 혼합 동률 확인
            for l in [SWs] + adams: add(f'mix_{c}', l, sd, mx)
    elif stage == 'S4':
        d1, d2 = dec(1), dec(2); c, k = d1['cell'], d1['k']
        SWs, UKs, EKs, ADs = d2['SW'], d2['UKF'], d2['EKF'], d2['Adam']
        nstar = int(learner_env(SWs, c=c)['RHUKF_N']); ns = sorted({10, 14, 2 * nstar})
        AP = adam_primary(SWs); adams = list(dict.fromkeys(['Adam3e-4', AP, ADs]))
        b64 = dict(stage_blk(c, k, BATCH_SIZE='64'), _hz=c)
        for sd in EVAL:
            for l in [f'{SWs}-N{n}' for n in ns] + [UKs, EKs] + adams: add(f'b64_{c}', l, sd, b64)
    else:
        raise ValueError(stage)
    names = [j[0] for j in jobs]; assert len(set(names)) == len(names), 'job name 중복'
    return jobs

# ---------------- 실행 ----------------
def run(job):
    name, lenv, ag, sd, extra, cell, lrn = job
    os.environ.update(ENV0)
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = str(v)
    import env.reward as RW; importlib.reload(RW); import surrogate_env8 as SE; importlib.reload(SE); import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(dict(lenv), n_ep=NEP, seed=sd, ep_steps=300, agent_type=ag)
    m = dict(cell=cell, learner=lrn, seed=sd, extra=dict(extra), lenv=dict(lenv), n_ep=NEP, sec=time.time() - t0, pool=os.path.basename(POOL), hist=[dict(r) for r in h])
    mm = metrics(m)
    print(f"  {name:28s} Rblk={mm['Rblk']:.2f} FPblk={mm['FPblk']:.4f} FPent={mm['FPent']:.4f} F1blk={mm['F1blk']:.3f} F1late={mm['F1late']:.3f} learn={mm['learn_ok']:.2f} cr={mm['crash']} ({m['sec']:.0f}s)", flush=True)
    return m

def wmean(h, a, b, key, atk=False):
    v = [r.get(key, nan) for r in h[a:b] if (not atk or r.get('has_atk'))]
    v = [float(x) for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else nan
def metrics(m):
    h = m['hist']
    return dict(Rblk=wmean(h, 60, 110, 'reward'), Rpre=wmean(h, 40, 60, 'reward'), Rlate=wmean(h, 110, 150, 'reward'),
                FPblk=wmean(h, 60, 110, 'fpr'), FPent=wmean(h, 60, 70, 'fpr'), FPpre=wmean(h, 40, 60, 'fpr'),
                F1blk=wmean(h, 60, 110, 'f1', True), F1late=wmean(h, 110, 150, 'f1', True), RECblk=wmean(h, 60, 110, 'rec', True),
                PRblk=wmean(h, 60, 110, 'prec', True), crash=int(sum(r.get('crashed', 0) for r in h)), F1tail=float(m.get('F1', nan)), sec=float(m.get('sec', nan)),
                learn_ok=float(np.mean([r.get('loss', 0) > 0 for r in h[10:]])) if len(h) > 10 else nan)

POOLS_SEEN = {}
def pool_check(*prefixes):
    """09-15 PRECHAIN P2: 한 판정에 쓰는 런들의 풀이 하나인지 확인. 섞였으면 종료코드 4(재시도로 못 고침 → 사슬 중단)."""
    ps = set().union(*(POOLS_SEEN.get(p, set()) for p in prefixes)) - {'?'}
    if len(ps) > 1:
        msg = f'풀 혼합 {sorted(ps)} ({prefixes}) — 판정 중단'; print(msg, flush=True); open(f'{C}/DECISIONS.md', 'a').write(f'\n- ⚠ {msg}\n'); raise SystemExit(4)
    return sorted(ps)[0] if ps else '?'
def load(prefix, d=None):
    """{(cell, learner, seed): metrics} — prefix 파일들(w*.json) 병합."""
    res = {}
    for f in sorted(glob.glob(f'{d or C}/{prefix}*_w*.json')):
        try: dd = json.load(open(f))
        except Exception as e: print('  읽기 실패', f, e); continue
        for nm, m in dd.items():
            if 'hist' not in m: continue
            key = (m['cell'], m['learner'], int(m['seed'])) if 'cell' in m else None
            if key is None:                                     # v30/v33 형식: cell_learner_sSEED
                parts = nm.rsplit('_s', 1); cl = parts[0].rsplit('_', 1); key = (cl[0], cl[1], int(parts[1]))
            res[key] = metrics(m); POOLS_SEEN.setdefault(prefix, set()).add(m.get('pool', '?'))
    return res

def paired(fn, seeds):
    """fn(seed) → 차이 또는 None. 반환 n, 평균, t, 양수 개수, 원자료."""
    d = [x for x in (fn(s) for s in seeds) if x is not None and not np.isnan(x)]
    n = len(d); mu = float(np.mean(d)) if n else nan; sd = float(np.std(d, ddof=1)) if n > 1 else nan
    t = float(mu / (sd / np.sqrt(n))) if n > 1 and sd > 0 else nan
    return dict(n=n, mean=mu, t=t, pos=int(sum(x > 0 for x in d)), neg=int(sum(x < 0 for x in d)), d=[round(x, 4) for x in d])
def diff(res, a, b, key):
    def f(s):
        x, y = res.get((a[0], a[1], s)), res.get((b[0], b[1], s))
        return None if x is None or y is None else x[key] - y[key]
    return f
def did(res, a1, b1, a2, b2, key):
    f1, f2 = diff(res, a1, b1, key), diff(res, a2, b2, key)
    def f(s):
        u, v = f1(s), f2(s)
        return None if u is None or v is None else u - v
    return f
def fmt(p): return f"{p['mean']:+.4f} (t {p['t']:+.2f}, +{p['pos']}/−{p['neg']}, n={p['n']})"
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}   # 양측 .05, df = n−1
def sig(p, need_n): return p['n'] >= need_n and p['n'] >= 2 and p['mean'] > 0 and not np.isnan(p['t']) and p['t'] >= TCRIT.get(p['n'] - 1, 2.262)
TOST_M = 3.0   # 09-15 등록: UKF*−EKF* 블록 보상 동등 여유 (v30 blk50k SW−Adam 블록 보상 격차 +6.8 의 절반 이하)
TCRIT1 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}   # 단측 .05
def need(res, keys, tag):
    """리뷰 수정(09-15): 원자료가 불완전하면 판정 JSON 을 쓰지 않고 종료코드 3 → 구동기가 해당 단계 워커를 재착수(누락 런만)."""
    miss = sorted({k for k in keys if k not in res})
    if miss:
        msg = f'{tag}: 원자료 불완전 {len(miss)}건 누락 {miss[:8]} — 판정 JSON 미작성'
        print(msg, flush=True); open(f'{C}/DECISIONS.md', 'a').write(f'\n- ⚠ {msg}\n'); raise SystemExit(3)
def mean_of(res, cell, lrn, key, seeds):
    v = [res[(cell, lrn, s)][key] for s in seeds if (cell, lrn, s) in res]
    return float(np.nanmean(v)) if v else nan
def log_md(txt):
    open(f'{C}/DECISIONS.md', 'a').write(txt + '\n'); print(txt)

# ---------------- 판정 (사전등록) ----------------
def decide_D0():
    """v33 패널티 k. DIRECTION_0915 §7.3 의 M1(블록 FP)·M2(블록 공격 F1)·M5(후기 F1) 규칙을 채택 판정용으로 축약.
       k 채택 조건(모두): ① M1 '확대' = ΔG_FP(= (SW−Adam)_k − (SW−Adam)_1) < 0 ∧ 짝 t ≤ −2.5 ∧ 5/5 음수
                         ② M2 '축소' 아님 = ¬(ΔG_F1blk < 0 ∧ t ≤ −2.5 ∧ 5/5)
                         ③ SW 후기 F1(k) ≥ SW 후기 F1(k=1) − 0.015 (과제 붕괴 없음)
                         ④ k=4 는 추가로 mix50k_k4 후기 F1 |SW−Adam| ≤ 0.015 (정상 동률 유지)
       둘 다 통과하면 |평균 ΔG_FP| 가 큰 쪽. 아니면 k=1. v33 는 하한 0.05 무대라 판정은 '벌점 크기 방향'만 이월한다."""
    r33 = load('v33', V33); r30 = load('v30', V30); res = {**r30, **r33}
    need(res, [(c, l, s) for s in TUNE for c in ('blk50k_k2', 'blk50k_k4', 'mix50k_k4', 'blk50k') for l in ('SW', 'Adam')], 'D0')
    S = TUNE; out = dict(k=1, rule='D0', detail={})
    log_md(f"\n## D0 — v33 패널티 k 판정 ({time.strftime('%m-%d %H:%M')})\n- 원자료: v33 {len(r33)} 런, v30 {len(r30)} 런")
    cand = []
    for k in (2, 4):
        cell = f'blk50k_k{k}'
        m1 = paired(did(res, (cell, 'SW'), (cell, 'Adam'), ('blk50k', 'SW'), ('blk50k', 'Adam'), 'FPblk'), S)
        m2 = paired(did(res, (cell, 'SW'), (cell, 'Adam'), ('blk50k', 'SW'), ('blk50k', 'Adam'), 'F1blk'), S)
        swl = paired(diff(res, (cell, 'SW'), ('blk50k', 'SW'), 'F1tail'), S)   # §7.3 M5 정의(요약 F1, 마지막 공격 에피 20개)
        c1 = m1['n'] == 5 and m1['mean'] < 0 and m1['t'] <= -2.5 and m1['neg'] == 5
        c2 = not (m2['n'] == 5 and m2['mean'] < 0 and m2['t'] <= -2.5 and m2['neg'] == 5)
        c3 = swl['n'] >= 4 and swl['mean'] >= -0.015
        c4 = True
        if k == 4:
            mx = paired(diff(res, ('mix50k_k4', 'SW'), ('mix50k_k4', 'Adam'), 'F1tail'), S)
            c4 = mx['n'] >= 4 and abs(mx['mean']) <= 0.015
            log_md(f"- mix50k_k4 후기 F1 SW−Adam {fmt(mx)} → ④ {c4}")
        log_md(f"- k={k}: M1 ΔG_FP {fmt(m1)} ① {c1} | M2 ΔG_F1blk {fmt(m2)} ② {c2} | SW 후기 F1 k−1 {fmt(swl)} ③ {c3}")
        out['detail'][k] = dict(M1=m1, M2=m2, SWlate=swl, pass_=bool(c1 and c2 and c3 and c4))
        if c1 and c2 and c3 and c4: cand.append((abs(m1['mean']), k))
    if cand: out['k'] = max(cand)[1]
    log_md(f"- **판정 k* = {out['k']}**" + ("" if cand else " (채택 조건 통과 셀 없음 → 기본 벌점 유지)"))
    json.dump(out, open(f'{C}/D0.json', 'w'), indent=1, default=float)

def decide_D1():
    """시야 셀. 1차 = 블록 보상 격차 G_c(s) = Rblk(SW) − Rblk(Adam3e-4).
       후보 가드(모두): (a) G_c 평균 > 0 ∧ 짝 t ≥ 2.0 ∧ 양수 ≥ 4/5
                        (b) SW 후기 F1(c) ≥ SW 후기 F1(g90n3) − 0.02  (과제 붕괴 없음)
                        (c) SW 후기 F1(c) ≥ Adam3e-4 후기 F1(c) − 0.02  (정상 구간에서 SW 열위 아님)
       선택: 가드 통과 비기준 셀 중 DiD_c = G_c − G_base 의 짝 평균 > 0 ∧ t ≥ 1.5 인 셀에서 평균 DiD 최대. 없으면 기준 g90n3.
       (선별 단계라 t 1.5 로 관대. 선택 편향은 S3 새 시드 확인으로 처리)"""
    k = dec(0)['k']; res = load('S1'); S = TUNE; out = dict(k=k, cell=HZBASE, rule='D1', table={})
    need(res, [(j[5], j[6], j[3]) for js in ('gpu', 'cpu') for j in build('S1', js)], 'D1')
    out['pool'] = pool_check('S1')
    log_md(f"\n## D1 — 시야 셀 판정 ({time.strftime('%m-%d %H:%M')}), k*={k}, S1 {len(res)} 런")
    log_md("| 셀 | SW Rblk | Adam3e-4 Rblk | G(SW−Adam) | FPblk G | FPent G | Rpre G | SW F1late | Adam F1late | SW−Adam* 최선lr Rblk | DiD vs 기준 | 가드 |")
    log_md("|---|---|---|---|---|---|---|---|---|---|---|---|")
    swbase_late = mean_of(res, HZBASE, 'SW', 'F1late', S); best = None
    for c in HZ:
        G = paired(diff(res, (c, 'SW'), (c, 'Adam3e-4'), 'Rblk'), S)
        Gfp = paired(diff(res, (c, 'SW'), (c, 'Adam3e-4'), 'FPblk'), S)
        Gfe = paired(diff(res, (c, 'SW'), (c, 'Adam3e-4'), 'FPent'), S)
        Gpre = paired(diff(res, (c, 'SW'), (c, 'Adam3e-4'), 'Rpre'), S)
        alr = {a: mean_of(res, c, a, 'Rblk', S) for a in ADAM_LRS}; abest = max(alr, key=lambda a: -1e9 if np.isnan(alr[a]) else alr[a])
        Gb = paired(diff(res, (c, 'SW'), (c, abest), 'Rblk'), S)
        D = paired(did(res, (c, 'SW'), (c, 'Adam3e-4'), (HZBASE, 'SW'), (HZBASE, 'Adam3e-4'), 'Rblk'), S) if c != HZBASE else None
        swl, adl = mean_of(res, c, 'SW', 'F1late', S), mean_of(res, c, 'Adam3e-4', 'F1late', S)
        ga = G['n'] >= 4 and G['mean'] > 0 and G['t'] >= 2.0 and G['pos'] >= 4
        gb = swl >= swbase_late - 0.02; gc = swl >= adl - 0.02
        ok = bool(ga and gb and gc); out['table'][c] = dict(G=G, Gfp=Gfp, Gfe=Gfe, Gpre=Gpre, Gbest=Gb, adam_best=abest, DiD=D, SWlate=swl, ADlate=adl, guard=[bool(ga), bool(gb), bool(gc)])
        log_md(f"| {c} | {mean_of(res, c, 'SW', 'Rblk', S):.2f} | {mean_of(res, c, 'Adam3e-4', 'Rblk', S):.2f} | {fmt(G)} | {fmt(Gfp)} | {fmt(Gfe)} | {fmt(Gpre)} | {swl:.3f} | {adl:.3f} | {abest} {fmt(Gb)} | {fmt(D) if D else '기준'} | {'통과' if ok else 'a' * (not ga) + 'b' * (not gb) + 'c' * (not gc)} |")
        if c != HZBASE and ok and D['n'] >= 4 and D['mean'] > 0 and D['t'] >= 1.5:
            if best is None or D['mean'] > best[0]: best = (D['mean'], c)
    if best: out['cell'] = best[1]
    out['base_guard_pass'] = bool(all(out['table'][HZBASE]['guard']))
    log_md(f"- **판정 c* = {out['cell']}** (γ {HZ[out['cell']][0]}, n-step {'1' if HZ[out['cell']][1] else '3'})" + ("" if best else " — 기준 대비 DiD 가드 통과 셀 없음 → 기준 유지, 사슬은 기준 무대에서 튜닝·확인으로 진행"))
    json.dump(out, open(f'{C}/D1.json', 'w'), indent=1, default=float)

def decide_D2():
    """학습기 선택(튜닝 시드 42–46, c* 무대). SW 기준 = S1 의 c* SW.
       SW*: 변형 v 가 (i) SW 후기 F1(v) ≥ SW 후기 F1(기준) − 0.02 ∧ (ii) DiD = [Rblk(v) − Rblk(SW기준)] 짝 평균 > 0 ∧ t ≥ 1.5 이면 후보, 평균 최대 채택. 없으면 SW 기준.
       UKF*/EKF*: P₀ 0.03 vs 0.1 중 블록 보상 평균이 큰 쪽(learn_ok 평균 < 0.5 인 설정은 제외 — 무학습 방지). 둘 다 무학습이면 0.03.
       Adam*: {3e-4, 1e-3, 3e-3 (+β 스케일 변형)} 중 블록 보상 평균 최대 — 강건성 행 전용. 1차 비교군은 Adam3e-4 고정."""
    d1 = dec(1); c = d1['cell']; S = TUNE
    res = {**{kk: v for kk, v in load('S1').items() if kk[0] == c}, **load('S2')}
    need(res, [(j[5], j[6], j[3]) for js in ('gpu', 'cpu') for j in build('S2', js)] + [(c, l, s) for s in S for l in ('SW', 'Adam3e-4')], 'D2')
    out_pool = pool_check('S1', 'S2')
    out = dict(rule='D2', cell=c, k=d1['k']); out['pool'] = out_pool
    log_md(f"\n## D2 — 학습기 선택 ({time.strftime('%m-%d %H:%M')}), c*={c}")
    base_late = mean_of(res, c, 'SW', 'F1late', S); best = None
    log_md("| 학습기 | Rblk 평균 | FPblk 평균 | F1blk | F1late | learn_ok | vs SW기준 Rblk DiD | vs Adam3e-4 Rblk |")
    log_md("|---|---|---|---|---|---|---|---|")
    allL = sorted({kk[1] for kk in res if kk[0] == c})
    for l in allL:
        vsb = paired(diff(res, (c, l), (c, 'SW'), 'Rblk'), S) if l != 'SW' else None
        vsa = paired(diff(res, (c, l), (c, 'Adam3e-4'), 'Rblk'), S) if l != 'Adam3e-4' else None
        log_md(f"| {l} | {mean_of(res, c, l, 'Rblk', S):.2f} | {mean_of(res, c, l, 'FPblk', S):.4f} | {mean_of(res, c, l, 'F1blk', S):.3f} | {mean_of(res, c, l, 'F1late', S):.3f} | {mean_of(res, c, l, 'learn_ok', S):.2f} | {fmt(vsb) if vsb else '—'} | {fmt(vsa) if vsa else '—'} |")
        if l.startswith('SW') and l != 'SW' and vsb and vsb['n'] >= 4 and vsb['mean'] > 0 and vsb['t'] >= 1.5 and mean_of(res, c, l, 'F1late', S) >= base_late - 0.02:
            if best is None or vsb['mean'] > best[0]: best = (vsb['mean'], l)
    out['SW'] = best[1] if best else 'SW'
    for fam in ('UKF', 'EKF'):
        opts = [(mean_of(res, c, f'{fam}{p}', 'Rblk', S), f'{fam}{p}') for p in ('p03', 'p1') if mean_of(res, c, f'{fam}{p}', 'learn_ok', S) >= 0.5]
        opts = [o for o in opts if not np.isnan(o[0])]
        out[fam] = max(opts)[1] if opts else f'{fam}p03'
    aopts = [(mean_of(res, c, a, 'Rblk', S), a) for a in allL if a.startswith('Adam')]
    aopts = [o for o in aopts if not np.isnan(o[0])]; out['Adam'] = max(aopts)[1] if aopts else 'Adam3e-4'
    lad = sorted([(mean_of(res, c, out[x], 'Rblk', S), x) for x in ('SW', 'UKF', 'EKF')] + [(mean_of(res, c, 'Adam3e-4', 'Rblk', S), 'Adam3e-4')], reverse=True)
    out['ladder_tune'] = [(x, float(v)) for v, x in lad]
    log_md(f"- **판정** SW*={out['SW']} · UKF*={out['UKF']} · EKF*={out['EKF']} · Adam*(강건성)={out['Adam']}")
    log_md(f"- 튜닝 시드 사다리(블록 보상): " + ' > '.join(f"{x} {v:.2f}" for x, v in out['ladder_tune']))
    json.dump(out, open(f'{C}/D2.json', 'w'), indent=1, default=float)

def decide_D3():
    """최종(새 시드). H1 1차: 블록 보상 SW* − Adam3e-4 짝(n=10) t ≥ 2.26(df 9, 양측 .05) ∧ 평균 > 0 → '확인'.
       H2 사다리: SW*>UKF*, UKF*>EKF*, EKF*>Adam3e-4 각 짝 평균·t (t ≥ 2.26 이면 '유의', 평균 > 0 이면 '방향 일치').
       H3 기전 대조: DiD = G(blk) − G(하한 0.20) > 0 (G = SW*−Adam3e-4 블록 보상), 짝 t 보고.
       H4 시야 레버(c*≠기준): DiD = G(blk c*) − G(blk 기준) 시드 101–105.
       H5 정상 동률: mix c* 후기 F1 |SW* − Adam3e-4| ≤ 0.015 이면 '동률'.
       보조: FPblk·FPent·F1blk·F1late·RECblk 짝, SW*−Adam*."""
    d1, d2 = dec(1), dec(2); c = d1['cell']; res = load('S3'); SWs, UKs, EKs, ADs = d2['SW'], d2['UKF'], d2['EKF'], d2['Adam']
    B, CT, BB, MX = f'blk_{c}', f'ctl020_{c}', f'blk_{HZBASE}', f'mix_{c}'; AP = adam_primary(SWs)
    need(res, [(j[5], j[6], j[3]) for js in ('gpu', 'cpu') for j in build('S3', js) if j[5] == B], 'D3')
    pl = pool_check('S3')
    if d1.get('pool', '?') not in ('?', pl): print('풀 불일치 S1', d1.get('pool'), 'S3', pl); raise SystemExit(4)
    part = sorted({j[5] for js in ('gpu', 'cpu') for j in build('S3', js) if (j[5], j[6], j[3]) not in res})
    L = [f"# chain_hz 최종 보고 ({time.strftime('%m-%d %H:%M')})", "",
         f"- 사슬 판정: k*={d1['k']} → c*={c} (γ {HZ[c][0]}, n-step {'1' if HZ[c][1] else '3'}) → SW*={SWs}, UKF*={UKs}, EKF*={EKs}, Adam*={ADs}",
         f"- S3 런 {len(res)}개, 평가 시드 101–110 (튜닝 시드 42–46 과 분리)" + (f" — ⚠ 불완전 셀: {part}" if part else ""), "", f"- 풀 {pl} · 1차 Adam 비교군 {AP}", "", "## H1 1차 가설 (블록 보상 SW* − 1차 Adam 비교군)"]
    h1 = paired(diff(res, (B, SWs), (B, AP), 'Rblk'), EVAL); ok1 = sig(h1, 10)
    L += [f"- {fmt(h1)} → **{'확인' if ok1 else ('데이터 부족' if h1['n'] < 10 else '미확인')}**"]
    if ADs != AP:   # 09-15 PRECHAIN C②: 격하 규칙
        hA = paired(diff(res, (B, SWs), (B, ADs), 'Rblk'), EVAL)
        L.append(f"- SW*−Adam*({ADs}) {fmt(hA)}" + (" → ⚠ 격하 규칙: 튜닝 Adam 대비 유의하지 않으므로 헤드라인은 '기본 lr Adam 대비'로 한정" if ok1 and not sig(hA, 10) else ""))
    L += ["", "## 보조 지표 (SW* − 비교군, 블록 무대)", f"| 지표 | vs {AP}(1차) | vs Adam* | vs UKF* | vs EKF* |", "|---|---|---|---|---|"]
    for key in ('Rblk', 'FPblk', 'FPent', 'F1blk', 'RECblk', 'F1late', 'Rpre'):
        row = [fmt(paired(diff(res, (B, SWs), (B, o), key), EVAL)) for o in (AP, ADs, UKs, EKs)]
        L.append(f"| {key} | " + ' | '.join(row) + ' |')
    L += ["", "## H2 사다리 (블록 보상, 인접 짝)"]
    for a, b in ((SWs, UKs), (UKs, EKs), (EKs, AP), (UKs, AP)):
        p = paired(diff(res, (B, a), (B, b), 'Rblk'), EVAL)
        L.append(f"- {a} − {b}: {fmt(p)} → {'데이터 부족' if p['n'] < 10 else ('유의' if sig(p, 10) else ('방향 일치' if p['mean'] > 0 else '역전'))}")
    L.append("- 평균: " + ' / '.join(f"{x} {mean_of(res, B, x, 'Rblk', EVAL):.2f}" for x in (SWs, UKs, EKs, AP, ADs)))
    pu = paired(diff(res, (B, UKs), (B, EKs), 'Rblk'), EVAL)   # 09-15 PRECHAIN C①: UKF*≈EKF* TOST, 사다리 '확인' 문장 금지
    if pu['n'] >= 2 and len(pu['d']) >= 2 and np.std(pu['d'], ddof=1) > 0:
        se = np.std(pu['d'], ddof=1) / np.sqrt(len(pu['d'])); tl = (pu['mean'] + TOST_M) / se; th = (pu['mean'] - TOST_M) / se; cr = TCRIT1.get(len(pu['d']) - 1, 1.833)
        L.append(f"- UKF*−EKF* 동등성 TOST(여유 ±{TOST_M} 보상, 09-15 사전등록): t_lo {tl:+.2f} · t_hi {th:+.2f} · 임계 {cr} → {'동등' if (tl >= cr and th <= -cr) else '동등 입증 안 됨'}. 사다리 '확인' 문장은 쓰지 않는다")
    L += ["", "## H3 기전 대조 (하한 0.20 에서 격차 축소?)"]
    h3 = paired(did(res, (B, SWs), (B, AP), (CT, SWs), (CT, AP), 'Rblk'), EVAL)
    g3 = paired(diff(res, (CT, SWs), (CT, AP), 'Rblk'), EVAL)
    L += [f"- 하한 0.20 격차 G {fmt(g3)}", f"- DiD G(0.10) − G(0.20) {fmt(h3)} → {'데이터 부족' if h3['n'] < 10 else ('축소 지지' if sig(h3, 10) else '미지지/경계')}"]
    if c != HZBASE:
        h4 = paired(did(res, (B, SWs), (B, AP), (BB, SWs), (BB, AP), 'Rblk'), EVAL5)
        L += ["", "## H4 시야 레버 (새 시드 101–105)", f"- 기준 셀 G {fmt(paired(diff(res, (BB, SWs), (BB, AP), 'Rblk'), EVAL5))}",
              f"- DiD G(c*, SW*) − G(기준, SW*) {fmt(h4)} → {'데이터 부족' if h4['n'] < 5 else ('레버 지지' if sig(h4, 5) else '미지지/경계')}"]
    h5 = paired(diff(res, (MX, SWs), (MX, AP), 'F1late'), EVAL5)
    L += ["", "## H5 정상 혼합 동률", f"- mix 후기 F1 SW*−Adam3e-4 {fmt(h5)} → {'데이터 부족' if h5['n'] < 5 else ('동률' if abs(h5['mean']) <= 0.015 else '차이')}"]
    L += ["", "## 무학습·추락 점검", "| 셀 | 학습기 | learn_ok 평균 | 추락 합 |", "|---|---|---|---|"]
    for (cell, l) in sorted({(kk[0], kk[1]) for kk in res}):
        v = [res[(cell, l, s)] for s in EVAL if (cell, l, s) in res]
        L.append(f"| {cell} | {l} | {np.nanmean([x['learn_ok'] for x in v]):.2f} | {sum(x['crash'] for x in v)} |")
    open(f'{C}/REPORT.md', 'w').write('\n'.join(L) + '\n'); log_md(f"\n## D3 — 최종 보고 작성: {C}/REPORT.md (H1 {'확인' if ok1 else '미확인'})")

def decide_D4():
    """배치 64 절제. H6(1차): G = Rblk(SW*) − Rblk(Adam3e-4) 에 대해 DiD = G_b64 − G_b128 (b64 쪽 SW* 는 창 표본 B·N 이 같은 N=2·N*),
       sig(n=10) 이면 '배치 64 에서 격차 확대', 음의 방향으로 유의하면 '축소', 그 외 '차이 없음/경계'. 절제이므로 사슬 판정(D1–D3)을 바꾸지 않는다.
       보조: 학습기별 b64 − b128 짝(같은 학습기, 같은 시드), b64 사다리, b64 안 SW 창 N 비교, 런당 시간."""
    d1, d2 = dec(1), dec(2); c = d1['cell']; SWs, UKs, EKs, ADs = d2['SW'], d2['UKF'], d2['EKF'], d2['Adam']
    res = {**load('S3'), **load('S4')}; B, B4 = f'blk_{c}', f'b64_{c}'; AP = adam_primary(SWs); pool_check('S3', 'S4')
    need(res, [(j[5], j[6], j[3]) for js in ('gpu', 'cpu') for j in build('S4', js)] + [(B, l, s) for s in EVAL for l in (SWs, AP)], 'D4')
    nstar = int(learner_env(SWs, c=c)['RHUKF_N']); SWeq = f'{SWs}-N{2 * nstar}'; ns = sorted({10, 14, 2 * nstar})
    L = [f"# 배치 64 절제 보고 ({time.strftime('%m-%d %H:%M')})", "",
         f"- 무대 c*={c}, 벌점 k*={d1['k']}. SW*={SWs}(N {nstar}) ↔ 배치 64 창 표본 동일 짝 {SWeq}. 시드 101–110, 배치 128 쪽은 S3 {B} 재사용", "",
         "## H6 1차 — SW*−Adam3e-4 블록 보상 격차의 배치 DiD"]
    g128 = paired(diff(res, (B, SWs), (B, AP), 'Rblk'), EVAL); g64 = paired(diff(res, (B4, SWeq), (B4, AP), 'Rblk'), EVAL)
    h6 = paired(did(res, (B4, SWeq), (B4, AP), (B, SWs), (B, AP), 'Rblk'), EVAL)
    neg = dict(h6, mean=-h6['mean'], t=-h6['t'] if not np.isnan(h6['t']) else nan)
    v6 = '데이터 부족' if h6['n'] < 10 else ('배치 64 에서 격차 확대' if sig(h6, 10) else ('배치 64 에서 격차 축소' if sig(neg, 10) else '차이 없음/경계'))
    L += [f"- G(배치 128) {fmt(g128)}", f"- G(배치 64, {SWeq}) {fmt(g64)}", f"- DiD {fmt(h6)} → **{v6}**", "",
          "## 학습기별 배치 64 − 128 (같은 학습기·시드 짝)", "| 학습기 | Rblk | FPblk | FPent | F1blk | RECblk | F1late | 런당 시간 128 → 64 (s) |", "|---|---|---|---|---|---|---|---|"]
    for l128, l64 in [(SWs, SWeq), (UKs, UKs), (EKs, EKs), (AP, AP)] + ([(ADs, ADs)] if ADs != AP else []):
        row = [fmt(paired(diff(res, (B4, l64), (B, l128), key), EVAL)) for key in ('Rblk', 'FPblk', 'FPent', 'F1blk', 'RECblk', 'F1late')]
        L.append(f"| {l128} → {l64} | " + ' | '.join(row) + f" | {mean_of(res, B, l128, 'sec', EVAL):.0f} → {mean_of(res, B4, l64, 'sec', EVAL):.0f} |")
    L += ["", "## 배치 64 안 비교", "| 학습기 | Rblk 평균 | FPblk | F1late | vs Adam3e-4 Rblk |", "|---|---|---|---|---|"]
    for l in [f'{SWs}-N{n}' for n in ns] + [UKs, EKs, AP] + ([ADs] if ADs != AP else []):
        vs = fmt(paired(diff(res, (B4, l), (B4, AP), 'Rblk'), EVAL)) if l != AP else '—'
        L.append(f"| {l} | {mean_of(res, B4, l, 'Rblk', EVAL):.2f} | {mean_of(res, B4, l, 'FPblk', EVAL):.4f} | {mean_of(res, B4, l, 'F1late', EVAL):.3f} | {vs} |")
    if len(ns) > 1:
        L.append(f"- 창 N 비교(배치 64): " + ' · '.join(f"N{b}−N{a} {fmt(paired(diff(res, (B4, f'{SWs}-N{b}'), (B4, f'{SWs}-N{a}'), 'Rblk'), EVAL))}" for a, b in zip(ns[:-1], ns[1:])))
    open(f'{C}/REPORT_B64.md', 'w').write('\n'.join(L) + '\n'); log_md(f"\n## D4 — 배치 64 절제 보고: {C}/REPORT_B64.md (H6 {v6})")

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'count':
        st = sys.argv[2]; js = sys.argv[3] if len(sys.argv) > 3 else None
        for j in ([js] if js else ['gpu', 'cpu']):
            jb = build(st, j); print(len(jb) if js else f'{st} {j}: {len(jb)} jobs', end='\n')
            if not js:
                sec = sum((4250 if j_[6].startswith('SW') else 2650) if j_[2] == 'rhukf' else 45 for j_ in jb)
                print(f'   추정 {sec/3600:.1f} 워커·h;', sorted({(x[5], x[6]) for x in jb}))
    elif mode == 'decide':
        {'D0': decide_D0, 'D1': decide_D1, 'D2': decide_D2, 'D3': decide_D3, 'D4': decide_D4}[sys.argv[2]]()
    elif mode == 'work':
        st, WID, NW = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); js = os.environ.get('JOBSET', 'gpu')
        jobs = build(st, js); mine = jobs[WID::NW]; fo = f'{C}/{st}{js}_w{WID}.json'
        try: out = json.load(open(fo))
        except Exception: out = {}
        print(f'[{st} {js} w{WID}] {len(mine)}/{len(jobs)} runs: {[j[0] for j in mine]}', flush=True)
        for job in mine:
            if job[0] in out: continue
            try: out[job[0]] = run(job)
            except Exception as e: import traceback; traceback.print_exc(); print(f'  !! {job[0]} 실패: {e}', flush=True)
            json.dump(out, open(fo + '.tmp', 'w'), default=float); os.replace(fo + '.tmp', fo)
        open(f'{C}/{st}{js}_w{WID}_DONE', 'w').close(); print(f'[{st} {js} w{WID}] 완료', flush=True)
