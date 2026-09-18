#!/usr/bin/env python3
"""cp_regime_scan (2026-09-18, 사용자 확정) — CartPole/LL 레짐 이식: "reward 상승 ∧ loss 하락" 동시 성립 셀 찾기.

고정 (전 셀 공통)
  γ 0.97 · n-step 3 · 확약 없음(HOVER_DWELL=0) · 바람 티어 {0,6} 균등(풀 v5e 에 연속 ws 없음 → 2점 근사)
  관측 [0,1]: SURR_CLIP=4 · SURR_OBS_CLIP=4 · SURR_OBS_DIV=4 (셋 다 필수 — 하나라도 빠지면 3.0 에서 조용히 포화, 09-16 이중클립 버그)
  보상: FN 온셋가중 끔(FN_ONSET_MULT=1), 추락 벌점 없음(절단만). 단계 lin = FN 선형(−0.4−0.35·d, d≤5) → 단계 const = FN 상수 −0.4(FN_PER=0)
  SWIRL: error-state · anchor=target · h=0 SPAS / h≥1 online_moving(θ_anchor+μΔ_prev) · pΔ 0.01 · huber c 5 · N 5 · q 1e-3 · α 0.1 · 망 [16,16]
  Adam : lr 3e-4 · Huber β 1 · AMSGrad off · He init · τ 0.005 / ui 1 (표준)
격자
  보상 배율 c ∈ {0.2, 0.4, 0.6, 1.0} × R ∈ {0.25, 0.5, 1, 2, 4} × τ/ui ∈ {0.005/1, 0.02/4}  (SWIRL 40런/단계)
  Adam 참조: c 4개 (CPU)
  시드 42, 200 에피소드, 300 스텝
판정 (사전 고정 — 결과 보기 전)
  L1 loss 초반최고점(ep0–59, 5판 이동평균)/loss(ep150–199) ≥ 2      L2 Spearman(ep, loss | 최고점→199) < −0.3
     (09-18 착수 전 수정: 스모크에서 Q 상승기 동안 loss 가 먼저 오름 → ep2–11 창은 상승기를 잡음. ep2–11 비는 L_ratio_e 로 병기)
  L3 loss(175–199) ≤ 1.2·loss(125–149)
  R1 reward(150–199) > reward(0–9) ∧ Spearman(ep, reward) > +0.3      R2 F1(150–199, 공격 에피) ≥ 0.80 ∧ FPR(150–199) ≤ 0.05
  Q  Qmax(150–199) ≤ 3·V_exp,  V_exp = c·0.5/(1−γ)
사용: jobs | work STAGESET WID NW (env JOBSET=gpu|cpu) | job NAME | judge
"""
import sys, os, json, time, subprocess, glob
import numpy as np
ROOT = '/home/acsl/projects/Issacsim-rhukf'
OUT = os.environ.get('CP_DIR', f'{ROOT}/results/claudecodefortest/cp_regime')
POOL = f'{ROOT}/results/claudecodefortest/night/train_pool_v5e.npz'
NEP = int(os.environ.get('CP_NEP', '200')); SEED = 42; GAMMA = 0.97

ENV0 = {'SURR_V7': '1', 'SURR_V8': '1', 'SURR_V5': '1', 'SURR_POOL': POOL, 'SURR_ATK_PROB': '0.5',
        'SURR_PLOC_TIERS': '7:0.70,0.2,0.76,0.3,0.80,0.9,0.84,1;10:0.70,0.3,0.76,0.7,0.80,0.8,0.84,1', 'SURR_DEAD_TIERS': '10:0.78,26,0.80,16,0.84,4',
        'ATK_DELTA_LO': '0.10', 'ATK_SPLIT': '0.72', 'ATK_DELTA_HI': '0.84', 'ATK_START_LO': '60', 'ATK_START_HI': '200',
        'ATK_ON_LO': '25', 'ATK_ON_HI': '40', 'ATK_P_UPPER': '0.5',
        'SURR_DEAD_MODE': 'curve', 'SURR_PLATEAU_CURVE': '1', 'SURR_EP_STEPS': '300', 'SURR_EP_RNG': '1',
        'SURR_TIER_P': '0:0.5,6:0.5',
        'SURR_CLIP': '4.0', 'SURR_OBS_CLIP': '4.0', 'SURR_OBS_DIV': '4.0',          # ★ 관측 [0,1] — 셋 다 필수
        'GAMMA': str(GAMMA), 'EPS_DECAY': '4000', 'EPS_HOVER_P': '0.1', 'HOVER_DWELL': '0',
        'TERMINAL_PEN': '0', 'FN_ONSET_MULT': '1', 'BUFFER_SIZE': '50000', 'REPLAY_MODE': 'uniform',
        'PROBE_EVERY': '1', 'PROBE_N': '4', 'NET_HIDDEN': '16'}
SW = dict(RHUKF_FORM='error', RHUKF_PD='0.01', RHUKF_HUBER_C='5', RHUKF_N='5', RHUKF_Q='1e-3', RHUKF_ALPHA='0.1')
ADAM = dict(ADAM_LR='3e-4', ADAM_AMSGRAD='0', ADAM_INIT='he', ADAM_HUBER_BETA='1')
CS = [0.2, 0.4, 0.6, 1.0]; RS = [0.25, 0.5, 1.0, 2.0, 4.0]; TU = {'cp': ('0.005', '1'), 'll': ('0.02', '4')}
STAGES = {'lin': {}, 'const': {'FN_PER': '0'}}
# 이전 실험 env 가 새지 않도록 비울 키
CLR = ('RHUKF_PINIT', 'RHUKF_SPAS', 'RHUKF_MODE', 'RHUKF_UPD', 'RHUKF_TAU', 'RHUKF_UI', 'RHUKF_R', 'FN_PER', 'FN_BASE', 'R_TP', 'R_TN', 'R_FP',
       'REWARD_MODE', 'SURR_TIER_SCHED', 'SURR_FAM_SCHED', 'SURR_GUST_P', 'SURR_SHIFT_EP', 'SURR_OBS_CENTER', 'SURR_OBS_NOISE', 'ATK_BURSTS',
       'NSTEP_OFF', 'BATCH_SIZE', 'OPT', 'ADAM_LR', 'ADAM_AMSGRAD', 'ADAM_INIT', 'ADAM_HUBER_BETA', 'SCENARIO_SEED', 'OBS_MODE', 'REWARD_SCALE')


def jobs():
    """(name, jobset, agent, env) — 단계 lin 전부가 const 보다 먼저 온다."""
    J = []
    for st, se in STAGES.items():
        for tu, (tau, ui) in TU.items():
            for c in CS:
                for R in RS:
                    J.append((f'{st}_SW_{tu}_c{c:g}_R{R:g}', 'gpu', 'rhukf',
                              dict(se, **SW, REWARD_SCALE=f'{c:g}', RHUKF_R=f'{R:g}', RHUKF_TAU=tau, RHUKF_UI=ui)))
        for c in CS:
            J.append((f'{st}_Adam_c{c:g}', 'cpu', 'adam', dict(se, **ADAM, REWARD_SCALE=f'{c:g}')))
    return J


def run_job(name):
    j = {x[0]: x for x in jobs()}[name]
    for k in CLR: os.environ.pop(k, None)
    os.environ.update(ENV0); os.environ.update(j[3])
    assert all(os.environ.get(k) == '4.0' for k in ('SURR_CLIP', 'SURR_OBS_CLIP', 'SURR_OBS_DIV')), '관측 클립/정규화 3종 누락'
    sys.path[:0] = [ROOT, f'{ROOT}/etc/scripts']; os.chdir(ROOT)
    import surrogate_run as SR, surrogate_env8 as SE
    assert SE.SurrogateEnv.CLIP == 4.0, f'SurrogateEnv.CLIP={SE.SurrogateEnv.CLIP}'
    import swrl_config
    cfg = swrl_config.Config()
    print(f'[{name}] γ={cfg.gamma} rscale={cfg.reward_scale} fn_per={cfg.reward.fn_per_step} fn_onset_mult={cfg.reward.fn_onset_mult} '
          f'form={cfg.state_form} anchor={cfg.anchor_type} argmax={cfg.ddqn_argmax}/{cfg.h0_online_moving_init} pΔ={cfg.p_delta_init} '
          f'huber={cfg.huber_c} N={cfg.N_horizon} R={cfg.r_init} q={cfg.q_init} τ={cfg.tau_srrhuif} ui={cfg.update_interval} '
          f'dwell={os.environ.get("HOVER_DWELL")} obs clip/div={os.environ["SURR_OBS_CLIP"]}/{os.environ["SURR_OBS_DIV"]}', flush=True)
    t0 = time.time()
    h = SR.run_config({}, n_ep=NEP, seed=SEED, ep_steps=300, agent_type=j[2])
    rec = dict(name=name, agent=j[2], env=j[3], n_ep=NEP, seed=SEED, sec=time.time() - t0, hist=[dict(r) for r in h])
    tmp = f'{OUT}/{name}.json.tmp'; json.dump(rec, open(tmp, 'w'), default=float); os.replace(tmp, f'{OUT}/{name}.json')
    print(f'[{name}] 완료 {rec["sec"]/60:.1f}분', flush=True)


def work(stageset, wid, nw):
    js = os.environ.get('JOBSET', 'gpu')
    mine = [j for j in jobs() if j[1] == js and (stageset == 'all' or j[0].startswith(stageset + '_'))]
    for i, j in enumerate(mine):
        if i % nw != wid or os.path.exists(f'{OUT}/{j[0]}.json'): continue
        r = subprocess.run([sys.executable, __file__, 'job', j[0]], stdout=open(f'{OUT}/logs/{j[0]}.log', 'w'), stderr=subprocess.STDOUT)
        print(f'{time.strftime("%H:%M")} {j[0]} rc={r.returncode}', flush=True)


# ---------------- 판정 ----------------
def _rank(x): return np.argsort(np.argsort(x)).astype(float)
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5: return np.nan
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1])
def seg(h, k, a, b, atk=False):
    v = [r.get(k) for r in h[a:b] if (not atk or r.get('has_atk'))]
    v = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else np.nan

def gate(rec):
    h = rec['hist']; e = rec['env']; c = float(e['REWARD_SCALE'])
    ep = np.arange(len(h)); L = np.array([r['loss'] for r in h], float); Rw = np.array([r['reward'] for r in h], float)
    L_e, L_l = seg(h, 'loss', 2, 12), seg(h, 'loss', 150, 200)
    # ★ 초반 최고점: Q 가 0 → V_exp 로 오르는 동안 loss 가 먼저 오른다(스모크 실측) → '초반 높다가 감소' = 초반 60판 안 최고점 대비
    Ls = np.convolve(np.nan_to_num(L), np.ones(5) / 5, mode='same'); pk = int(np.argmax(Ls[:60])); L_pk = float(Ls[pk])
    g = dict(name=rec['name'], agent=rec['agent'], stage=rec['name'].split('_')[0], c=c,
             R=float(e.get('RHUKF_R', 'nan')), tu=('ll' if e.get('RHUKF_UI') == '4' else 'cp') if rec['agent'] == 'rhukf' else 'adam',
             L_early=L_e, L_late=L_l, L_ratio_e=L_e / L_l if L_l > 0 else np.nan, L_peak=L_pk, L_peak_ep=pk,
             L_ratio=L_pk / L_l if L_l > 0 else np.nan,
             L_rho=spear(ep[pk:], L[pk:]), L_tail=seg(h, 'loss', 175, 200) / seg(h, 'loss', 125, 150),
             R_early=seg(h, 'reward', 0, 10), R_late=seg(h, 'reward', 150, 200), R_rho=spear(ep, Rw),
             F1_late=seg(h, 'f1', 150, 200, True), FPR_late=seg(h, 'fpr', 150, 200), crash=int(sum(r.get('crashed', 0) for r in h)),
             qmax_late=seg(h, 'qmax', 150, 200), V_exp=c * 0.5 / (1 - GAMMA),
             nisf_early=seg(h, 'nisf', 2, 12), nisf_late=seg(h, 'nisf', 150, 200), kgain_late=seg(h, 'kgain', 150, 200),
             tvar_early=seg(h, 'tvar', 2, 12), adapt_late=seg(h, 'adapt', 150, 200), aflip_late=seg(h, 'aflip', 150, 200),
             min_=rec['sec'] / 60)
    g['c2R'] = c * c / g['R'] if np.isfinite(g['R']) else np.nan
    g['L1'] = bool(g['L_ratio'] >= 2); g['L2'] = bool(g['L_rho'] < -0.3); g['L3'] = bool(g['L_tail'] <= 1.2)
    g['R1'] = bool(g['R_late'] > g['R_early'] and g['R_rho'] > 0.3); g['R2'] = bool(g['F1_late'] >= 0.80 and g['FPR_late'] <= 0.05)
    g['Q'] = bool(np.isfinite(g['qmax_late']) and g['qmax_late'] <= 3 * g['V_exp'])
    g['PASS'] = all(g[k] for k in ('L1', 'L2', 'L3', 'R1', 'R2', 'Q'))
    return g

def judge():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/*_*.json')) if not f.endswith('gates.json')]
    G = [gate(r) for r in recs]
    if not G: print('결과 없음'); return
    json.dump(G, open(f'{OUT}/gates.json', 'w'), indent=1, default=float)
    ok = lambda b: 'O' if b else '·'
    lines = ['# CP 레짐 격자 판정', '', f'갱신 {time.strftime("%m-%d %H:%M")} · 완료 {len(G)}/{len(jobs())}', '',
             '| stage | 학습기 | τ/ui | c | R | c²/R | L 최고점@ep/후 | ρ(L) | L꼬리 | R 초→후 | ρ(R) | F1 | FPR | 추락 | Qmax/V | K | NIS초→후 | huber팽창 | flip | L1 L2 L3 R1 R2 Q | 합격 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g in sorted(G, key=lambda g: (g['stage'], g['agent'], g['tu'], g['c2R'] if np.isfinite(g['c2R']) else g['c'])):
        lines.append(f"| {g['stage']} | {g['agent']} | {g['tu']} | {g['c']:g} | {g['R']:g} | {g['c2R']:.3g} | {g['L_peak']:.3g}@{g['L_peak_ep']}/{g['L_late']:.3g} ({g['L_ratio']:.2f}) | "
                     f"{g['L_rho']:+.2f} | {g['L_tail']:.2f} | {g['R_early']:.1f}→{g['R_late']:.1f} | {g['R_rho']:+.2f} | {g['F1_late']:.3f} | {g['FPR_late']:.3f} | {g['crash']} | "
                     f"{g['qmax_late']/g['V_exp']:.2f} | {g['kgain_late']:.3g} | {g['nisf_early']:.2g}→{g['nisf_late']:.2g} | {g['adapt_late']:.2f} | {g['aflip_late']:.3f} | "
                     f"{ok(g['L1'])} {ok(g['L2'])} {ok(g['L3'])} {ok(g['R1'])} {ok(g['R2'])} {ok(g['Q'])} | {'**PASS**' if g['PASS'] else ''} |")
    open(f'{OUT}/REPORT.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    try: plot(recs, G)
    except Exception as e: print('그림 실패', e)

def plot(recs, G):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    try: matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    except Exception: pass
    sm = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode='valid')
    by = {r['name']: r for r in recs}; gd = {g['name']: g for g in G}
    for st in STAGES:
        for tu in TU:
            fig, ax = plt.subplots(len(CS), len(RS), figsize=(3.2 * len(RS), 2.4 * len(CS)), squeeze=False)
            for i, c in enumerate(CS):
                for k, R in enumerate(RS):
                    a = ax[i][k]; nm = f'{st}_SW_{tu}_c{c:g}_R{R:g}'
                    if nm not in by: a.set_axis_off(); continue
                    h = by[nm]['hist']; L = np.array([r['loss'] for r in h]); Rw = np.array([r['reward'] for r in h])
                    a.plot(sm(L), color='C3', lw=1); a.set_yscale('log'); a.tick_params(labelsize=6)
                    b = a.twinx(); b.plot(sm(Rw), color='C0', lw=1); b.tick_params(labelsize=6)
                    ad = by.get(f'{st}_Adam_c{c:g}')
                    if ad: b.plot(sm(np.array([r['reward'] for r in ad['hist']])), color='C0', lw=0.8, ls=':', alpha=0.7)
                    g = gd[nm]; a.set_title(f"c{c:g} R{R:g} c²/R {g['c2R']:.2g} {'PASS' if g['PASS'] else ''}", fontsize=7,
                                            color='green' if g['PASS'] else 'black')
            fig.suptitle(f'{st} · SWIRL {tu} — 빨강 loss(log, 좌) · 파랑 reward(우) · 점선 Adam reward', fontsize=9)
            fig.tight_layout(); fig.savefig(f'{OUT}/curves_{st}_{tu}.png', dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for g in G:
        if g['agent'] != 'rhukf': continue
        m = 'o' if g['tu'] == 'cp' else 's'; col = 'green' if g['PASS'] else ('C1' if g['L1'] else 'C7')
        ax[0 if g['stage'] == 'lin' else 1].scatter(g['c2R'], g['L_ratio'], marker=m, c=col, s=28)
    for a, st in zip(ax, STAGES):
        a.set_xscale('log'); a.set_yscale('log'); a.axhline(2, ls='--', c='k', lw=0.7); a.set_xlabel('c²/R'); a.set_ylabel('loss 초반최고점/후반'); a.set_title(f'{st} (○ τ0.005/ui1 · □ τ0.02/ui4, 초록=합격)', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/loss_ratio_vs_c2R.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(f'{OUT}/logs', exist_ok=True)
    m = sys.argv[1]
    if m == 'jobs':
        for j in jobs(): print(j[1], j[0])
        print(len(jobs()), 'jobs')
    elif m == 'job': run_job(sys.argv[2])
    elif m == 'work': work(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif m == 'judge': judge()
