#!/usr/bin/env python3
# alpha_sweep (2026-09-03) — σ스프레드 축을 게인 축과 분리하는 유일한 실험.
#   P 스윕은 게인(P0/R)과 FD스텝(h=α√(n_x·P0))을 동시에 움직여 판별 불가.
#   → P0 를 {0.1, 0.2} 로 고정(게인 고정)하고 α 만 {0.03,0.1,0.2,0.3} 이동 = 스프레드 단독 축.
#   나머지는 grid64 승자 고정: tau0.005/ui1(세트) · R1 · Q1e-2 · N5 · err모드 · [16,16].
#   출력: results_alpha/ (repo 내부 — /tmp 소실 재발 방지)
import sys, os, time, json
import numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
from surrogate_run import run_config, summarize

WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NEP = int(os.environ.get('ALPHA_NEP', 160))
SEED = int(os.environ.get('ALPHA_SEED', 42))
OUT = 'results_alpha'
os.makedirs(OUT, exist_ok=True)
N_X = 514   # NET_HIDDEN=16, dimS=12, nA=2

CONFIGS = []
for P0 in ('0.2', '0.1'):
    for al in ('0.30', '0.20', '0.10', '0.03'):
        name = f'a{al}_P{P0}'
        env = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005',
                   RHUKF_UI=1, RHUKF_R='1', RHUKF_PD=P0, RHUKF_ALPHA=al)
        CONFIGS.append((name, env, float(al), float(P0)))

mine = CONFIGS[WID::2]
print(f'[worker {WID}] {len(mine)} configs, {NEP}ep, seed={SEED}', flush=True)
out = {}
for name, env, al, P0 in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT'):   # 이전 config 잔류 방지 (err 모드 고정)
        os.environ.pop(k, None)
    h_fd = al * np.sqrt(N_X * P0)             # 좌표별 FD 스텝
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=NEP, seed=SEED,
                             agent_type='rhukf', probe=True)
        s = summarize(hist); s.update(p); s['h_fd'] = h_fd; s['alpha'] = al; s['P0'] = P0
        out[name] = s
        print(f'  {name:14} h={h_fd:5.3f} F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} rwd={s["reward"]:.1f} '
              f'loss={s["loss"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:14} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'h_fd': h_fd, 'alpha': al, 'P0': P0}
    json.dump(out, open(f'{OUT}/alpha_w{WID}_s{SEED}.json', 'w'), default=float)
open(f'{OUT}/ALPHA_W{WID}_S{SEED}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
