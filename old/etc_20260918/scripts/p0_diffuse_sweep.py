#!/usr/bin/env python3
# p0_diffuse_sweep (2026-09-03) — 승자 config 에서 P0 를 diffuse(무편향 FIR) 쪽으로 확장.
#   P0 ∈ {0.3, 0.4, 0.5} (기존 그리드 상한 0.2 위). 나머지 전부 승자 고정:
#   tau0.005/ui1 · R1 · Q1e-2 · N5 · alpha0.1 · error-state · [16,16].
#   가설: P0↑ = 슬라이딩 윈도우 사용률↑ (R=1 위 고유방향 비율) → FIR 극한에 접근.
#         계속 좋아지면 "무편향 FIR 극한이 최적", 꺾이면 "ridge 가 실제로 필요".
#   ★ alpha 스윕 완료(ALPHA_W{0,1}_S42_DONE) 후 자동 시작.
import sys, os, time, json
import numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
OUT = 'results_alpha'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print('[wait] alpha 스윕 완료 대기...', flush=True)
while not (os.path.exists(f'{OUT}/ALPHA_W0_S42_DONE') and os.path.exists(f'{OUT}/ALPHA_W1_S42_DONE')):
    time.sleep(60)
time.sleep(5)
from surrogate_run import run_config, summarize
N_X = 514
CONFIGS = [(f'a0.10_P{p}', p) for p in ('0.3', '0.4', '0.5')]
mine = CONFIGS[WID::2]
print(f'[worker {WID}] {len(mine)} configs: {[c[0] for c in mine]}', flush=True)
out = {}
for name, P0 in mine:
    for k in ('RHUKF_FORM', 'RHUKF_PINIT'):
        os.environ.pop(k, None)
    env = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-2', RHUKF_TAU='0.005',
               RHUKF_UI=1, RHUKF_R='1', RHUKF_PD=P0, RHUKF_ALPHA='0.10')
    h_fd = 0.10 * np.sqrt(N_X * float(P0))
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, agent_type='rhukf', probe=True)
        s = summarize(hist); s.update(p); s['h_fd'] = h_fd; s['alpha'] = 0.10; s['P0'] = float(P0)
        out[name] = s
        print(f'  {name:14} h={h_fd:5.3f} F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} rwd={s["reward"]:.1f} '
              f'loss={s["loss"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:14} FAIL {type(e).__name__}: {e}', flush=True)
        out[name] = {'FAIL': str(e), 'h_fd': h_fd, 'P0': float(P0)}
    json.dump(out, open(f'{OUT}/p0diff_w{WID}.json', 'w'), default=float)
open(f'{OUT}/P0DIFF_W{WID}_DONE', 'w').close()
print(f'[worker {WID}] done', flush=True)
