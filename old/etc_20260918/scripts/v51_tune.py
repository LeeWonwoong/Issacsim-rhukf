#!/usr/bin/env python3
# v51_tune (2026-09-04) — env5.1(주변분포+자기상관 Isaac 정합)에서 3축 튜닝 스윕.
#   축: P(p_init) {0.02, 0.1} × R {1, 2} × alpha {0.1, 0.5} = 8 config
#   고정: absolute · N5 · tau0.005/ui1 · Q1e-3 · [16,16] · seed42 · 160ep · 300스텝
#   참고 FD 반경 h = α√(n_x·P₀), n_x=514:
#     α0.1/P0.02 h=0.32 | α0.1/P0.1 h=0.72 | α0.5/P0.02 h=1.60 | α0.5/P0.1 h=3.59
import sys, os, time, json, math
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V51'] = '1'; os.environ.pop('SURR_V5', None); os.environ.pop('SURR_V4', None)
os.environ['SURR_ATK_PROB'] = '0.70'
OUT = 'results_v51tune'; os.makedirs(OUT, exist_ok=True)
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
from surrogate_run import run_config, summarize
CFG = []
for P in ('0.02', '0.1'):
    for R in ('1', '2'):
        for al in ('0.10', '0.50'):
            CFG.append((f'P{P}_R{R}_a{al}',
                        dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005',
                             RHUKF_UI=1, RHUKF_R=R, RHUKF_FORM='absolute',
                             RHUKF_PINIT=P, RHUKF_ALPHA=al),
                        0.1 * float(al) / 0.1 * math.sqrt(514 * float(P)) * float(al) / float(al)))
CFG = [(n, e, float(e['RHUKF_ALPHA']) * math.sqrt(514 * float(e['RHUKF_PINIT']))) for n, e, _ in CFG]
mine = CFG[WID::2]
print(f'[worker {WID}] {[c[0] for c in mine]}  env5.1', flush=True)
out = {}
for name, env, h in mine:
    for k in ('RHUKF_FORM','RHUKF_PINIT','RHUKF_PD','RHUKF_ALPHA','RHUKF_HUBER_C','OPT',
              'ADAM_LOSS','ADAM_AMSGRAD','ADAM_LR'):
        os.environ.pop(k, None)
    os.environ['SURR_V51']='1'; os.environ['SURR_ATK_PROB']='0.70'
    t0 = time.time()
    try:
        hist, p = run_config(env, obs_mode='raw4', n_ep=160, seed=42, ep_steps=300,
                             agent_type='rhukf', probe=True, obs_noise=0.0)
        s = summarize(hist); s.update(p); s['h_fd'] = h; out[name] = s
        print(f'  {name:18} h={h:5.2f} F1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} fpr={s["fpr"]:.3f} '
              f'pFP={s["probe_fp"]:.3f} wR={s["probe_weak_rec"]:.2f} sR={s["probe_strong_rec"]:.2f} '
              f'conv={s["conv_ep"]:.0f} stab={s["stab"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'  {name:18} FAIL {type(e).__name__}: {e}', flush=True); out[name] = {'FAIL': str(e), 'h_fd': h}
    json.dump(out, open(f'{OUT}/tune_w{WID}.json', 'w'), default=float)
open(f'{OUT}/TUNE_W{WID}_DONE', 'w').close(); print(f'[worker {WID}] done', flush=True)
