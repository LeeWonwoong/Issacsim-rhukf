# γ·버퍼 스캔 — 옵티마이저 격차의 가치학습측 레버 (L0.5 표준형·decay4000)
#   γ {0.5, 0.8, 0.95} × 3옵티 × seed{42,43} = 18런  (basic Adam 3e-4 고정 — 교수 공정성 방침)
#   buffer {20000(현행), 5000} × 3옵티 × seed{42,43} = 12런 (γ=0.85 기본에서)
import sys, os, json, time, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V7'] = '1'; os.environ['SURR_P_LETHAL'] = '0.5'; os.environ['EPS_DECAY'] = '4000'
WID = int(sys.argv[1])
CLR = ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_R','RHUKF_N','RHUKF_HUBER_C',
       'RHUKF_SPAS','RHUKF_Q','RHUKF_TAU','RHUKF_UI','ADAM_LR','OPT','ADAM_LOSS','GAMMA','BUFFER_SIZE')
SW = dict(NET_HIDDEN=16, RHUKF_N=6, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
          RHUKF_FORM='absolute', RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_R='1', RHUKF_PINIT='0.03')
OPTS = [('SWIRL', dict(SW), 'rhukf'), ('Adam', dict(NET_HIDDEN=16, ADAM_LR='3e-4'), 'adam'),
        ('SGD', dict(NET_HIDDEN=16, ADAM_LR='3e-4', OPT='sgd'), 'adam')]
jobs = []
for g in ('0.5', '0.8', '0.95'):
    for nm, e, ag in OPTS:
        for sd in (42, 43):
            jobs.append((f'{nm}_g{g}_s{sd}', e, ag, sd, {'GAMMA': g}))
for bf in ('20000', '5000'):
    for nm, e, ag in OPTS:
        for sd in (42, 43):
            jobs.append((f'{nm}_b{bf}_s{sd}', e, ag, sd, {'BUFFER_SIZE': bf}))
import surrogate_run as SR
import importlib
mine = jobs[WID::2]; out = {}
print(f'[w{WID}] {len(mine)} runs', flush=True)
for name, env, ag, sd, extra in mine:
    for k in CLR: os.environ.pop(k, None)
    for k, v in extra.items(): os.environ[k] = v
    importlib.reload(SR)
    t0 = time.time(); h = SR.run_config(env, n_ep=160, seed=sd, ep_steps=400, agent_type=ag)
    s = SR.summarize(h)
    rw = [r['reward'] for r in h]
    dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    out[name] = dict(F1=float(s['F1']), fpr=float(s['fpr']),
                     delay=float(np.mean(dl)) if dl else -1.0,
                     rwd=float(np.mean(rw[120:])), e40=float(np.mean(rw[20:40])),
                     crash=int(sum(r['crashed'] for r in h)))
    m = out[name]
    print(f"  {name:18s} F1={m['F1']:.3f} fpr={m['fpr']:.3f} dly={m['delay']:.2f} "
          f"rwd={m['rwd']:.0f} e40={m['e40']:.0f} cr={m['crash']} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(f'/tmp/v7gb_w{WID}.json', 'w'), default=float)
open(f'V7GB_W{WID}_DONE', 'w').close()
