import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V7'] = '1'
WID = int(sys.argv[1])
CLR = ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_R','RHUKF_N','RHUKF_HUBER_C',
       'RHUKF_SPAS','RHUKF_Q','RHUKF_TAU','RHUKF_UI','NET_HIDDEN')
def go(name, env, seed):
    for k in CLR: os.environ.pop(k, None)
    import surrogate_run as SR
    t0 = time.time(); h = SR.run_config(env, n_ep=160, seed=seed, ep_steps=400, agent_type='rhukf')
    s = SR.summarize(h); s['crash'] = int(sum(r['crashed'] for r in h))
    s['f1_early'] = float(np.mean([r['f1'] for r in h[20:60] if r['has_atk']]))
    dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    s['delay'] = float(np.mean(dl)) if dl else -1.0
    rw = [r['reward'] for r in h]
    fin = float(np.mean(rw[120:])); tgt = 0.9 * fin
    mv = np.convolve(rw, np.ones(5) / 5, 'valid')
    s['rwd_final'] = fin
    s['conv_ep'] = int(next((i for i, r in enumerate(mv) if r >= tgt), len(rw)))
    print(f"  {name:24s} F1={s['F1']:.3f} e={s['f1_early']:.3f} dly={s['delay']:.2f} fpr={s['fpr']:.3f} "
          f"cr={s['crash']} conv={s['conv_ep']} rwd={fin:.0f} ({time.time()-t0:.0f}s)", flush=True)
    return s
# R1 · α0.1 · N6 고정, P 5개, huber on/off, seed 2개 = 20런
B = dict(NET_HIDDEN=16, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_FORM='absolute',
         RHUKF_SPAS=1, RHUKF_ALPHA='0.1', RHUKF_R='1', RHUKF_N=6)
jobs = []
for P in ('0.01', '0.03', '0.05', '0.07', '0.1'):
    for hub, hc in (('hON', '3.0'), ('hOFF', '1e9')):
        for sd in (42, 43):
            jobs.append((f'P{P}_{hub}_s{sd}', dict(B, RHUKF_PINIT=P, RHUKF_HUBER_C=hc), sd))
mine = jobs[WID::2]; out = {}
print(f'[w{WID}] {len(mine)} runs', flush=True)
for n, e, sd in mine: out[n] = go(n, e, sd)
json.dump(out, open(f'/tmp/v7hns_w{WID}.json', 'w'), default=float)
open(f'V7HNS_W{WID}_DONE', 'w').close()
