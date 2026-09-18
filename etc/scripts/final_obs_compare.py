#!/usr/bin/env python3
# final_obs_compare v2 — topN(배터리) 상위 5 → 관측 {[0,3],[0,1](/3)} × seed{42,43} 최종 배터리 비교.
#   [0,3] seed42 는 topN 결과 재사용, 나머지 15런 신규.
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V4'] = '1'
print('[wait] GRID64_TOPN_DONE 대기...', flush=True)
while not os.path.exists('GRID64_TOPN_DONE'):
    time.sleep(60)
time.sleep(5)
topn = json.load(open('/tmp/grid64_topN.json'))
rank = sorted(topn.items(), key=lambda kv: -kv[1]['F1'])
top5 = [n for n, _ in rank[:5]]
print('[top5]', top5, flush=True)
from surrogate_run import run_config, summarize
def env_of(full):
    name, Ns = full.rsplit('_N', 1)
    t, R, Q, P, m = name.split('_')
    tau = '0.' + t[1:t.index('u')]; ui = int(t[t.index('u') + 1:])
    env = dict(NET_HIDDEN=16, RHUKF_N=int(Ns), RHUKF_Q=('1e-' + Q[1:])[:4],
               RHUKF_TAU=tau, RHUKF_UI=ui, RHUKF_R=R[1:])
    if m == 'err': env['RHUKF_PD'] = P[1:]
    else: env['RHUKF_FORM'] = 'absolute'; env['RHUKF_PINIT'] = P[1:]
    return env
out = {}
for full in top5:
    for od, otag in [(1.0, 'obs03'), (3.0, 'obs01')]:
        for sd in (42, 43):
            if otag == 'obs03' and sd == 42:
                out[f'{full}|{otag}|s42'] = topn[full]; continue   # 재사용
            for k in ('RHUKF_FORM', 'RHUKF_PINIT', 'RHUKF_PD'):
                os.environ.pop(k, None)
            t0 = time.time()
            h, p = run_config(env_of(full), n_ep=160, seed=sd, agent_type='rhukf', probe=True, obs_div=od)
            s = summarize(h); s.update(p); out[f'{full}|{otag}|s{sd}'] = s
            print(f'  {full} {otag} s{sd}: F1={s["F1"]:.3f} conv={s["conv_ep"]:.0f} pFP={s["probe_fp"]:.3f} '
                  f'wR={s["probe_weak_rec"]:.2f} ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open('/tmp/final_obs_compare.json', 'w'), default=float)
print('\n══ 최종 배터리 (config × obs, 2시드 평균) ══', flush=True)
print(f'{"config":28} {"obs":6} | {"F1":>11} {"conv":>5} {"stab":>6} {"pFP":>6} {"wRec":>10} {"sRec":>10}', flush=True)
for full in top5:
    for otag in ('obs03', 'obs01'):
        ss = [out[f'{full}|{otag}|s{sd}'] for sd in (42, 43) if f'{full}|{otag}|s{sd}' in out]
        def mm(k): 
            v=[x[k] for x in ss if not (isinstance(x[k],float) and np.isnan(x[k]))]; return (np.mean(v), np.std(v)) if v else (float('nan'),0)
        f1=mm('F1'); cv=mm('conv_ep'); st=mm('stab'); pf=mm('probe_fp'); wr=mm('probe_weak_rec'); sr=mm('probe_strong_rec')
        print(f'{full:28} {otag:6} | {f1[0]:.3f}±{f1[1]:.3f} {cv[0]:>5.0f} {st[0]:>6.3f} {pf[0]:>6.3f} '
              f'{wr[0]:.2f}±{wr[1]:.2f} {sr[0]:.2f}±{sr[1]:.2f}', flush=True)
open('FINAL_OBS_DONE', 'w').close()
