#!/usr/bin/env python3
# v7_night — 보정된 v7 무대 밤샘 4단 (인자: worker_id 0|1):
#  S1: SWIRL 스윕 α{0.1,0.5}×R{1,1.5,2}×P{0.01,0.03,0.05,0.1} = 24 (N5·seed42)
#  S2: 상위5 × N6
#  S3: 종합순위(F1+reward+샘플효율) 상위3 × terminal −4.0 A/B
#  S4: S3 판정 반영 후 상위3 × FN_MODE=linear (딜레이 재설계) A/B
import sys, os, time, json, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf'); sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
os.environ['SURR_V7'] = '1'
WID = int(sys.argv[1])
CLR = ('RHUKF_PD','RHUKF_PINIT','RHUKF_FORM','RHUKF_ALPHA','RHUKF_R','RHUKF_N','ADAM_LR','OPT',
       'RHUKF_SPAS','TERMINAL_PEN','FN_MODE','FN_C','RHUKF_Q','RHUKF_TAU','RHUKF_UI','NET_HIDDEN')
def go(name, env, seed=42, extra=None):
    for k in CLR: os.environ.pop(k, None)
    for k, v in (extra or {}).items(): os.environ[k] = str(v)
    import importlib, env.reward as RW; importlib.reload(RW)
    import surrogate_run as SR; importlib.reload(SR)
    t0 = time.time()
    h = SR.run_config(env, n_ep=160, seed=seed, ep_steps=400, agent_type='rhukf')
    s = SR.summarize(h)
    s['crash'] = int(sum(r['crashed'] for r in h)); s['crash_late'] = int(sum(r['crashed'] for r in h[80:]))
    s['f1_early'] = float(np.mean([r['f1'] for r in h[20:60] if r['has_atk']]))
    dl = [r['delay'] for r in h[100:] if r['has_atk'] and not np.isnan(r['delay'])]
    s['delay_late'] = float(np.mean(dl)) if dl else -1.0
    rw = [r['reward'] for r in h]
    s['rwd_slope'] = float(np.mean(rw[100:]) - np.mean(rw[20:60]))   # reward 상승량
    print(f"  {name:26s} F1={s['F1']:.3f} early={s['f1_early']:.3f} dly={s['delay_late']:.2f} "
          f"fpr={s['fpr']:.3f} crash={s['crash']}/{s['crash_late']} rwd={s['reward']:.0f} slope={s['rwd_slope']:+.0f} ({time.time()-t0:.0f}s)", flush=True)
    return s
BASE = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1,
            RHUKF_FORM='absolute', RHUKF_SPAS=1)
# ── S1
CFG = []
for a in ('0.1', '0.5'):
    for R in ('1', '1.5', '2'):
        for P in ('0.01', '0.03', '0.05', '0.1'):
            CFG.append((f'a{a}_R{R}_P{P}', dict(BASE, RHUKF_ALPHA=a, RHUKF_R=R, RHUKF_PINIT=P)))
mine = CFG[WID::2]; out = {}
print(f'[w{WID}] S1: {len(mine)}/24', flush=True)
for n, e in mine: out[n] = go(n, e)
json.dump(out, open(f'/tmp/v7n_s1_w{WID}.json', 'w'), default=float)
open(f'V7N_S1_W{WID}_DONE', 'w').close()
while not (os.path.exists('V7N_S1_W0_DONE') and os.path.exists('V7N_S1_W1_DONE')): time.sleep(20)
res = {}
for w in (0, 1): res.update(json.load(open(f'/tmp/v7n_s1_w{w}.json')))
def score(s):   # 종합순위: F1 + 샘플효율 + reward 상승 (z-정규화 합)
    return s['F1'] + 0.5 * s['f1_early'] + 0.002 * s['rwd_slope']
rank = sorted(res.items(), key=lambda kv: -score(kv[1]))
top5 = [n for n, _ in rank[:5]]
if WID == 0:
    print('[S1 상위5] ' + ' | '.join(f"{n}({res[n]['F1']:.3f})" for n in top5), flush=True)
# ── S2: top5 × N6
def envof(n):
    a = n.split('_')[0][1:]; R = n.split('_')[1][1:]; P = n.split('_')[2][1:]
    return dict(BASE, RHUKF_ALPHA=a, RHUKF_R=R, RHUKF_PINIT=P)
jobs2 = [(f'{n}_N6', dict(envof(n), RHUKF_N=6)) for n in top5]
out2 = {}
for n, e in jobs2[WID::2]: out2[n] = go(n, e)
json.dump(out2, open(f'/tmp/v7n_s2_w{WID}.json', 'w'), default=float)
open(f'V7N_S2_W{WID}_DONE', 'w').close()
while not (os.path.exists('V7N_S2_W0_DONE') and os.path.exists('V7N_S2_W1_DONE')): time.sleep(20)
res2 = dict(res)
for w in (0, 1): res2.update(json.load(open(f'/tmp/v7n_s2_w{w}.json')))
rank2 = sorted(((n, s) for n, s in res2.items() if n.split('_N')[0] in top5 or n in top5),
               key=lambda kv: -score(kv[1]))
top3 = [n for n, _ in rank2[:3]]
if WID == 0: print('[S2 후 상위3] ' + ' | '.join(top3), flush=True)
# ── S3: top3 × terminal −4.0
def envof2(n):
    base = n.replace('_N6', ''); e = envof(base)
    if n.endswith('_N6'): e['RHUKF_N'] = 6
    return e
jobs3 = [(f'{n}_t4', envof2(n), {'TERMINAL_PEN': '-4.0'}) for n in top3]
out3 = {}
for n, e, x in jobs3[WID::2]: out3[n] = go(n, e, extra=x)
json.dump(out3, open(f'/tmp/v7n_s3_w{WID}.json', 'w'), default=float)
open(f'V7N_S3_W{WID}_DONE', 'w').close()
while not (os.path.exists('V7N_S3_W0_DONE') and os.path.exists('V7N_S3_W1_DONE')): time.sleep(20)
res3 = {}
for w in (0, 1): res3.update(json.load(open(f'/tmp/v7n_s3_w{w}.json')))
# terminal 채택 판정: top3 평균 score 개선?
base_sc = np.mean([score(res2[n]) for n in top3])
t4_sc = np.mean([score(res3[f'{n}_t4']) for n in top3 if f'{n}_t4' in res3])
use_t4 = t4_sc > base_sc
if WID == 0: print(f'[S3] terminal-4 score {t4_sc:.3f} vs base {base_sc:.3f} → {"채택" if use_t4 else "기각"}', flush=True)
# ── S4: top3 × FN linear (terminal 판정 반영)
extra4 = {'FN_MODE': 'linear', 'FN_C': '0.35'}
if use_t4: extra4['TERMINAL_PEN'] = '-4.0'
jobs4 = [(f'{n}_lin{"_t4" if use_t4 else ""}', envof2(n), dict(extra4)) for n in top3]
out4 = {}
for n, e, x in jobs4[WID::2]: out4[n] = go(n, e, extra=x)
json.dump(out4, open(f'/tmp/v7n_s4_w{WID}.json', 'w'), default=float)
open(f'V7N_S4_W{WID}_DONE', 'w').close()
print(f'[w{WID}] 전체 완료', flush=True)
