#!/usr/bin/env python3
# obs_scale_swirl — obs [0,3] vs [0,1](÷3) 이 SWIRL vs Adam 에 상대적으로 다르게 작용하나?
#   가설: 고정-R 칼만(SWIRL)은 입력스케일 민감, Adam은 스케일 불변 → /3 이 SWIRL 만 도움.
#   surrogate(실 agent+normalizer+reward, NIS만 합성) 로 다시드 빠른 검증. 상대비교라 유효.
#   ⚠ A/B speed 테스트(GPU) 완료 후 실행 (SPEED_AB_DONE 대기 — RTF 오염 방지).
import sys, os, time, numpy as np
os.chdir('/home/acsl/projects/Issacsim-rhukf')
sys.path.insert(0, 'etc/scripts')

# ── A/B 완료 대기 (GPU free) ──
print('[wait] SPEED_AB_DONE 대기 (A/B GPU 점유 중)...', flush=True)
while not os.path.exists('SPEED_AB_DONE'):
    time.sleep(60)
time.sleep(30)
print('[go] A/B 완료 확인 → obs-scale 실험 시작', flush=True)

from surrogate_run import run_config, summarize

RH = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU='0.005', RHUKF_UI=1, RHUKF_R='1.5', RHUKF_PD='0.01')
AD = dict(NET_HIDDEN=16, ADAM_AMSGRAD='0')
SEEDS = [42]; NEP = 200
CFG = [('rhukf', RH, '[0,3]', 1.0), ('rhukf', RH, '[0,1]', 3.0),
       ('adam', AD, '[0,3]', 1.0), ('adam', AD, '[0,1]', 3.0)]

res = {}
for ag, env, tag, od in CFG:
    key = (ag, tag); res[key] = []
    for sd in SEEDS:
        t0 = time.time()
        h = run_config(env, obs_mode='raw4', n_ep=NEP, seed=sd, agent_type=ag, obs_div=od)
        s = summarize(h); res[key].append(s)
        print(f'  {ag:5s} {tag} seed{sd}: F1={s["F1"]:.3f} P={s["P"]:.3f} R={s["R"]:.3f} '
              f'rwd={s["reward"]:.1f} ({time.time()-t0:.0f}s)', flush=True)

def agg(key, k): return np.mean([r[k] for r in res[key]]), np.std([r[k] for r in res[key]])

print('\n' + '='*64)
print(' obs-scale × optimizer (mean±std, 3 seeds)')
print('='*64)
print(f'{"":6} {"obs":6} | {"F1":>13} | {"reward":>13}')
for ag in ['rhukf', 'adam']:
    for tag in ['[0,3]', '[0,1]']:
        f1m, f1s = agg((ag, tag), 'F1'); rm, rs = agg((ag, tag), 'reward')
        print(f'{ag:6} {tag:6} | {f1m:.3f}±{f1s:.3f} | {rm:6.1f}±{rs:4.1f}')

print('\n── 가설 검정: /3([0,1]) 가 어느 옵티마이저를 더 돕나 ──')
for ag in ['rhukf', 'adam']:
    f1_03 = agg((ag, '[0,3]'), 'F1')[0]; f1_01 = agg((ag, '[0,1]'), 'F1')[0]
    r_03 = agg((ag, '[0,3]'), 'reward')[0]; r_01 = agg((ag, '[0,1]'), 'reward')[0]
    print(f'  {ag:6}: ΔF1([0,1]-[0,3])={f1_01-f1_03:+.3f}  Δreward={r_01-r_03:+.1f}')
ds = agg(('rhukf','[0,1]'),'F1')[0]-agg(('rhukf','[0,3]'),'F1')[0]
da = agg(('adam','[0,1]'),'F1')[0]-agg(('adam','[0,3]'),'F1')[0]
print(f'\n  판정: ΔF1_SWIRL={ds:+.3f} vs ΔF1_Adam={da:+.3f}')
print(f'  → {"가설 지지: /3 이 SWIRL 을 더 도움" if ds>da+0.01 else ("무차별/역방향: obs스케일 효과 미미 or Adam도 동등" )}')
np.save('/tmp/obs_scale_res.npy', res, allow_pickle=True)
open('OBS_SCALE_DONE','w').close()
print('\n[done] OBS_SCALE_DONE')
