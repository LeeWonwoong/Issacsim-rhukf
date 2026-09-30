#!/usr/bin/env python3
"""n132–n134 SWIRL 넓은 튜닝 드라이버 (원격 surrogate, GPU 1개씩, 시드 게이트) — 09-30 23:40 사용자:
   "A-1·vfinal 에서 N 10·12·15 + P·R·α 등 다양하게, CP/LL 튜닝 참고, 3시드지만 s42 가 명백히 나쁘면 넘어가기, SWIRL 핵심 논리 유지".
   설계 = 워크플로 wf_6becddc3-50a (scratchpad/wf_tune_design.json), 규칙 = configs/grids/PREREG_swirl_wide_0930.md (결과 전 등록).

   공통 고정(SWIRL 핵심): form error(매 창 P=pΔ·I 리셋) · 창 N 의 UT 갱신 · 타깃 anchor · act target · argmax spas · huber 0,
   τ .005 · ui 1 · B128 · [24,24] · W6. 모든 셀 q·N = 0.5·pΔ (창 끝 P = 1.5·pΔ). E1 등변 셀 = (pΔ, q, R)×c · α/√c.

   실행(원격): cd ~/projects/Issacsim-rhukf-int && setsid nohup ~/venvs/swirl/bin/python scripts/tune_driver_n132.py > .../n132_driver.out 2>&1 &
   상태: results/claudecodefortest/n132_state.json · 로그 n132_driver.log · 중단: results/claudecodefortest/STOP_N132
"""
import json
import math
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'claudecodefortest')
LOG = os.path.join(RES, 'n132_driver.log')
STATE = os.path.join(RES, 'n132_state.json')
STOP = os.path.join(RES, 'STOP_N132')
T0 = time.time()
CAP_H = 33.5

BASE = {'A1': ['configs/newenv_vfinal.yaml', 'configs/overlays/surrogate_knn_v6.yaml', 'configs/overlays/reward_kA1.yaml'],
        'vf': ['configs/newenv_vfinal.yaml', 'configs/overlays/surrogate_knn_v6.yaml'],
        'A1w': ['configs/newenv_vfinal.yaml', 'configs/overlays/surrogate_knn_v6.yaml', 'configs/overlays/reward_kA1.yaml',
                'configs/overlays/wind_w79.yaml']}
OUT = {'A1': 'n132s_vf_A1', 'vf': 'n133s_vf_cur', 'A1w': 'n134s_w79_A1'}
PFX = {'A1': 'a1', 'vf': 'vf', 'A1w': 'a1w'}
COMMON = {'agent.type': 'swirl', 'run.device': 'cuda', 'run.episodes': 200, 'log.eval_n': 100, 'log.probe_every': 20, 'log.probe_n': 8,
          'obs.window': 6, 'agent.hidden': [24, 24], 'agent.batch': 128, 'agent.tau': 0.005, 'agent.update_interval': 1,
          'agent.buffer': 50000, 'agent.swirl.huber_c': 0, 'agent.swirl.argmax': 'spas', 'agent.swirl.act': 'target'}
R0 = {'A1': 0.25, 'vf': 0.5, 'A1w': 0.25}
MIN = {10: 24.5, 12: 29.5, 15: 36.5, 20: 49.0}


def spec(tag, N=10, c=1.0, alpha=None, rmul=1.0):
    """중심(pΔ .01·R0·α .1)에서: N, 등변 배율 c((pΔ,q,R)×c·α/√c), α 직접 지정, R 추가 배율."""
    pd = 0.01 * c
    return {'agent.swirl.N': N, 'agent.swirl.p_delta': round(pd, 6), 'agent.swirl.R': round(R0[tag] * c * rmul, 6),
            'agent.swirl.q': round(0.5 * pd / N, 9),
            'agent.swirl.alpha': round(alpha if alpha is not None else 0.1 / math.sqrt(c), 6)}


def cells(tag):
    return {'c': spec(tag), 'e10': spec(tag, c=10), 'a32': spec(tag, alpha=0.3162), 'e3': spec(tag, c=3), 'n15': spec(tag, N=15),
            'n12': spec(tag, N=12), 'a05': spec(tag, alpha=0.05), 'n15e3': spec(tag, N=15, c=3), 'e5': spec(tag, c=5),
            'n15iso': spec(tag, N=15, rmul=1.49), 'e3k': spec(tag, c=3, rmul=0.5), 'n20': spec(tag, N=20)}


# 기존 중심 런(짝 기준) — S0 재현 통과 시 그대로 사용
REF_DIRS = {'A1': {s: f'n122s_vfinal_rwA1/a_s_s{s}' for s in (42, 43, 44, 45, 46)},
            'vf': {**{s: f'n110s_vfinal_swirl_tune/a_c_s{s}' for s in (42, 43, 44)}, **{s: f'n122s_vfinal_cur/a_s_s{s}' for s in (45, 46)}}}

ST = {'runs': {}, 'decisions': [], 'ref': {}}


def lg(msg):
    line = f"[{time.strftime('%m-%d %H:%M')}] ({(time.time() - T0) / 3600:.1f} h) {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def save():
    with open(STATE, 'w') as f:
        json.dump(ST, f, ensure_ascii=False, indent=1)


def hours():
    return (time.time() - T0) / 3600


def mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return sum(v) / len(v) if v else float('nan')


def met(d):
    d = d if d.startswith('/') else os.path.join(RES, d)
    H = json.load(open(os.path.join(d, 'hist.json')))['hist']
    E = json.load(open(os.path.join(d, 'eval.json')))
    r = [x['reward'] for x in H]; q = [x.get('qmax') for x in H]
    fin = all(math.isfinite(x) for x in r) and all(x is None or math.isfinite(x) for x in q)
    return {'n': len(H), 'n_eval': E.get('n_ep'), 'finite': fin, 'r200': mean(r), 'r1_60': mean(r[:60]), 'r61_80': mean(r[60:80]),
            'r61_120': mean(r[60:120]), 'r61_200': mean(r[60:200]), 'endQ': mean(q[180:200]), 'trendQ': mean(q[160:200]) - mean(q[120:160]),
            'maxQ': max(x for x in q if x is not None), 'f1': E.get('f1'), 'fpr': E.get('fpr'),
            'kg20': mean([x.get('kgain') for x in H[:20]]), 'pmax5': H[4].get('pmax') if len(H) > 4 else None,
            'sec': mean([x.get('sec') for x in H[20:]])}


def ref_stats(tag):
    R = ST['ref'][tag]
    ss = [s for s in R['seeds'] if R['seeds'][s]]
    return {'endQ5': mean([R['seeds'][s]['endQ'] for s in ss]), 'maxQ5': mean([R['seeds'][s]['maxQ'] for s in ss]),
            'f1_3': mean([R['seeds'][s]['f1'] for s in ss if int(s) <= 44])}


def run(tag, name, seed, cfg):
    """한 셀·한 시드. 이미 hist.json 있으면 재사용. 불완전·NaN 이면 1회 재실행. 반환 = met 또는 None."""
    out = OUT[tag]; d = os.path.join(RES, out, f'a_{name}_s{seed}')
    key = f'{tag}/{name}/s{seed}'
    if not os.path.exists(os.path.join(d, 'hist.json')):
        if os.path.exists(STOP):
            lg('STOP_N132 → 종료'); save(); sys.exit(0)
        if hours() > CAP_H:
            lg(f'누적 {hours():.1f} h > {CAP_H} h → 새 런 시작 안 함 ({key})'); return None
        gy = {'base': BASE[tag], 'outdir': f'results/claudecodefortest/{out}', 'seeds': [seed], 'stages': {'a': {}},
              'learners': {name: {'device': 'gpu', 'set': {**COMMON, **cfg}}}}
        os.makedirs(os.path.join(ROOT, 'configs', 'grids', 'tmp_n132'), exist_ok=True)
        gp = os.path.join('configs', 'grids', 'tmp_n132', f'{out}_{name}_s{seed}.yaml')
        with open(os.path.join(ROOT, gp), 'w') as f:
            yaml.safe_dump(gy, f, allow_unicode=True, sort_keys=False)
        for attempt in (1, 2):
            lg(f'시작 {key} {json.dumps({k.split(".")[-1]: v for k, v in cfg.items()})} (시도 {attempt})')
            env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1')
            with open(os.path.join(RES, f'{out}.launch.log'), 'a') as lf:
                rc = subprocess.run([sys.executable, 'scripts/grid.py', 'run', gp, '--gpu', '1', '--cpu', '0', '--omp-gpu', '2', '--omp-cpu', '1'],
                                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            ok = os.path.exists(os.path.join(d, 'hist.json')) and os.path.exists(os.path.join(d, 'eval.json'))
            if ok:
                m = met(d)
                if m['n'] == 200 and m['finite'] and m['n_eval']:
                    break
            lg(f'  G0 불완전(rc={rc}, hist={ok}) → ' + ('재실행' if attempt == 1 else '포기'))
            if attempt == 1 and os.path.isdir(d):
                os.rename(d, d + f'_bad{int(time.time())}')
        else:
            ST['runs'][key] = None; save(); return None
    m = met(d)
    pr = (m['pmax5'] or 0) / cfg['agent.swirl.p_delta']
    m['g0_pmax_ok'] = abs(pr - 1.5) <= 0.02
    if not m['g0_pmax_ok']:
        lg(f'  ⚠ G0 설정 확인 실패: pmax(ep5)/pΔ = {pr:.3f} (기대 1.50)')
    ST['runs'][key] = m; save()
    lg(f'끝 {key}: r200 {m["r200"]:.2f} r61–200 {m["r61_200"]:.2f} endQ {m["endQ"]:.1f} maxQ {m["maxQ"]:.1f} F1 {m["f1"]:.3f} kg20 {m["kg20"]:.3f} sec/ep {m["sec"]:.2f}')
    return m


def delta(tag, m, seed, k):
    return m[k] - ST['ref'][tag]['seeds'][str(seed)][k]


def g1(tag, m, seed):
    rs = ref_stats(tag); why = []
    if m['endQ'] >= 1.20 * rs['endQ5'] or m['trendQ'] > 1.5: why.append(f'(a) endQ {m["endQ"]:.1f}≥{1.2 * rs["endQ5"]:.1f} 또는 추세 {m["trendQ"]:+.2f}')
    if delta(tag, m, seed, 'r61_200') <= -1.5: why.append(f'(b) Δ61–200 {delta(tag, m, seed, "r61_200"):+.2f}')
    if m['f1'] <= ST['ref'][tag]['seeds'][str(seed)]['f1'] - 0.08: why.append(f'(c) F1 {m["f1"]:.3f}')
    if m['maxQ'] >= 1.35 * rs['maxQ5']: why.append(f'(d) maxQ {m["maxQ"]:.1f}≥{1.35 * rs["maxQ5"]:.1f}')
    return why


def g2(tag, ms):
    rs = ref_stats(tag); why = []
    d200 = mean([delta(tag, ms[s], s, 'r200') for s in (42, 43)]); d61 = mean([delta(tag, ms[s], s, 'r61_200') for s in (42, 43)])
    if d200 <= -1.25 and d61 <= -0.5: why.append(f'(i) 2시드 Δ200 {d200:+.2f}·Δ61–200 {d61:+.2f}')
    w43 = g1(tag, ms[43], 43)
    if any(x.startswith('(a)') or x.startswith('(d)') for x in w43): why.append('(ii) s43 G1(a/d): ' + '; '.join(w43))
    if mean([ms[s]['endQ'] for s in (42, 43)]) >= 1.15 * rs['endQ5'] and d61 < 0: why.append('(iii) 2시드 endQ ≥1.15×·Δ61–200<0')
    return why


def gated(tag, name, cfg, seeds=(42, 43, 44)):
    """s42 → G1 → s43 → G2 → s44. 반환 = {seed: met} (완주 시 3개)."""
    ms = {}
    for s in seeds:
        m = run(tag, name, s, cfg)
        if m is None:
            return ms
        ms[s] = m
        if s == 42:
            w = g1(tag, m, 42)
            if w:
                lg(f'  ✗ G1 탈락 {tag}/{name}: ' + '; '.join(w)); ST['decisions'].append([tag, name, 'G1', w]); save(); return ms
        if s == 43:
            w = g2(tag, ms)
            if w:
                lg(f'  ✗ G2 탈락 {tag}/{name}: ' + '; '.join(w)); ST['decisions'].append([tag, name, 'G2', w]); save(); return ms
    return ms


def summ(tag, ms):
    rs = ref_stats(tag); S = sorted(ms)
    return {'n': len(S), 'd200': mean([delta(tag, ms[s], s, 'r200') for s in S]), 'd61_200': mean([delta(tag, ms[s], s, 'r61_200') for s in S]),
            'd61_80': mean([delta(tag, ms[s], s, 'r61_80') for s in S]), 'pos200': sum(delta(tag, ms[s], s, 'r200') > 0 for s in S),
            'pos61': sum(delta(tag, ms[s], s, 'r61_200') > 0 for s in S),
            'endQr': mean([ms[s]['endQ'] for s in S]) / rs['endQ5'], 'dF1': mean([ms[s]['f1'] for s in S]) - rs['f1_3']}


def g3(tag, name, cfg, ms):
    if len(ms) < 3: return False
    s = summ(tag, {k: ms[k] for k in (42, 43, 44)})
    return (s['d200'] >= 0.5 and s['pos200'] >= 2 and s['endQr'] < 1.10 and s['dF1'] >= -0.01 and s['d61_200'] >= 0
            and cfg['agent.swirl.N'] <= 15)


def g3p(tag, name, cfg, ms):
    if len(ms) < 3 or not (cfg['agent.swirl.p_delta'] >= 0.03 or cfg['agent.swirl.N'] >= 12): return False
    s = summ(tag, {k: ms[k] for k in (42, 43, 44)})
    return s['d200'] >= -0.3 and s['d61_200'] >= -0.3 and s['endQr'] < 1.10 and s['dF1'] >= -0.01 and cfg['agent.swirl.N'] <= 15


def set_ref(tag, seed_dirs):
    ST['ref'][tag] = {'seeds': {str(s): met(d) for s, d in seed_dirs.items()}}; save()


def s0(tag):
    """중심 s42 재현: |Δ200|·|Δ61–200| ≤ 0.3 이면 기존 5시드 재사용, 아니면 중심 s43·s44 새로 → 새 3시드(+기존 s45·46) 기준."""
    C = cells(tag)
    set_ref(tag, REF_DIRS[tag])
    m = run(tag, 'c', 42, C['c'])
    if m is None:
        return
    d2, d6 = delta(tag, m, 42, 'r200'), delta(tag, m, 42, 'r61_200')
    ok = abs(d2) <= 0.3 and abs(d6) <= 0.3
    lg(f'S0 {tag}: 재현 Δ200 {d2:+.3f} Δ61–200 {d6:+.3f} → {"통과(기존 기준 재사용)" if ok else "실패 → 중심 s43·s44 새로"}')
    ST['decisions'].append([tag, 'c', 'S0', [d2, d6, ok]])
    if not ok:
        for s in (43, 44):
            run(tag, 'c', s, C['c'])
        nd = dict(REF_DIRS[tag]); nd.update({s: os.path.join(OUT[tag], f'a_c_s{s}') for s in (42, 43, 44)})
        set_ref(tag, nd)
    save()


def block(tag, main, conds):
    C = cells(tag); done = {}
    s0(tag)
    alive = list(main)
    for s in (42, 43, 44):                                   # 웨이브: 모든 셀 s42 → 통과 셀 s43 → s44
        for name in list(alive):
            m = run(tag, name, s, C[name])
            if m is None:
                alive.remove(name); continue
            done.setdefault(name, {})[s] = m
            w = g1(tag, m, 42) if s == 42 else (g2(tag, done[name]) if s == 43 else [])
            if w:
                lg(f'  ✗ {"G1" if s == 42 else "G2"} 탈락 {tag}/{name}: ' + '; '.join(w))
                ST['decisions'].append([tag, name, 'G1' if s == 42 else 'G2', w]); alive.remove(name); save()
    full = lambda n: n in done and len(done[n]) == 3
    for name, cond, cap in conds:                              # 조건부 셀
        if hours() > cap:
            lg(f'조건부 {tag}/{name}: 누적 {hours():.1f} h > 캡 {cap} h → 생략'); continue
        try:
            fire = cond(done, full)
        except Exception as e:
            fire = False; lg(f'조건부 {tag}/{name} 판정 오류 {e}')
        lg(f'조건부 {tag}/{name}: {"발동" if fire else "미발동"}')
        ST['decisions'].append([tag, name, 'cond', bool(fire)])
        if fire:
            done[name] = gated(tag, name, C[name])
    save()
    S = {n: summ(tag, done[n]) for n in done if len(done[n]) == 3}
    ST.setdefault('summary', {})[tag] = S; save()
    for n, s in sorted(S.items(), key=lambda kv: -kv[1]['d61_200']):
        lg(f'  3시드 {tag}/{n}: Δ200 {s["d200"]:+.2f} ({s["pos200"]}/3) Δ61–200 {s["d61_200"]:+.2f} Δ61–80 {s["d61_80"]:+.2f} endQ×{s["endQr"]:.2f} ΔF1 {s["dF1"]:+.3f}'
           f'{" G3" if g3(tag, n, C[n], done[n]) else ""}{" G3′" if g3p(tag, n, C[n], done[n]) else ""}')
    return done


def confirm(tag, done, cap):
    """5시드 확인 대상: G3 후보(Δ61–200 최대) → 없으면 Δ200≥0·Δ61–200≥0·endQ<1.10·ΔF1≥−.01·N≤15 최상위(순위 확인용)."""
    C = cells(tag); S = {n: summ(tag, {k: done[n][k] for k in (42, 43, 44)}) for n in done if len(done[n]) >= 3 and n != 'c'}
    g3s = [n for n in S if g3(tag, n, C[n], done[n])]
    pool = g3s or [n for n in S if S[n]['d200'] >= 0 and S[n]['d61_200'] >= 0 and S[n]['endQr'] < 1.10 and S[n]['dF1'] >= -0.01
                   and C[n]['agent.swirl.N'] <= 15]
    rank = sorted(pool, key=lambda n: -S[n]['d61_200'])
    ST.setdefault('confirm', {})[tag] = {'g3': g3s, 'rank': rank}; save()
    if not rank:
        lg(f'5시드 확인 {tag}: 대상 없음 → 중심 유지'); return None, g3s
    for i, n in enumerate(rank[:2]):                           # 1위 → (캡 안이면) 2위
        if i == 1 and hours() > 30.0:
            break
        if i == 0 and hours() > cap:
            lg(f'5시드 확인 {tag}/{n}: 캡 {cap} h 초과 → 생략'); break
        lg(f'5시드 확인 {tag}/{n} ({"G3 후보" if n in g3s else "순위 확인용"})')
        for s in (45, 46):
            m = run(tag, n, s, C[n])
            if m: done[n][s] = m
        s5 = summ(tag, done[n]) if len(done[n]) == 5 else None
        if s5:
            adopt = n in g3s and s5['d61_200'] >= 0 and s5['pos61'] >= 3 and s5['endQr'] < 1.10
            ST['confirm'][tag][f'{n}_5seed'] = {**s5, 'adopt': adopt}; save()
            lg(f'  5시드 {tag}/{n}: Δ200 {s5["d200"]:+.2f} Δ61–200 {s5["d61_200"]:+.2f} ({s5["pos61"]}/5) endQ×{s5["endQr"]:.2f} → {"채택 권고" if adopt else "채택 아님"}')
    return rank[0], g3s


def w79(best_name):
    """A-1 w79 이식: a1w_c 3시드(중심) 기준으로 a1w_best 3시드 게이트 비교. 제안만(Isaac 자동 교체 없음)."""
    CA = cells('A1')
    ms_c = {}
    for s in (42, 43, 44):
        m = run('A1w', 'c', s, CA['c'])
        if m: ms_c[s] = m
    if len(ms_c) < 3:
        return
    ST['ref']['A1w'] = {'seeds': {str(s): ms_c[s] for s in ms_c}}; save()
    if best_name:
        ms = gated('A1w', 'best_' + best_name, CA[best_name])
        if len(ms) == 3:
            s = summ('A1w', ms)
            lg(f'  w79 이식 {best_name}: Δ200 {s["d200"]:+.2f} ({s["pos200"]}/3) Δ61–120 '
               f'{mean([delta("A1w", ms[k], k, "r61_120") for k in ms]):+.2f} endQ×{s["endQr"]:.2f}')
            ST['w79'] = {'best': best_name, **s}; save()


def main():
    lg(f'시작 — 드라이버 {os.path.basename(__file__)} · 캡 {CAP_H} h · 원격 GPU 1개씩(CUDA_LAUNCH_BLOCKING=1)')
    while True:                                              # 다른 surrogate 학습(r28 의 S21 등)이 끝날 때까지
        p = subprocess.run(['bash', '-c', 'ps -eo args | grep -v "bash -c" | grep -c "[t]rain.py.*surrogate_knn"'], capture_output=True, text=True).stdout.strip()
        if p == '0':
            break
        time.sleep(30)
    lg('GPU 비어 있음 확인 → 시작')
    main_cells = ['e10', 'a32', 'e3', 'n15', 'n12', 'a05']
    condA = [('n15e3', lambda d, f: f('e3') and f('n15'), 99),
             ('e5', lambda d, f: f('e3') and 'e10' in d and len(d['e10']) < 3, 99),
             ('n15iso', lambda d, f: f('n15') and (abs(summ('A1', d['n15'])['d200']) >= 0.5 or abs(summ('A1', d['n15'])['d61_80']) >= 2.0), 99),
             ('e3k', lambda d, f: f('e3') and g3p('A1', 'e3', cells('A1')['e3'], d['e3']), 99)]
    doneA = block('A1', main_cells, condA)
    bestA, g3A = confirm('A1', doneA, 99)
    candA = None
    ca = ST.get('confirm', {}).get('A1', {})
    adoptA = [k[:-6] for k, v in ca.items() if k.endswith('_5seed') and v.get('adopt')]
    g3pA = [n for n in doneA if n != 'c' and g3p('A1', n, cells('A1')[n], doneA[n])]
    candA = adoptA[0] if adoptA else (sorted(g3pA, key=lambda n: -summ('A1', doneA[n])['d61_200'])[0] if g3pA else None)
    if candA:
        lg(f'w79 이식 후보(A-1): {candA} ({"채택 권고" if adoptA else "G3′"})'); w79(candA)
    condV = [('n15e3', lambda d, f: f('e3') and f('n15'), 26),
             ('n15iso', lambda d, f: f('n15') and (abs(summ('vf', d['n15'])['d200']) >= 0.5 or abs(summ('vf', d['n15'])['d61_80']) >= 2.0), 28),
             ('e3k', lambda d, f: f('e3') and g3p('vf', 'e3', cells('vf')['e3'], d['e3']), 28)]
    doneV = block('vf', main_cells, condV)
    confirm('vf', doneV, 31)
    # 꼬리
    sA = {n: summ('A1', doneA[n]) for n in ('n12', 'n15') if n in doneA and len(doneA[n]) >= 3}
    if len(sA) == 2 and sA['n15']['d200'] >= sA['n12']['d200'] >= 0 and sA['n15']['d200'] >= 0.3 and hours() <= 30:
        lg('꼬리: a1_n20 발동(N 단조 증가)'); doneA['n20'] = gated('A1', 'n20', cells('A1')['n20'])
    else:
        lg('꼬리: a1_n20 미발동')
    if not candA and hours() <= 32:
        lg('꼬리: 후보 없음 → a1w_c 3시드만(w79 중심 기준값)'); w79(None)
    lg('드라이버 종료'); ST['done'] = True; save()
    open(os.path.join(RES, 'N132_DONE'), 'w').close()


if __name__ == '__main__':
    main()
