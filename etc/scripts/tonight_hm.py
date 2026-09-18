#!/usr/bin/env python3
"""tonight_hm (09-15 22:45, 사용자: 내일 랩미팅 전 오늘 밤 확정) — 새 풀 v5e·하한 0.10·γ0.9·n-step3·짝 무결성(SURR_EP_RNG) 에서 세 무대.
   mix : 정상 티어 혼합 i.i.d. ws0/7/10 = .4/.3/.3 (평시: SWIRL 이 비슷하거나 살짝 우위인가)
   blk : 첫 노출 계단 ep0–59 잔잔 → ep60–109 ws10 → 복귀 (= 사슬 S1 g90n3; SW s42·s43·Adam 전부는 S1 결과 재사용)
   hm  : 은닉 모드 무대 — 소티(에피소드)마다 바뀌는 숨은 환경 모드 m=(바람, 공격자). ep0–59 는 투입 전 익숙한 조건(잔잔 ws0/7 + 결과형 공격자),
         ep60 현장 투입 뒤 두 모드가 독립 마르코프 체인으로 바뀐다. 학습기는 모드를 관측하지 않는다(12D 창 관측만).
         바람 C(ws0/7 .6/.4)↔S(ws10): 체류 기하 평균 τ_C 20·τ_S 15 ep, ep60 초기 모드는 정상분포 추첨.
         공격자 K(결과형, v5 가족 δ ½U(.10,.72)+½U(.72,.84))↔St(은밀 탐색형, δ .8·U(.10,.30)+.2·U(.30,.84)): 체류 τ 20 ep, ep60 초기 정상분포.
         체인은 시드별 고정 난수(3000+seed)로 만들어 학습기 간 동일(짝).
   학습기: GPU = SW(P₀.03·R1·N7) · UKF-TD · EKF-TD(hm 만) / CPU = Adam 3e-4 · Adam 1e-3. 시드 42–46, 150 ep.
   사용: work WID NW (env JOBSET=gpu|cpu) | sched (체인 출력) | count"""
import sys, os, json, time, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf/etc/scripts')
os.environ.setdefault('CHAIN_DIR', '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/chain_hz')   # 풀 v5e 선택 표식 위치
import chain_hz as CH
OUT = f'{CH.N}/tonight'; os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(42, 47)); CALM = '0:0.6,7:0.4'; STORM = '10:1'; FAM_K = '0.10,0.72,0.84,0.5'; FAM_ST = '0.10,0.30,0.84,0.2'
def chain_seq(rng, n, t_a, t_b, pi_b):
    s = rng.random() < pi_b; out = []
    for _ in range(n):
        out.append(s)
        s = (rng.random() >= 1.0 / t_b) if s else (rng.random() < 1.0 / t_a)
    return out
def segs(states, start, spec_true, spec_false, pre_spec):
    out = [f'0-{start - 1}:{pre_spec}']; a = start
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[i - 1]:
            b = start + i - 1; last = (i == len(states))
            out.append(f"{a}-{'' if last else b}:{spec_true if states[i - 1] else spec_false}"); a = b + 1
    return ';'.join(out)
def hm_env(sd):
    rng = np.random.default_rng(3000 + sd)
    wind = chain_seq(rng, 90, 20.0, 15.0, 15.0 / 35.0); atk = chain_seq(rng, 90, 20.0, 20.0, 0.5)
    return dict(SURR_TIER_SCHED=segs(wind, 60, STORM, CALM, CALM), SURR_FAM_SCHED=segs(atk, 60, FAM_ST, FAM_K, FAM_K), BUFFER_SIZE='50000', GAMMA='0.9'), wind, atk
def stage(st, sd):
    if st == 'mix': return CH.stage_mix('g90n3', 1)
    if st == 'blk': return CH.stage_blk('g90n3', 1)
    return hm_env(sd)[0]
def build(js):
    jobs = []
    # 우선순위: GPU 는 SWIRL 3무대 전 시드를 먼저(핵심 주장), 칼만-TD(hm)는 뒤. CPU 는 Adam 전부.
    passes = ([[('blk', 'SW'), ('mix', 'SW'), ('hm', 'SW')], [('hm', 'UKFp03'), ('hm', 'EKFp03')]] if js == 'gpu'
              else [[('mix', 'Adam3e-4'), ('mix', 'Adam1e-3'), ('hm', 'Adam3e-4'), ('hm', 'Adam1e-3')]])
    for ps in passes:
        for sd in SEEDS:
            for st, l in ps:
                if st == 'blk' and sd in (42, 43): continue          # S1 g90n3 SW s42·s43 재사용
                jobs.append((f'{st}_{l}_s{sd}', CH.learner_env(l, c='g90n3'), 'adam' if l.startswith('Adam') else 'rhukf', sd, stage(st, sd), st, l))
    return jobs
def build_tune(js):
    """09-15 23:00 사용자: SWIRL 새 풀에서 재튜닝. P₀ 0.1 · N5 두 변형 × {blk, hm, mix} × 시드 42–44 (GPU 18런). 기본 SW 는 build() 결과와 짝.
       선택은 blk·hm 블록/투입 후 보상에서 가장 좋은 설정 — 평가 시드와 겹치는 탐색 튜닝임을 보고서에 명시."""
    if js != 'gpu': return []
    jobs = []
    for sd in (42, 43, 44):
        for st in ('blk', 'hm', 'mix'):
            for l in ('SWp1', 'SWn5'):
                jobs.append((f'{st}_{l}_s{sd}', CH.learner_env(l, c='g90n3'), 'rhukf', sd, stage(st, sd), st, l))
    return jobs

def build_ams(js):
    """09-15 23:05 사용자: Adam AMSGrad on/off 성능 차이. Adam lr {3e-4, 1e-3} × AMSGrad on × {mix, blk, hm} × 시드 42–46 (CPU 30런). off 는 기존 결과(mix·hm: tonight, blk: S1)."""
    if js != 'cpu': return []
    jobs = []
    for sd in SEEDS:
        for st in ('blk', 'mix', 'hm'):
            for lr, nm in (('3e-4', 'Adam3e-4ams'), ('1e-3', 'Adam1e-3ams')):
                env = dict(CH.ADAMP, ADAM_LR=lr, ADAM_AMSGRAD='1')
                jobs.append((f'{st}_{nm}_s{sd}', env, 'adam', sd, stage(st, sd), st, nm))
    return jobs

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sched':
        for sd in SEEDS:
            e, w, a = hm_env(sd)
            fw = next((60 + i for i, x in enumerate(w) if x), None); fa = next((60 + i for i, x in enumerate(a) if x), None)
            print(f'seed {sd}: 폭풍 비중 {np.mean(w):.2f} 첫 폭풍 ep {fw} · 은밀 비중 {np.mean(a):.2f} 첫 은밀 ep {fa}')
            print('   바람', e['SURR_TIER_SCHED']); print('   공격', e['SURR_FAM_SCHED'])
    elif mode == 'count':
        for js in ('gpu', 'cpu'): jb = build(js); print(js, len(jb), sorted({(j[5], j[6]) for j in jb}))
    elif mode == 'work':
        WID, NW = int(sys.argv[2]), int(sys.argv[3]); js = os.environ.get('JOBSET', 'gpu')
        SET = os.environ.get('TONIGHT_SET', ''); TUNE_SET = SET in ('tune', 'ams'); PFX = {'tune': 'U', 'ams': 'A'}.get(SET, 'T')
        jobs = build_tune(js) if SET == 'tune' else (build_ams(js) if SET == 'ams' else build(js)); mine = jobs[WID::NW]; fo = f'{OUT}/{PFX}{js}_w{WID}.json'
        try: out = json.load(open(fo))
        except Exception: out = {}
        print(f'[tonight {js} w{WID}] {len(mine)}/{len(jobs)}: {[j[0] for j in mine]} pool={CH.POOL}', flush=True)
        for job in mine:
            if job[0] in out: continue
            try: out[job[0]] = CH.run(job)
            except Exception as ex: import traceback; traceback.print_exc(); print(f'  !! {job[0]} 실패: {ex}', flush=True)
            json.dump(out, open(fo + '.tmp', 'w'), default=float); os.replace(fo + '.tmp', fo)
        open(f'{OUT}/{PFX}{js}_w{WID}_DONE', 'w').close(); print(f'[tonight {js} w{WID}] 완료', flush=True)
