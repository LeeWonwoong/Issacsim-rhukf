#!/usr/bin/env python3
"""tonight_queue (09-15 23:20, 사용자 승인) — 작업 선점(claim) 대기열. 워커는 아무 때나 더 띄울 수 있다.
   GPU 순서: ① 지속시간 스윕 SWIRL 10런 → ② SWIRL 재튜닝 18런(tonight_hm.build_tune). CPU: 지속시간 스윕 Adam 3e-4·1e-3 20런.
   지속시간 스윕 per{5,20}: 바람 모드 C(ws0 .571 / ws7 .429) ↔ S(ws10), 주변 비율 π_S 0.3 고정(= mix 의 ws0 .4·ws7 .3·ws10 .3 과 같은 비율),
     에피소드 간 1차 자기상관 ρ 로 지속만 바꾼다: P(S→S)=π+(1−π)ρ, P(C→S)=π(1−ρ). 강풍 평균 지속 1/((1−π)(1−ρ)) = 5 ep(ρ .714) / 20 ep(ρ .929).
     ρ=0(평균 지속 1.43 ep)이 무작위 혼합 mix 셀과 같다. ep0 초기 모드는 π 에서 추첨. 공격자 가족은 v5 고정(mix 와 동일). 체인은 시드별 고정 난수(4000+seed)로 학습기 간 동일.
   사용: work (env JOBSET=gpu|cpu) | sched | count"""
import sys, os, json, time, numpy as np
sys.path.insert(0, '/home/acsl/projects/Issacsim-rhukf/etc/scripts')
os.environ.setdefault('CHAIN_DIR', '/home/acsl/projects/Issacsim-rhukf/results/claudecodefortest/night/chain_hz')
import chain_hz as CH, tonight_hm as TH
Q = f'{CH.N}/tonight/q'; os.makedirs(f'{Q}/claims', exist_ok=True)
PI = 0.3; RHO = {'per5': 1 - 0.2 / 0.7, 'per20': 1 - 0.05 / 0.7}; CALM = '0:0.571,7:0.429'; STORM = '10:1'
def _draw(rng, rho, n):
    s = rng.random() < PI; out = []
    for _ in range(n):
        out.append(s); u = rng.random()
        s = (u < PI + (1 - PI) * rho) if s else (u < PI * (1 - rho))
    return out
def pers_seq(sd, rho, n=150):
    """결과를 보기 전 스케줄 속성만으로 고르는 층화 추출(09-15 23:25 등록): 실현 강풍 비중 0.25–0.35 ∧ 강풍 진입 ≥2 ∧ ep40 이후 진입 ≥1.
       150 ep 는 짧아 ρ 가 크면 실현 비중이 0.08–0.71 로 흔들려 무대 난이도가 시드마다 달라지는 것을 막는다. 시드별 결정적(난수 4000+seed, k 번째 추출)."""
    rng = np.random.default_rng(4000 + sd)
    for k in range(20000):
        seq = _draw(rng, rho, n); ent = [i for i in range(1, n) if seq[i] and not seq[i - 1]] + ([0] if seq[0] else [])
        if 0.25 <= np.mean(seq) <= 0.35 and len(ent) >= 2 and any(e >= 40 for e in ent): return seq
    raise RuntimeError('조건 만족 체인 없음')
def pers_env(sd, cell):
    seq = pers_seq(sd, RHO[cell]); segs = []; a = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[i - 1]:
            segs.append(f"{a}-{'' if i == len(seq) else i - 1}:{STORM if seq[i - 1] else CALM}"); a = i
    return dict(SURR_TIER_SCHED=';'.join(segs), BUFFER_SIZE='50000', GAMMA='0.9'), seq
def pers_seq_g(sd, rho, pi, base, lo, hi, min_ent=2, n=150):
    """일반화 체인(09-16 연장용): 주변 비율 pi, 자기상관 rho, 실현 비중 [lo, hi] ∧ 진입 ≥min_ent ∧ ep40 이후 진입 ≥1 층화. rho=0 이면 i.i.d."""
    rng = np.random.default_rng(base + sd)
    for k in range(20000):
        s = rng.random() < pi; seq = []
        for _ in range(n):
            seq.append(s); u = rng.random(); s = (u < pi + (1 - pi) * rho) if s else (u < pi * (1 - rho))
        ent = [i for i in range(1, n) if seq[i] and not seq[i - 1]] + ([0] if seq[0] else [])
        if lo <= np.mean(seq) <= hi and len(ent) >= min_ent and any(e >= 40 for e in ent): return seq
    raise RuntimeError('조건 만족 체인 없음')
def segs_of(seq, t, f):
    out = []; a = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[i - 1]:
            out.append(f"{a}-{'' if i == len(seq) else i - 1}:{t if seq[i - 1] else f}"); a = i
    return ';'.join(out)
FAM_K = '0.10,0.72,0.84,0.5'; FAM_ST = '0.10,0.30,0.84,0.2'
RHO_ALL = {'per5': 1 - 0.2 / 0.7, 'per10': 1 - (1 / 10) / 0.7, 'per20': 1 - 0.05 / 0.7}
def ext_stage(cell, sd):
    """연장 셀: per5/per10/per20(바람만), mixJ(바람 i.i.d. mix + 공격자 i.i.d. 은밀 50 %), perJ20(바람 per20 + 공격자 유형 체인 ρ 같은 강도·은밀 비중 .4–.6), blk10k(버퍼 10k 급변 블록)."""
    if cell in RHO_ALL:
        seq = pers_seq_g(sd, RHO_ALL[cell], PI, 4000, 0.25, 0.35); return dict(SURR_TIER_SCHED=segs_of(seq, STORM, CALM), BUFFER_SIZE='50000', GAMMA='0.9')
    if cell == 'mixJ':
        at = pers_seq_g(sd, 0.0, 0.5, 5000, 0.4, 0.6); return dict(SURR_TIER_P='0:0.4,7:0.3,10:0.3', SURR_FAM_SCHED=segs_of(at, FAM_ST, FAM_K), BUFFER_SIZE='50000', GAMMA='0.9')
    if cell == 'perJ20':
        w = pers_seq_g(sd, RHO_ALL['per20'], PI, 4000, 0.25, 0.35); at = pers_seq_g(sd, 1 - (1 / 20) / 0.5, 0.5, 5000, 0.4, 0.6)
        return dict(SURR_TIER_SCHED=segs_of(w, STORM, CALM), SURR_FAM_SCHED=segs_of(at, FAM_ST, FAM_K), BUFFER_SIZE='50000', GAMMA='0.9')
    GUST = dict(SURR_GUST_P='0.5', SURR_GUST_LO='20', SURR_GUST_HI='50', SURR_GUST_G='2.5', SURR_GUST_V='2.0', SURR_GUST_ATK_AFTER='1')
    if cell == 'gblk': return dict(CH.stage_blk('g90n3', 1), **GUST)          # 돌풍(라벨 clean) + 첫 노출 급변 (탐색 1순위)
    if cell == 'gmix': return dict(CH.stage_mix('g90n3', 1), **GUST)          # 돌풍 + 조건 고정 (탐색 5순위, gblk 의 계단-끔 대조)
    if cell == 'final1':   # ★09-16 확정 무대: 바람 점진 확대 + 다중 버스트 3 + 상수 보상(FP=FN=-2) + 확약 없음 + 1스텝 TD + 200판
        return dict(SURR_TIER_SCHED='0-59:0:1;60-119:0:0.4,6:0.3,7:0.3;120-:0:0.25,6:0.25,7:0.25,10:0.25',
                    ATK_BURSTS='3', R_FP='-2.0', FN_BASE='-2.0', FN_PER='0', FN_ONSET_MULT='1',
                    HOVER_DWELL='1', NSTEP_OFF='1', BUFFER_SIZE='50000', GAMMA='0.9', _NEP='200')
    if cell in ('final5', 'final5n', 'final51', 'final51n', 'final5t', 'final5th'):   # ★09-16 결과성 밴드 한정 + 관측 클립 제거
        # final5  : final4 와 동일(바람 점진 확대) + 공격 밴드 [0.72, 0.80)  ← 워크플로 판정 구간
        # final51 : 바람을 ws0/6/7 균일(ws10 제외) · 스케줄 없이 전 에피소드 동일 분포
        # *n      : 관측 /4 정규화([0,1]) 추가
        d = dict(ATK_BURSTS='1', ATK_DELTA_LO='0.72', ATK_SPLIT='0.76', ATK_DELTA_HI='0.80', ATK_P_UPPER='0.5',
                 R_FP='-2.0', FN_BASE='-2.0', FN_PER='0', FN_ONSET_MULT='1',
                 HOVER_DWELL='1', NSTEP_OFF='1', BUFFER_SIZE='50000', GAMMA='0.9', TERMINAL_PEN='0',
                 SURR_CLIP='1e9', SURR_OBS_CLIP='1e9', _NEP='200')
        if cell.startswith('final51'):
            d['SURR_TIER_P'] = '0:0.3334,6:0.3333,7:0.3333'      # 균일, 전 에피소드 동일 (ws10 제외)
        else:
            d['SURR_TIER_SCHED'] = '0-59:0:1;60-119:0:0.4,6:0.3,7:0.3;120-:0:0.25,6:0.25,7:0.25,10:0.25'
        if cell.endswith('n'):
            d['SURR_OBS_DIV'] = '4.0'
        if cell in ('final5t', 'final5th'):
            d['TERMINAL_PEN'] = '-20'   # ★09-16 절벽을 가치함수에 보이게: γ0.9 지평 10스텝에서 추락 손실 5.0 → 25.0
        if cell == 'final5th':          # ★09-16 사용자: 상한을 0.84 로 되돌려 절벽 빈도 판정(p_loc 1.0 구간 복원)
            d['ATK_SPLIT'] = '0.78'; d['ATK_DELTA_HI'] = '0.84'
        return d
    if cell == 'per5n1c4':   # ★09-16 per5n1c + 관측 /4 정규화([0,1])
        # 근거: final5 계열 실측에서 /4 가 SW−Adam 격차를 세 무대 모두 축소.
        #   δ0.30-0.84 −1.46→−0.38 · δ0.72-0.80 −1.70→−0.36 · ws0/6/7 −4.29→−0.39 (후반 120-199)
        #   기전: 칼만 이득은 관측 크기에 직접 의존, Adam 은 2차 모멘트로 스케일 흡수 → SWIRL 만 손해
        d = dict(ext_stage('per5n1c', sd)); d['SURR_OBS_DIV'] = '4.0'
        return d
    if cell == 'per5n1c':   # ★09-16 전수조사 결과 가장 유망한 조건 + 오늘의 클립 수정
        # 근거: xper5_n1 에서 SW−Adam3e-4 +0.751 (t 2.45, 3/3), vs Adam3e-4ams +1.085 (t 4.59, 3/3)
        # 구성: per5 바람 불규칙 전환(시드별) + 1스텝 TD + 확약 5(ENV0) + δ0.10-0.84(ENV0) + 관측 클립 제거
        d = dict(ext_stage('per5', sd))
        d.update(NSTEP_OFF='1', SURR_CLIP='1e9', SURR_OBS_CLIP='1e9')
        return d
    if cell in ('final4d5', 'final4d10', 'final4d1'):   # ★09-16 확약(HOVER_DWELL) 축: final4 무대 + 확약만 변경
        # 한 번의 hover 결정이 D 스텝 동안 강제 유지 → 결정이 다중스텝 결과를 낳음 = 신용할당 생성
        # per5 계열 실측: D1 에서 SW−Adam1e-3 −0.683 (0/5), D10 에서 +0.407 (1/3) → 부호 역전
        d = dict(ext_stage('final4', sd))
        d['HOVER_DWELL'] = {'final4d1': '1', 'final4d5': '5', 'final4d10': '10'}[cell]
        return d
    if cell in ('final4', 'final4n'):   # ★09-16 사용자 확정: final3 공격 + final1 점진 바람 + 관측 클립 제거(log1p 유지)
        # 관측 상한이 두 곳(SURR_CLIP=환경단, OBS_CLIP=관측빌더단)이라 후자가 전자를 무효화하고 있었다 → 둘 다 해제.
        # 실측 효과: 강공격 b6 vs b7 분리도 d' 0.055 → 0.260 (풀 검열비율 0.002% 로 신호 거의 온전)
        d = dict(SURR_TIER_SCHED='0-59:0:1;60-119:0:0.4,6:0.3,7:0.3;120-:0:0.25,6:0.25,7:0.25,10:0.25',
                 ATK_BURSTS='1', ATK_DELTA_LO='0.30', ATK_SPLIT='0.74', ATK_DELTA_HI='0.84', ATK_P_UPPER='0.5',
                 R_FP='-2.0', FN_BASE='-2.0', FN_PER='0', FN_ONSET_MULT='1',
                 HOVER_DWELL='1', NSTEP_OFF='1', BUFFER_SIZE='50000', GAMMA='0.9', TERMINAL_PEN='0',
                 SURR_CLIP='1e9', SURR_OBS_CLIP='1e9', _NEP='200')
        if cell == 'final4n':
            d.update(SURR_OBS_DIV='4.0')   # gyro [0,1.00] · vel [0,0.79] (같은 제수로 채널 상대크기 유지)
        return d
    if cell in ('final3', 'final3g'):   # ★09-16 확정안: 공격 δ 0.30–0.74(50%)+0.74–0.84(50%) · 버스트 1 · 평시조건 20판 주기 반복 전환 · 돌풍 on/off 대조
        WS = ';'.join([f'{k*20}-{(k+1)*20-1}:' + ('0:1' if k % 2 == 0 else '0:0.2,6:0.3,7:0.3,10:0.2') for k in range(9)] + ['180-:0:0.2,6:0.3,7:0.3,10:0.2'])
        d = dict(SURR_TIER_SCHED=WS, ATK_BURSTS='1',
                 ATK_DELTA_LO='0.30', ATK_SPLIT='0.74', ATK_DELTA_HI='0.84', ATK_P_UPPER='0.5',
                 R_FP='-2.0', FN_BASE='-2.0', FN_PER='0', FN_ONSET_MULT='1',
                 HOVER_DWELL='1', NSTEP_OFF='1', BUFFER_SIZE='50000', GAMMA='0.9', TERMINAL_PEN='0', _NEP='200')
        if cell == 'final3g':
            d.update(SURR_GUST_P='0.5', SURR_GUST_LO='20', SURR_GUST_HI='50', SURR_GUST_G='2.5', SURR_GUST_V='2.0', SURR_GUST_ATK_AFTER='1')
        return d
    if cell == 'final1s':   # ★09-16 정상(비변화) 대조군: final1 과 바람 '주변분포'는 같고 처음부터 고정 = 비정상성만 제거
        return dict(SURR_TIER_P='0:0.525,6:0.18,7:0.205,10:0.09',
                    ATK_BURSTS='3', R_FP='-2.0', FN_BASE='-2.0', FN_PER='0', FN_ONSET_MULT='1',
                    HOVER_DWELL='1', NSTEP_OFF='1', BUFFER_SIZE='50000', GAMMA='0.9', _NEP='200')
    if cell == 'gmixk4': return dict(CH.stage_mix('g90n3', 4), _NEP='60', **GUST)   # 09-16 09:25 사용자 요구 순서(SW≫칼만-TD≫Adam) 겨냥: 돌풍(Adam 취약) + 벌점 ×4 초기 구간(칼만-TD 취약)
    if cell == 'mixk4': return dict(CH.stage_mix('g90n3', 4), _NEP='60')       # 조건 고정 + FP·FN 벌점 ×4 (초기 효율, 탐색 4순위, 60 ep)
    if cell == 'dual200': return dict(CH.stage_blk('g90n3', 1), _NEP='200', **GUST)   # 09-16 08:05: 긴 지평 표본 보강(150–199 창)
    if cell == 'dual300': return dict(CH.stage_blk('g90n3', 1), _NEP='300', **GUST)   # gblk 300 ep (칼만-TD P 누적, 탐색 3순위)
    if cell == 'blk10k':
        return dict(CH.stage_blk('g90n3', 1), BUFFER_SIZE='10000')
    if cell == 'mix':
        return CH.stage_mix('g90n3', 1)
    if cell == 'blk':   # 배치 64 절제용: tonight blk·S1 g90n3 와 같은 dict(배치 128 짝 재사용)
        return CH.stage_blk('g90n3', 1)
    raise ValueError(cell)
SWG = {'P': 'RHUKF_PINIT', 'R': 'RHUKF_R', 'A': 'RHUKF_ALPHA', 'N': 'RHUKF_N'}
def ext_learner(l):
    if l.startswith('SWg'):   # ★09-16 튜닝 격자: SWg_P0.03_R1_A0.1_N7
        d = dict(CH.SWB)
        for tok in l.split('_')[1:]:
            d[SWG[tok[0]]] = tok[1:]
        return d
    if l.startswith('Adam') and l.endswith('ams'): return dict(CH.ADAMP, ADAM_LR=l[4:-3], ADAM_AMSGRAD='1')
    return CH.learner_env(l, c='g90n3')
def ext_jobs(js, qset):
    """09-16 새벽 자동 연장. EXT_SW(선택된 SWIRL 설정 이름), EXT_ADAM(튜닝 Adam 이름). GPU 우선순위 순, CPU 는 Adam 전부."""
    SWs = os.environ.get('EXT_SW', 'SW'); ADs = os.environ.get('EXT_ADAM', 'Adam1e-3'); F = list(range(101, 111)); T5 = list(range(42, 47)); T3 = [42, 43, 44]
    adams = list(dict.fromkeys(['Adam3e-4', ADs]))
    if qset == 'ext_G':   # 09-16 04:40 결정(EXT_DECISION_0916.md): 새 풀 blk 칼만-TD 에서 SW>KTD 방향(진입 오탐 3/5·4/5) → 돌풍+급변 우선. 긴 작업 먼저.
        K = ['UKFp03', 'EKFp03']; T2 = [42, 43]; T44 = [44, 45, 46]
        plan = [(T2, 'dual300', [SWs] + K), (T44, 'gblk', [SWs] + K), (T5, 'mixk4', [SWs] + K), (T3, 'gmix', [SWs] + K)]   # gblk 42–43 = dual300 앞 150 ep(같은 stage·ε 스텝 기준)
        cpu = [(T5, 'gblk'), (T2, 'dual300'), (T5, 'mixk4'), (T5, 'gmix')]
    elif qset == 'final3':   # 1시드 빠른 확인: 돌풍 off/on × (GPU: SW·UKF·EKF / CPU: Adam 2종·SGD)
        out = []
        for cn in ('final3', 'final3g'):
            if js == 'gpu':
                for l in ('SW', 'SWg_P0.05_R1_N6', 'UKFp03', 'EKFp03'): out.append((f'{cn}_{l}_s42', ext_learner(l), 'rhukf', 42, ext_stage(cn, 42), f'x{cn}', l))
            else:
                for l in ('Adam3e-4', 'Adam1e-3', 'SGD1e-2'): out.append((f'{cn}_{l}_s42', ext_learner(l), 'adam', 42, ext_stage(cn, 42), f'x{cn}', l))
        return out
    elif qset == 'obsn':   # ★09-16 사용자 가설: 관측 스케일. 현재 NIS 관측은 [0,3] 전부 양수(gyro 28%>1, 10.5% 클립포화).
        # CartPole/LL 은 0 중심 [-1,1]. SWIRL 의 UKF 갱신은 관측 크기에 직접 의존(H∝입력, 이득 = PH'/(HPH'+R), R=1 고정)
        # → Adam(2차 모멘트 정규화로 스케일 불변)과 달리 SWIRL 만 스케일에 취약할 수 있다.
        # 무대 2종(final1=직전 프레임워크, final3=현 프레임워크) × 관측 2종(n1=[0,1], c1=[-1,1]) × 학습기(GPU:SWIRL / CPU:Adam 2종)
        # 기준선([0,3])은 final1_*, final3_* 로 이미 존재 → 재실행하지 않는다.
        out = []
        # n1/c1 = 스케일만 바꿈(포화 3.0 유지) · n4/c4 = 상한 4.0 으로 포화까지 제거(풀 캡처 상한과 동일)
        VAR = {'n1': dict(SURR_OBS_DIV='3.0'),
               'c1': dict(SURR_OBS_DIV='3.0', SURR_OBS_CENTER='1'),
               'n4': dict(SURR_OBS_DIV='4.0', SURR_OBS_CLIP='4.0'),
               'c4': dict(SURR_OBS_DIV='4.0', SURR_OBS_CLIP='4.0', SURR_OBS_CENTER='1')}
        for cn in ('final1', 'final3'):
            for vk, vv in VAR.items():
                st = dict(ext_stage(cn, 42)); st.update(vv)
                ls = ('SW',) if js == 'gpu' else ('Adam1e-3', 'Adam3e-4')
                for l in ls:
                    out.append((f'obsn_{cn}_{vk}_{l}_s42', ext_learner(l), 'rhukf' if l == 'SW' else 'adam', 42, st, f'xobsn_{cn}_{vk}', l))
        return out
    elif qset == 'final5':   # ★09-16 final5(밴드 0.72-0.80) + final51(바람 ws0/6/7 균일) × 관측 기본//4, 시드 42
        out = []
        G = ('SW', 'UKFp03', 'EKFp03'); C = ('Adam1e-3', 'Adam3e-4', 'SGD1e-2')
        GN = ('SW',); CN = ('Adam1e-3', 'Adam3e-4')
        if js == 'cpu':   # ★09-16 사용자 즉시 판정: final5th(종단벌점 -20 + 밴드 0.72-0.84) Adam 2종
            for l in ('Adam1e-3', 'Adam3e-4'):
                out.append((f'final5th_{l}_s42', ext_learner(l), 'adam', 42, ext_stage('final5th', 42), 'xfinal5th', l))
        for cn in ('final5', 'final51', 'final5t'):
            for l in (G if js == 'gpu' else C):
                out.append((f'{cn}_{l}_s42', ext_learner(l), 'rhukf' if js == 'gpu' else 'adam', 42, ext_stage(cn, 42), f'x{cn}', l))
        for cn in ('final5n', 'final51n'):
            for l in (GN if js == 'gpu' else CN):
                out.append((f'{cn}_{l}_s42', ext_learner(l), 'rhukf' if js == 'gpu' else 'adam', 42, ext_stage(cn, 42), f'x{cn}', l))
        return out
    elif qset == 'retune':   # ★09-17 P0 다중시드 재튜닝 — 1시드 스크린이 1번(튜닝 손해)·2번(2/5 붕괴)에서 결론을 뒤집음
        # per5n1c4 무대, P0 4점 x 시드 3개(42,43,45) = 12런. R2·N7·a0.1 고정(1차 최적)
        out = []
        if js == 'gpu':
            for P in ('0.003', '0.01', '0.03', '0.10'):
                for sd in (42, 43, 45):
                    l = f'SWg_P{P}_R2_A0.1_N7'
                    out.append((f'retune_{l}_s{sd}', ext_learner(l), 'rhukf', sd, ext_stage('per5n1c4', sd), 'xretune', l))
        return out
    elif qset == 'main1':   # ★09-17 1번 온라인RL 본선 — per5n1c4 무대, 튜닝 SWIRL(P0.03·R2·N7·a0.1) + 칼만-TD
        # Adam 4종은 per5n1c4 에서 시드 42-46 이미 완료 → 재실행 불필요
        out = []
        if js == 'gpu':
            for sd in (42, 43, 44, 45, 46):
                for l in ('SWg_P0.03_R2_A0.1_N7', 'EKFp03', 'UKFp03'):
                    out.append((f'main1_{l}_s{sd}', ext_learner(l), 'rhukf', sd, ext_stage('per5n1c4', sd), 'xmain1', l))
        return out
    elif qset == 'tune1':   # ★09-16 1번(온라인 RL) SWIRL 튜닝 — per5n1c4 무대(클립제거+/4), 시드 42, 좌표 스캔
        out = []
        combos = ([f'SWg_P{p}_R1_A0.1_N7' for p in ('0.01','0.03','0.10','0.30')] +
                  [f'SWg_P0.03_R{r}_A0.1_N7' for r in ('2','3','5')] +
                  [f'SWg_P0.03_R1_A0.1_N{n}' for n in ('5','6')] +
                  ['SWg_P0.03_R1_A0.5_N7'])
        if js == 'gpu':
            for l in combos:
                out.append((f'tune1_{l}_s42', ext_learner(l), 'rhukf', 42, ext_stage('per5n1c4', 42), 'xtune1', l))
        return out
    elif qset == 'per5c4':   # ★09-16 per5n1c4 × 시드 42-46 (GPU: SWIRL / CPU: Adam 4종)
        out = []
        for sd in (42, 43, 44, 45, 46):
            if js == 'gpu':
                out.append((f'per5n1c4_SW_s{sd}', ext_learner('SW'), 'rhukf', sd, ext_stage('per5n1c4', sd), 'xper5n1c4', 'SW'))
            else:
                for l in ('Adam3e-4ams', 'Adam3e-4', 'Adam1e-3ams', 'Adam1e-3'):
                    out.append((f'per5n1c4_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage('per5n1c4', sd), 'xper5n1c4', l))
        return out
    elif qset == 'per5c':   # ★09-16 per5n1c × 시드 42-46 (GPU: SWIRL / CPU: Adam 4종)
        out = []
        seeds = [42, 43, 44, 45, 46]
        for sd in seeds:
            if js == 'gpu':
                out.append((f'per5n1c_SW_s{sd}', ext_learner('SW'), 'rhukf', sd, ext_stage('per5n1c', sd), 'xper5n1c', 'SW'))
            else:
                for l in ('Adam3e-4ams', 'Adam3e-4', 'Adam1e-3ams', 'Adam1e-3'):
                    out.append((f'per5n1c_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage('per5n1c', sd), 'xper5n1c', l))
        return out
    elif qset == 'dwell':   # ★09-16 확약 축 검증: final4 무대 × 확약 {1,5,10} × 시드 42-44 × (GPU: SWIRL / CPU: Adam 2종)
        out = []
        for cn in ('final4d5', 'final4d10', 'final4d1'):
            for sd in (42, 43, 44):
                if cn == 'final4d1' and sd == 42: continue   # final4_*_s42 가 동일 설정으로 이미 존재
                if js == 'gpu':
                    out.append((f'{cn}_SW_s{sd}', ext_learner('SW'), 'rhukf', sd, ext_stage(cn, sd), f'x{cn}', 'SW'))
                else:
                    for l in ('Adam1e-3', 'Adam3e-4'):
                        out.append((f'{cn}_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage(cn, sd), f'x{cn}', l))
        return out
    elif qset == 'final4':   # ★09-16 확정: final4=전 학습기 · final4n(/4 정규화)=SWIRL+Adam 만, 시드 42 단일
        out = []
        G4 = ('SW', 'UKFp03', 'EKFp03'); C4 = ('Adam1e-3', 'Adam3e-4', 'SGD1e-2')
        for l in (G4 if js == 'gpu' else C4):
            out.append((f'final4_{l}_s42', ext_learner(l), 'rhukf' if js == 'gpu' else 'adam', 42, ext_stage('final4', 42), 'xfinal4', l))
        for l in (('SW',) if js == 'gpu' else ('Adam1e-3', 'Adam3e-4')):
            out.append((f'final4n_{l}_s42', ext_learner(l), 'rhukf' if js == 'gpu' else 'adam', 42, ext_stage('final4n', 42), 'xfinal4n', l))
        return out
    elif qset == 'tuneq':   # ★09-16 빠른 P₀ 스크린: P{0.01,0.03,0.05,0.1,0.2} × R1 × N6, 확정 비정상 무대·시드 42·200판 (기준선 SW = P0.03·R1·N7)
        out = []
        if js == 'gpu':
            for P in ('0.01', '0.03', '0.05', '0.1', '0.2'):
                nm = f'SWg_P{P}_R1_N6'
                out.append((f'tuneq_{nm}_s42', ext_learner(nm), 'rhukf', 42, ext_stage('final1', 42), 'xfinal1', nm))
        return out
    elif qset == 'tune48':   # ★09-16 SWIRL 튜닝: P₀3 × R4 × α2 × N2 = 48조합 × 시드 42·43 × 100판, 무대는 확정 비정상(final1)
        out = []
        combos = [(P, R, A, Nn) for P in ('0.01', '0.03', '0.1') for R in ('0.5', '1', '1.5', '2') for A in ('0.1', '0.5') for Nn in ('5', '7')]
        if js == 'gpu':
            for (P, R, A, Nn) in combos:
                nm = f'SWg_P{P}_R{R}_A{A}_N{Nn}'
                for sd in (42, 43):
                    st = dict(ext_stage('final1', sd)); st['_NEP'] = '100'
                    out.append((f'tune48_{nm}_s{sd}', ext_learner(nm), 'rhukf', sd, st, 'xtune48', nm))
        return out
    elif qset == 'final1':   # 1시드 확인용: GPU 3 + CPU 4
        out = []
        for cn in ('final1', 'final1s'):   # 비정상(점진 확대) → 정상 대조(같은 주변분포·고정)
            for sd in [42]:
                if js == 'gpu':
                    for l in ('SW', 'UKFp03', 'EKFp03'): out.append((f'{cn}_{l}_s{sd}', ext_learner(l), 'rhukf', sd, ext_stage(cn, sd), f'x{cn}', l))
                else:
                    for l in ('Adam3e-4', 'Adam1e-3', 'SGD1e-3', 'SGD1e-2'): out.append((f'{cn}_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage(cn, sd), f'x{cn}', l))
        return out
    elif qset == 'ext_R':   # 09-16 14:00 주장별 스트레스 스윕(차분-in-차분용). 기준선은 ext_D 의 per5_D5i(같은 확약 5·계측 포함).
        #   잡음 강건성: 관측 σ 0.15 / 0.30.  aliasing 강건성: 돌풍 배율 ×1.75 / ×2.5(라벨 clean, 평시 관측만 부풀림).
        G = lambda g, v: dict(SURR_GUST_P='0.5', SURR_GUST_LO='20', SURR_GUST_HI='50', SURR_GUST_G=g, SURR_GUST_V=v, SURR_GUST_ATK_AFTER='1')
        cells = [('per5_N15', dict(SURR_OBS_NOISE='0.15')), ('per5_N30', dict(SURR_OBS_NOISE='0.30')),
                 ('per5_G175', G('1.75', '1.4')), ('per5_G250', G('2.5', '2.0'))]
        out = []
        for cn, extra in cells:
            for sd in D3 if False else [42, 43, 44]:
                st = dict(ext_stage('per5', sd), **extra)
                if js == 'gpu':
                    for l in [SWs, 'UKFp03', 'EKFp03']: out.append((f'ext_R_{cn}_{l}_s{sd}', ext_learner(l), 'rhukf', sd, dict(st), f'x{cn}', l))
                else:
                    for l in ('Adam3e-4', 'Adam3e-4ams', 'Adam1e-3', 'Adam1e-3ams'): out.append((f'ext_R_{cn}_{l}_s{sd}', ext_learner(l), 'adam', sd, dict(st), f'x{cn}', l))
        return out
    elif qset == 'ext_D':   # 09-16 10:30 사용자 질문: 부트스트랩을 키우지 않고 가치함수의 몫만 키우기 — ① n-step 끄기(n1) ② 호버 확약 D 5→10(선택이 10 스텝을 묶으므로 Q 가 확약 구간을 통합해야 한다) ③ 둘 다
        D3 = [42, 43, 44]; K = ['UKFp03', 'EKFp03']
        # 09-16 11:15 사용자 질문("확약을 없애면 SWIRL 우위가 남나?"): per5_D1 = 확약 제거(HOVER_DWELL=1 → 강제 스텝 0). 이게 서면 확약 회계 문제 전체가 사라지므로 맨 앞에 둔다.
        cells = [('per5_D1', dict(HOVER_DWELL='1')), ('per5_n1', dict(NSTEP_OFF='1')), ('per5_D10', dict(HOVER_DWELL='10')), ('per5_D10n1', dict(NSTEP_OFF='1', HOVER_DWELL='10'))]
        cells.insert(2, ('per5_D5i', dict(HOVER_DWELL='5')))   # 09-16 13:45 사용자 지적: 확약 5 기존 런은 강제 스텝 계측 전이라 D 간 비교가 회계에 오염. 같은 설정을 새 계측으로 재실행해 기준선을 만든다.
        cells += [('per5_D3', dict(HOVER_DWELL='3'))]   # 09-16 12:40: D1 은 칼만-TD 격차만, D5 는 Adam 격차만 선다 → 중간값 확인
        SEEDS = {'per5_D1': [42, 43, 44, 45, 46], 'per5_D5i': [42, 43, 44, 45, 46], 'per5_D3': [42, 43, 44, 45, 46]}
        out = []
        for cn, extra in cells:
            for sd in SEEDS.get(cn, D3):
                st = dict(ext_stage('per5', sd), **extra)
                if js == 'gpu':
                    for l in [SWs] + K: out.append((f'ext_D_{cn}_{l}_s{sd}', ext_learner(l), 'rhukf', sd, dict(st), f'x{cn}', l))
                else:
                    for l in ('Adam3e-4', 'Adam3e-4ams', 'Adam1e-3', 'Adam1e-3ams'): out.append((f'ext_D_{cn}_{l}_s{sd}', ext_learner(l), 'adam', sd, dict(st), f'x{cn}', l))
        return out
    elif qset == 'ext_E':   # 09-16 09:35 사용자 질문: 배치 64 에서 eps decay 를 늘리면(4000→8000 스텝) 배치 64 손해가 회복되는가. 배치 128·eps4000 과 배치 64·eps4000 은 이미 있다.
        out = []
        base = lambda sd: dict(ext_stage('blk', sd), BATCH_SIZE='64', EPS_DECAY='8000', _NEP='110')
        if js == 'gpu':
            for sd in (42, 43): out.append((f'ext_E_b64eps8k_{SWs}-N14_s{sd}', CH.learner_env(f'{SWs}-N14', c='g90n3'), 'rhukf', sd, base(sd), 'b64eblk', f'{SWs}-N14'))
        else:
            for sd in (42, 43):
                for l in ('Adam3e-4', 'Adam3e-4ams', 'Adam1e-3', 'Adam1e-3ams'): out.append((f'ext_E_b64eps8k_{l}_s{sd}', ext_learner(l), 'adam', sd, base(sd), 'b64eblk', l))
        return out
    elif qset == 'ext_P':   # 09-16 09:30: per5 에서 SW − Adam(AMSGrad on) 이 5/5·t 6.06 으로 기준 통과 → 그 무대의 칼만-TD 를 채워 순서(SW ≫ 칼만-TD ≫ Adam)를 확인
        plan = [(T5, 'per5', ['UKFp03', 'EKFp03']), (T5, 'per20', ['UKFp03', 'EKFp03'])]
        cpu = []
    elif qset == 'ext_K':   # 돌풍 + 벌점 ×4, 60 ep, 시드 42–46. CPU 는 Adam off·on 네 가지를 모두 돌린다(새 셀이라 off 자료가 없다)
        out = []
        if js == 'gpu':
            for sd in T5:
                for l in [SWs, 'UKFp03', 'EKFp03']: out.append((f'ext_K_gmixk4_{l}_s{sd}', ext_learner(l), 'rhukf', sd, ext_stage('gmixk4', sd), 'xgmixk4', l))
        else:
            for sd in T5:
                for l in ('Adam3e-4', 'Adam1e-3', 'Adam3e-4ams', 'Adam1e-3ams'): out.append((f'ext_K_gmixk4_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage('gmixk4', sd), 'xgmixk4', l))
        return out
    elif qset == 'ext_ams':   # 09-16 09:10 사용자 결정: 헤드라인 Adam 비교군을 AMSGrad on 으로 → on 런이 없는 무대에 CPU 보강(중요한 무대 먼저)
        AM = ['Adam3e-4ams', 'Adam1e-3ams']; plan = []
        cpu = [(T5, 'per5'), (T5, 'per20'), (T5, 'gblk'), (T5, 'mixk4'), (T5, 'gmix'), ([42, 43], 'dual300'), ([44, 45], 'dual200')]
        out = []
        if js == 'cpu':
            for seeds, cell in cpu:
                for sd in seeds:
                    for l in AM: out.append((f'ext_ams_{cell}_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage(cell, sd), f'x{cell}', l))
            for sd in (42, 43):     # 배치 64 절제의 AMSGrad on 짝
                for l in AM:
                    st = dict(ext_stage('blk', sd)); st.pop('_NEP', None); st.update(BATCH_SIZE='64', _NEP='110')
                    out.append((f'b64_blk_{l}_s{sd}', ext_learner(l), 'adam', sd, st, 'b64blk', l))
        return out
    elif qset == 'ext_H':   # 09-16 08:05: dual300 에서 후반 SW > 칼만-TD(공분산 누적)가 2/2 방향 → 200 ep 시드 44–45 로 n=4 보강
        plan = [([44, 45], 'dual200', [SWs, 'UKFp03', 'EKFp03'])]
        cpu = [([44, 45], 'dual200')]
    elif qset == 'ext_A':
        plan = [(F, 'per20', [SWs]), (F, 'mix', [SWs]), (T5, 'per10', [SWs]), (T5, 'mixJ', [SWs]), (T5, 'perJ20', [SWs]), (T3, 'per20', ['UKFp03', 'EKFp03', 'UKFp1', 'EKFp1']), (F, 'per5', [SWs])]   # 09-16 00:00: 칼만-TD P₀ 0.03·0.1 튜닝 포함
        cpu = [(F, 'per20'), (F, 'mix'), (T5, 'per10'), (T5, 'mixJ'), (T5, 'perJ20'), (F, 'per5')]
    else:
        plan = [(T5, 'per20', ['UKFp03', 'EKFp03', 'UKFp1', 'EKFp1']), (T5, 'blk10k', ['SW', 'UKFp03', 'EKFp03']), (T5, 'per20', ['SWp01', 'SWp3', 'SWn3', 'SWn10'])]
        cpu = [(T5, 'blk10k')]
    out = []
    if js == 'gpu':
        for seeds, cell, ls in plan:
            for sd in seeds:
                for l in ls: out.append((f'{qset}_{cell}_{l}_s{sd}', ext_learner(l), 'rhukf', sd, ext_stage(cell, sd), f'x{cell}', l))
    else:
        for seeds, cell in cpu:
            for sd in seeds:
                for l in adams: out.append((f'{qset}_{cell}_{l}_s{sd}', ext_learner(l), 'adam', sd, ext_stage(cell, sd), f'x{cell}', l))
    return out
def b64_jobs(js):
    """09-16 08:55 배치 64 절제(사용자 00:40): 확정 무대 B64_CELL(ext_stage 셀 이름) × 시드 B64_SEEDS(기본 42,43) × BATCH_SIZE 64, 에피소드 B64_NEP(기본 110).
       SWIRL 은 B64_SW(선택 설정)에 창 N 을 2배(창 표본 B·N 동일), 칼만-TD 는 B64_KTD(쉼표, 기본 UKFp03,EKFp03, N=1 그대로), CPU 는 Adam3e-4 와 EXT_ADAM.
       배치 128 대조는 같은 셀·시드의 기존 런(에피소드 앞부분)을 쓴다."""
    cell = os.environ.get('B64_CELL', 'per20'); SWs = os.environ.get('B64_SW', os.environ.get('EXT_SW', 'SW')); nep = os.environ.get('B64_NEP', '110')
    seeds = [int(x) for x in os.environ.get('B64_SEEDS', '42,43').split(',')]; ktd = os.environ.get('B64_KTD', 'UKFp03,EKFp03').split(',')
    base = ext_stage(cell, seeds[0]) if False else None
    nstar = int(CH.learner_env(SWs, c='g90n3')['RHUKF_N']); out = []
    for sd in seeds:
        st = dict(ext_stage(cell, sd)); st.pop('_NEP', None); st.update(BATCH_SIZE='64', _NEP=nep)
        if js == 'gpu':
            out.append((f'b64_{cell}_{SWs}-N{2 * nstar}_s{sd}', CH.learner_env(f'{SWs}-N{2 * nstar}', c='g90n3'), 'rhukf', sd, dict(st), f'b64{cell}', f'{SWs}-N{2 * nstar}'))
            for l in ktd: out.append((f'b64_{cell}_{l}_s{sd}', ext_learner(l), 'rhukf', sd, dict(st), f'b64{cell}', l))
        else:
            for l in dict.fromkeys(['Adam3e-4', os.environ.get('EXT_ADAM', 'Adam1e-3')]):
                out.append((f'b64_{cell}_{l}_s{sd}', ext_learner(l), 'adam', sd, dict(st), f'b64{cell}', l))
    return out
def jobs(js):
    out = []
    if js == 'gpu':
        for sd in TH.SEEDS:
            for cell in ('per5', 'per20'):
                out.append((f'{cell}_SW_s{sd}', CH.learner_env('SW', c='g90n3'), 'rhukf', sd, pers_env(sd, cell)[0], cell, 'SW'))
        # 09-16 00:25 탐색 워크플로 1순위 전제: 새 풀 칼만-TD 런 0 → 첫 노출 급변 blk 에 UKF/EKF-TD × 42–46 (tonight blk SW·S1 Adam 과 같은 stage dict, 짝)
        for sd in TH.SEEDS:
            for l in ('UKFp03', 'EKFp03'):
                out.append((f'blk_{l}_s{sd}', CH.learner_env(l, c='g90n3'), 'rhukf', sd, TH.stage('blk', sd), 'blk', l))
        out += TH.build_tune('gpu')
        # 09-15 23:45: γ0.97·n-step1 보강은 재튜닝 뒤로 — 시드 42 짝에서 SW 보상 우위가 오탐 감소에서만 나오고 후기 recall .78·지연 3.1 스텝으로 탐지 목표와 어긋남 → 본 프레임워크가 아닌 부록 분석용
        for sd in (43, 44, 45, 46):
            out.append((f'g97n1_SW_s{sd}', CH.learner_env('SW', c='g97n1'), 'rhukf', sd, CH.stage_blk('g97n1', 1), 'g97n1', 'SW'))
    else:
        for sd in TH.SEEDS:
            for cell in ('per5', 'per20'):
                for l in ('Adam3e-4', 'Adam1e-3'):
                    out.append((f'{cell}_{l}_s{sd}', CH.learner_env(l, c='g90n3'), 'adam', sd, pers_env(sd, cell)[0], cell, l))
    return out
def claim(name):
    try: fd = os.open(f'{Q}/claims/{name}', os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError: return False
if __name__ == '__main__':
    mode = sys.argv[1]; js = os.environ.get('JOBSET', 'gpu')
    if mode == 'sched':
        for cell in ('per5', 'per20'):
            for sd in TH.SEEDS:
                _, seq = pers_env(sd, cell); ent = [i for i in range(1, 150) if seq[i] and not seq[i - 1]] + ([0] if seq[0] else [])
                print(f'{cell} seed {sd}: 강풍 비중 {np.mean(seq):.2f}, 강풍 진입 {len(ent)}회 {sorted(ent)[:8]}')
    elif mode == 'count':
        for j in ('gpu', 'cpu'): print(j, len(jobs(j)), [x[0] for x in jobs(j)[:4]])
    elif mode == 'work':
        print(f'[queue {js} pid {os.getpid()}] pool={CH.POOL}', flush=True)
        while True:
            QSET = os.environ.get('QSET', '')
            if not QSET.startswith('b64') and QSET not in ('ext_ams', 'ext_K', 'ext_P', 'ext_E', 'ext_D', 'ext_R', 'final1', 'tune48', 'tuneq', 'final3', 'obsn', 'final4', 'final5', 'dwell', 'per5c', 'per5c4', 'tune1', 'main1', 'retune') and os.path.exists(f'{Q}/STOP_CLAIMS'):   # ext_ams = CPU 전용 AMSGrad on 보강(09-16 09:20), STOP_CLAIMS 예외
                print(f'[queue {js} pid {os.getpid()}] STOP_CLAIMS — 새 작업 선점 중지', flush=True); break
            todo = [j for j in (b64_jobs(js) if QSET.startswith('b64') else (ext_jobs(js, QSET) if QSET else jobs(js))) if not os.path.exists(f'{Q}/{j[0]}.json') and not os.path.exists(f'{Q}/claims/{j[0]}')]
            if not todo: break
            job = todo[0]
            if not claim(job[0]): continue
            nep = job[4].pop('_NEP', None); CH.NEP = int(nep) if nep else 150
            try:
                m = CH.run(job); json.dump({job[0]: m}, open(f'{Q}/{job[0]}.json.tmp', 'w'), default=float); os.replace(f'{Q}/{job[0]}.json.tmp', f'{Q}/{job[0]}.json')
            except Exception as ex:
                import traceback; traceback.print_exc(); print(f'  !! {job[0]} 실패: {ex}', flush=True)
        print(f'[queue {js} pid {os.getpid()}] 할 일 없음 — 종료', flush=True)
