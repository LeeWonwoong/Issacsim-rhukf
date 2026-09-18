# -*- coding: utf-8 -*-
"""surrogate 러너 v2 — Isaac online_rl_main 루프 정합 재현.
   ★핵심 정합(v1 버그 수정):
     ① 관측 정규화 /OBS_DIV(=3.5) — Isaac OBS_NORM=1 과 동일([0,1])
     ② eps = agent.get_epsilon() (step기반 exp(-steps/eps_decay)) — Isaac 과 동일
     ③ reward 호출 = calculate_reward(prev_action, atk_now, min(delay,5), fp, rc) + crash 시 terminal
     ④ off-by-one(reward=prev_action vs 현 atk) / push(prev_s, prev_a, r, s, term) 순서 동일
   관측빌더(ObsBuilder)로 raw4(Isaac) vs EMA기반 소프레임 비교 → 빠른(1~2스텝) 탐지 설계 검증.
"""
import os, sys, json, time, collections
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'etc/scripts')
from surrogate_env import SurrogateEnv
if os.environ.get('SURR_V4','') == '1':      # 2026-09-02 FROZEN-ENV v2 empirical 풀 버전
    from surrogate_env4 import SurrogateEnv
if os.environ.get('SURR_V5','') == '1':      # 2026-09-04 Isaac 학습로그 풀 — 인공노이즈 불필요
    from surrogate_env5 import SurrogateEnv
if os.environ.get('SURR_V51','') == '1':     # 2026-09-04 env5 + 시간 자기상관(가우시안 코퓰러)
    from surrogate_env51 import SurrogateEnv
if os.environ.get('SURR_V52','') == '1':     # 2026-09-05 env5.1 + POMDP 노브(백색성분·기동aliasing)
    from surrogate_env52 import SurrogateEnv
if os.environ.get('SURR_V6','') == '1':      # 2026-09-07 FROZEN-v3.1: v3풀·행동조건부·ramp절벽
    from surrogate_env6 import SurrogateEnv
if os.environ.get('SURR_V7','') == '1':      # 2026-09-08 최종: 2클래스(약버스트+급작치명) 실측절벽
    from surrogate_env7 import SurrogateEnv
if os.environ.get('SURR_V8','') == '1':      # 2026-09-14 env7 + v5c 풀(바람 티어 추첨·스텔스 실측 빈)
    from surrogate_env8 import SurrogateEnv
from env.reward import calculate_reward, RewardConfig

OBS_CLIP = 3.0   # Isaac 실측: OBS_NORM off → compute_nis_scaled clip=3.0, /OBS_DIV=/1.0 (검증 2026-08-28)


# ══════════════════════════════════════════════════════════════════════
#  관측 빌더 — 스텝당 (v_raw,g_raw,prev_action) → obs 벡터(정규화). None=window 미충전.
#  EMA/persist 는 '지속성'을 소프레임에 압축 → 4프레임 스택 없이 빠른 판별.
# ══════════════════════════════════════════════════════════════════════
class ObsBuilder:
    A_FAST, A_SLOW = 0.5, 0.22          # EMA 계수(빠른/느린 = 온셋/지속 레벨)
    PERSIST_TH, PERSIST_DECAY = 0.9, 0.7   # 지속성 카운터([0,3] 압축 g 기준, spike/attack 포착)

    def __init__(self, mode, div=1.0, center=False, clip=None):
        self.mode = mode
        self.div = div
        self.clip = OBS_CLIP if clip is None else float(clip)   # ★09-16 포화 상한 env 제어(풀은 4.0 으로 캡처됨)
        self.center = center   # ★09-16 True 면 [0,1] 특징을 2x-1 로 옮겨 [-1,1] 중심화(CartPole/LL 관측 규약)
        self.W = {'raw4': 4, 'raw2': 2, 'raw1': 1, 'ema2': 2, 'ema1': 1, 'g4': 4}[mode]   # ★09-14 'g4' = gyro-only [g, action]×4 (Isaac GYRO_ONLY 정합, 8D)
        self.rich = mode.startswith('ema')
        self.nfeat = (5 if mode == 'ema2' else 6 if mode == 'ema1' else 2 if mode == 'g4' else 3)
        self.dimS = self.W * self.nfeat

    def reset(self):
        self.buf = collections.deque(maxlen=self.W)
        self.ema_f = None; self.ema_s = None; self.persist = 0.0

    def push(self, v_raw, g_raw, prev_action):
        v = min(v_raw, self.clip)   # Isaac 정합: 기본 [0,3] 압축 NIS 그대로 (÷ 없음)
        g = min(g_raw, self.clip)
        # 시간통합 특징 갱신(항상)
        self.ema_f = g if self.ema_f is None else self.A_FAST * g + (1 - self.A_FAST) * self.ema_f
        self.ema_s = g if self.ema_s is None else self.A_SLOW * g + (1 - self.A_SLOW) * self.ema_s
        self.persist = self.PERSIST_DECAY * self.persist + (1.0 if g > self.PERSIST_TH else 0.0)
        pn = min(self.persist / 3.0, 1.0)
        a = float(prev_action)
        d = self.div
        # NIS스케일 특징([0,3])만 /div, persist(pn)·action 은 이미 정규화/무차원
        if self.mode == 'ema2':
            frame = [v/d, g/d, self.ema_s/d, pn, a]
        elif self.mode == 'ema1':
            frame = [v/d, g/d, self.ema_f/d, self.ema_s/d, pn, a]
        elif self.mode == 'g4':   # gyro-only ablation
            frame = [g/d, a]
        else:  # raw*
            frame = [v/d, g/d, a]
        if self.center:
            frame = [2.0 * x - 1.0 for x in frame]   # div 로 [0,1] 화된 뒤에만 의미 있음
        self.buf.append(frame)
        if len(self.buf) < self.W:
            return None
        return np.array(self.buf, dtype=np.float32).flatten()


# ══════════════════════════════════════════════════════════════════════
def run_config(env_vars, obs_mode='raw4', reward_over=None, n_ep=100, seed=42, ep_steps=400, crash=False,
               obs_div=1.0, fp_escalate=False, reward_mult=1.0, agent_type='rhukf', on_range=(40,81), off_range=(10,31),
               probe=False, obs_noise=0.0):
    for k, v in env_vars.items():
        os.environ[k] = str(v)
    os.environ['SURROGATE'] = '1'   # OBS_NORM 미설정 = Isaac 격자와 동일([0,3])
    import importlib, swrl_config
    importlib.reload(swrl_config)
    cfg = swrl_config.Config()

    _od = os.environ.get('SURR_OBS_DIV', '')          # ★09-16 관측 스케일 실험: [0,3] 기본, 3 이면 [0,1]
    _oc = os.environ.get('SURR_OBS_CENTER', '0') == '1'   # ★09-16 중심화: [0,1] → [-1,1]
    _ocl = os.environ.get('SURR_OBS_CLIP', '')        # ★09-16 강공격 구간 93~100% 포화 → 상한 완화 실험
    if _od: obs_div = float(_od)
    ob = ObsBuilder(obs_mode, div=obs_div, center=_oc, clip=float(_ocl) if _ocl else None)
    cfg.dimS = ob.dimS
    cfg.window_size = ob.W
    cfg.obs_scale = [1.0] * ob.dimS

    RC = RewardConfig()
    # ★2026-09-12 env 오버라이드(Isaac swrl_config.__post_init__ 와 동일 키): R_TP·R_TN·R_FP·FN_BASE·FN_PER — 보상구조 스캔(v7)용
    for _k, _a in (('R_TP', 'r_tp'), ('R_TN', 'r_tn'), ('R_FP', 'r_fp'), ('FN_BASE', 'fn_base'), ('FN_PER', 'fn_per_step')):
        if os.environ.get(_k): setattr(RC, _a, float(os.environ[_k]))
    if reward_over:
        for k, v in reward_over.items():
            setattr(RC, k, v)

    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    if agent_type=='adam':
        from rl.agent_adam import OnlineAdamAgent as _AG
    else:
        from rl.agent import OnlineRHUKFAgent as _AG
    agent = _AG(cfg)
    rscale = getattr(cfg, 'reward_scale', 1.0)

    # 2026-09-03 obs_noise 는 스칼라 또는 (σ_v, σ_g) 튜플 — Isaac d' 채널별 정합용
    def _theta(ag):   # ★09-16 파라미터 이동량 계측(학습 불변): 잘못된 갱신이 가중치에 남는지 보기 위함
        import torch
        for nm in ('q_net', 'net', 'policy_net', 'online_net', 'model'):
            m = getattr(ag, nm, None)
            if m is not None and hasattr(m, 'parameters'):
                try: return torch.cat([q.detach().reshape(-1) for q in m.parameters()]).clone()
                except Exception: return None
        th = getattr(ag, 'theta', None)
        try: return th.detach().reshape(-1).clone() if th is not None else None
        except Exception: return None
    _th_prev = [None]
    _env_on = os.environ.get('SURR_OBS_NOISE', '')   # ★09-16 잡음 강건성 스윕용 env (기본 미설정 = 기존 동작)
    if _env_on: obs_noise = float(_env_on)
    _sv, _sg = (obs_noise, obs_noise) if np.isscalar(obs_noise) else tuple(obs_noise)
    _nrng = np.random.default_rng(seed + 12345) if (_sv > 0 or _sg > 0) else None
    # ★09-13 비정상성 축(v17): SURR_SHIFT_EP 에피소드부터 관측 노이즈 σ=SURR_SHIFT_NOISE 추가 / NIS 배율 SURR_SHIFT_SCALE 적용 (기본 없음)
    _shift_ep = int(os.environ.get('SURR_SHIFT_EP', '-1') or -1)
    _shift_noise = float(os.environ.get('SURR_SHIFT_NOISE', '0') or 0); _shift_scale = float(os.environ.get('SURR_SHIFT_SCALE', '1') or 1)
    _srng = np.random.default_rng(seed + 777)
    genv = SurrogateEnv(ep_steps=ep_steps, crash=crash, seed=seed + 777, on_range=on_range, off_range=off_range)
    hist = []
    for ep in range(n_ep):
        genv.reset(); ob.reset()
        prev_s = None; prev_a = 0
        tp = fp = fn = tn = 0; wtp = wfn = stp = sfn = 0; epr = 0.0; losses = []; _qs = []; _tv = []; _innov_ep = []; _adapt_ep = []; _nisf_ep = []; _flip_ep = []; det_delay = None; crashed = 0
        pp_a = 0; relapses = 0   # ★2026-09-10 relapse(공격중 hover→track) 카운트/벌점
        cont_fp = 0
        fp_g = tn_g = tp_ag = fn_ag = 0   # ★09-14 돌풍 지표
        # ★2026-09-12 A2 정산 보상 (env REWARD_MODE=A2): 매 스텝 track +p / hover 0; 버스트 OFF 시 −(c0+c·n_track) 분할; 무공격 hover 구간 종료 시 −f·len 분할
        _A2 = os.environ.get('REWARD_MODE', '') == 'A2'
        # ★09-13 A3 '약속 hover'(commitment): hover 선택 시 최소 체류 HOVER_DWELL 스텝 강제(환경 동역학, failsafe 검증기동 시간). 보상 = track: +p − c·1{공격}, hover: 0. FA 비용 = 잃은 진행 D·p (라벨 불요).
        _A3 = os.environ.get('REWARD_MODE', '') == 'A3'
        _DW = int(os.environ.get('HOVER_DWELL', '0') or 0); _dw_left = 0
        _forced = False; prev_forced = False; fp_forced = 0; epr_forced = 0.0   # ★09-16 확약 회계: 환경이 강제한 hover 스텝을 표시해 지표에서 분리(학습 경로는 불변)
        _a3p = float(os.environ.get('A3_P', '0.5')); _a3c = float(os.environ.get('A3_C', '2.0'))
        _p = float(os.environ.get('A2_P', '0.5')); _c = float(os.environ.get('A2_C', '6.0')); _c0 = float(os.environ.get('A2_C0', '0.5')); _f = float(os.environ.get('A2_F', '1.0')); _K = max(1, int(os.environ.get('A2_INSTALL', '5')))
        _ntrack = 0; _hov_len = 0; _hov_atk = False; _prev_atk = False; _queue = []   # queue: 남은 분할 청구 [금액/스텝, 남은 횟수]
        _DK = int(os.environ.get('A2_DELAY_K', '-1') or -1); _dq = []   # ★09-13 진단: ≥0 이면 지연비용 c 를 track 스텝마다 k 스텝 뒤 지급(OFF 청구는 c0 만). -1 = OFF 일괄(기본)
        for t in range(ep_steps):
            v_raw, g_raw, atk = genv.nis(prev_a)
            if _nrng is not None:   # ★2026-09-03 관측 노이즈 주입(LL-6 정합). NIS 는 음수 불가
                v_raw = max(0.0, v_raw + _nrng.normal(0.0, _sv))
                g_raw = max(0.0, g_raw + _nrng.normal(0.0, _sg))
            if _shift_ep >= 0 and ep >= _shift_ep:
                v_raw *= _shift_scale; g_raw *= _shift_scale
                if _shift_noise > 0: v_raw = max(0.0, v_raw + _srng.normal(0.0, _shift_noise)); g_raw = max(0.0, g_raw + _srng.normal(0.0, _shift_noise))
            s = ob.push(v_raw, g_raw, prev_a)
            if s is None:                      # window 미충전 → Isaac 처럼 조기 return
                if genv.step():
                    break
                continue
            adly = genv.attack_delay()
            # FP 연속카운트(에스컬레이션용)
            if (not atk) and prev_a == 1:
                cont_fp += 1
            else:
                cont_fp = 0
            _rel = bool(atk and prev_a == 0 and pp_a == 1)
            if _rel: relapses += 1
            r = calculate_reward(prev_a, atk, min(adly, 5), 0, RC, relapse=_rel)
            if _A3:
                r = (_a3p - (_a3c if atk else 0.0)) if prev_a == 0 else 0.0
            if _A2:
                r = _p if prev_a == 0 else 0.0
                if atk and prev_a == 0:
                    _ntrack += 1
                    if _DK >= 0: _dq.append([-_c, _DK])
                if _prev_atk and not atk:                       # 버스트 OFF → 지연 청구 분할
                    _queue.append([-(_c0 + (_c * _ntrack if _DK < 0 else 0.0)) / _K, _K]); _ntrack = 0
                for q in _dq:
                    if q[1] <= 0: r += q[0]
                    q[1] -= 1
                _dq = [q for q in _dq if q[1] >= 0]
                if prev_a == 1: _hov_len += 1; _hov_atk = _hov_atk or atk
                if pp_a == 1 and prev_a == 0:                    # hover 구간 종료(복귀)
                    if not _hov_atk and _hov_len > 0: _queue.append([-_f * _hov_len / _K, _K])
                    _hov_len = 0; _hov_atk = False
                for q in _queue: r += q[0]; q[1] -= 1
                _queue = [q for q in _queue if q[1] > 0]
                _prev_atk = atk
            if fp_escalate and (not atk) and prev_a == 1:   # FP 에스컬레이션 override
                r = -0.2 - 0.25 * min(cont_fp - 1, 5)       # fp0 -0.2(싼 첫경보) … fp5 -1.45
            r = (r * rscale) * reward_mult
            if genv.crashed:
                r += getattr(RC, 'terminal_penalty', 0.0) * reward_mult; crashed = 1   # 09-11 사용자 확정: TERMINAL_PEN=0 (절단만)
            # confusion (prev_a vs 현 atk) — Isaac 정합
            if atk:
                _dl = float(getattr(genv, 'dl', [0]*ep_steps)[genv.t]) if hasattr(genv, 'dl') else 0.5
                _wk = _dl < 0.35        # 약공격(δ<0.35) / 강공격
                if prev_a == 1:
                    tp += 1
                    if _wk: wtp += 1
                    else: stp += 1
                    if det_delay is None:
                        det_delay = adly
                else:
                    fn += 1
                    if _wk: wfn += 1
                    else: sfn += 1
            else:
                if prev_a == 1:
                    fp += 1
                    if prev_forced: fp_forced += 1   # ★09-16 강제 확약 구간의 오탐(정책이 고른 것이 아님)
                else: tn += 1
            _gs = getattr(genv, 'gust', None)
            if _gs is not None and _gs[min(genv.t, len(_gs) - 1)] and not atk:
                if prev_a == 1: fp_g += 1
                else: tn_g += 1
            if atk and getattr(genv, 'atk_after_gust', False):
                if prev_a == 1: tp_ag += 1
                else: fn_ag += 1
            epr += r
            if prev_forced: epr_forced += r   # ★09-16 강제 확약 스텝이 보상에 기여한 몫(D 비교 공정성)
            if prev_s is not None:
                agent.push(prev_s, prev_a, r, s, bool(genv.crashed))
                out = agent.learn()
                if out is not None and out[0]:
                    losses.append(out[0])
                    if len(out) > 2 and out[2] is not None: _tv.append(float(out[2]))
                    _innov_ep.append(float(getattr(agent, '_last_innov', 0.0) or 0.0)); _adapt_ep.append(float(getattr(agent, '_last_adapt', 0.0) or 0.0)); _nisf_ep.append(float(getattr(agent, '_last_nis', 0.0) or 0.0)); _flip_ep.append(float(getattr(agent, '_last_argmax_flip', 0.0) or 0.0))   # ★09-18 CP 레짐 격자: 필터 NIS·argmax_flip   # ★09-15 갱신 인덱스 단위 혁신(리플레이 방어)
            if (len(losses) & 15) == 0:
                try: _qs.append(float(max(agent.get_q_values(s))))
                except Exception: pass
            eps = agent.get_epsilon()
            a = agent.act(s, eps)
            _forced = False
            if _DW > 0:                                   # 약속 hover: 진입 후 D−1 스텝 동안 hover 강제
                if _dw_left > 0: a = 1; _dw_left -= 1; _forced = True
                elif a == 1 and prev_a == 0: _dw_left = _DW - 1
            prev_s = s; pp_a = prev_a; prev_a = a; prev_forced = _forced
            if genv.step():
                break
        if _A2:   # 에피 종료 정산: 무공격 hover 구간이 열려 있으면 −f·len, 남은 분할 청구 일괄 (마지막 push 이후라 epr 통계에만 반영)
            _tail = sum(q[0] * q[1] for q in _queue) + sum(q[0] for q in _dq) + (0.0 if (_hov_atk or _hov_len == 0) else -_f * _hov_len)
            epr += _tail
        agent.end_episode(epr, t + 1)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        _pe = int(os.environ.get('PROBE_EVERY', '0') or 0)
        _pr = _probe_quick(agent, ob, ep_steps, seed, int(os.environ.get('PROBE_N', '6') or 6), on_range, off_range, obs_noise) if (_pe > 0 and ep % _pe == 0) else {}
        try:
            _th = _theta(agent)
            _wn = float(_th.norm()) if _th is not None else float('nan')
            _ws = float((_th - _th_prev[0]).norm()) if (_th is not None and _th_prev[0] is not None) else float('nan')
            _th_prev[0] = _th
        except Exception:
            _wn = _ws = float('nan')
        hist.append(dict(wnorm=_wn, wstep=_ws, ep=ep, reward=epr, f1=f1, prec=prec, rec=rec, fpr=fpr, fp_forced=fp_forced, reward_forced=epr_forced, fpr_chosen=((fp - fp_forced) / (fp - fp_forced + tn) if (fp - fp_forced + tn) else float('nan')), **_pr, had_gust=int(bool(getattr(genv, 'gust', np.zeros(1)).any())), fpr_gust=(fp_g / (fp_g + tn_g) if fp_g + tn_g else float('nan')), rec_after_gust=(tp_ag / (tp_ag + fn_ag) if tp_ag + fn_ag else float('nan')),
                         delay=(float(det_delay) if det_delay is not None else np.nan),
                         loss=float(np.mean(losses)) if losses else 0.0,
                         qmax=(float(np.max(_qs)) if _qs else None), qavg=(float(np.mean(_qs)) if _qs else None),
                         tvar=(float(np.mean(_tv)) if _tv else None),
                         has_atk=int(tp + fn > 0), crashed=crashed, relapse=relapses, dmax=float(getattr(genv, 'dl', np.zeros(1)).max()),
                         innov=float(np.mean(_innov_ep)) if _innov_ep else 0.0, innov_max=float(np.max(_innov_ep)) if _innov_ep else 0.0, adapt=float(np.mean(_adapt_ep)) if _adapt_ep else 0.0, n_upd=len(_innov_ep), nisf=float(np.mean(_nisf_ep)) if _nisf_ep else 0.0, aflip=float(np.mean(_flip_ep)) if _flip_ep else 0.0, kgain=float(getattr(agent, '_last_kgain', 0.0) or 0.0), pmax=float(getattr(agent, '_last_pmax', 0.0) or 0.0), prep=int(getattr(agent, '_p_repairs', 0) or 0),   # ★09-14 필터 진단(R4)
                         wrec=(wtp/(wtp+wfn) if wtp+wfn>0 else np.nan),
                         srec=(stp/(stp+sfn) if stp+sfn>0 else np.nan)))
    if probe:
        hist_probe = _greedy_probe(agent, ob, RC, rscale, reward_mult, seed, ep_steps, on_range, off_range,
                                   obs_noise=obs_noise)
        return hist, hist_probe
    return hist


def _probe_quick(agent, ob, ep_steps, seed, n, on_range, off_range, obs_noise=0.0):
    """★09-14 에피소드별 greedy(ε=0) 프로브: 정상 샘플러 n 에피(시드 고정), 학습·push 없음, steps_done(ε 스케줄) 복원."""
    import copy as _copy
    if os.environ.get('SURR_V8','') == '1':
        from surrogate_env8 import SurrogateEnv as _E
    elif os.environ.get('SURR_V7','') == '1':
        from surrogate_env7 import SurrogateEnv as _E
    else:
        from surrogate_env5 import SurrogateEnv as _E
    _sv, _sg = (obs_noise, obs_noise) if np.isscalar(obs_noise) else tuple(obs_noise)
    _pn = np.random.default_rng(seed + 777) if (_sv > 0 or _sg > 0) else None
    sd0 = getattr(agent, 'steps_done', None)
    genv = _E(ep_steps=ep_steps, seed=seed + 9000, on_range=on_range, off_range=off_range)
    pob = _copy.deepcopy(ob)
    tp = fp = fn = tn = 0; dels = []; crashes = 0
    for i in range(n):
        genv.reset(); pob.reset(); prev_a = 0; det = None
        for t in range(ep_steps):
            v, g, atk = genv.nis(prev_a)
            if _pn is not None:
                v = max(0.0, v + _pn.normal(0.0, _sv)); g = max(0.0, g + _pn.normal(0.0, _sg))
            s = pob.push(v, g, prev_a)
            if s is not None:
                a = agent.act(s, 0.0)
                if atk:
                    if prev_a == 1:
                        tp += 1
                        if det is None: det = genv.attack_delay()
                    else: fn += 1
                else:
                    if prev_a == 1: fp += 1
                    else: tn += 1
                prev_a = a
            if genv.step(): break
        if det is not None: dels.append(det)
        elif genv.has_atk: dels.append(ep_steps)
        crashes += int(genv.crashed)
    if sd0 is not None: agent.steps_done = sd0
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(probe_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, probe_fpr=fp / (fp + tn) if fp + tn else 0.0,
                probe_rec=rec, probe_delay=float(np.mean(dels)) if dels else float('nan'), probe_crash=crashes)


def _greedy_probe(agent, ob, RC, rscale, reward_mult, seed, ep_steps, on_range, off_range, obs_noise=0.0):
    """학습 후 eps=0 프로브: benign 10 / 약공격 δ{0.1,0.2} 10 / 강공격 δ{0.6,0.7} 10 에피.
       탐험 오염 없는 정책 성능: FP rate(benign), recall/delay(약·강 분리)."""
    import numpy as np
    if os.environ.get('SURR_V7','') == '1':
        from surrogate_env7 import SurrogateEnv as _E
    elif os.environ.get('SURR_V6','') == '1':
        from surrogate_env6 import SurrogateEnv as _E
    elif os.environ.get('SURR_V52','') == '1':
        from surrogate_env52 import SurrogateEnv as _E
    elif os.environ.get('SURR_V51','') == '1':
        from surrogate_env51 import SurrogateEnv as _E
    elif os.environ.get('SURR_V5','') == '1':
        from surrogate_env5 import SurrogateEnv as _E
    else:
        from surrogate_env4 import SurrogateEnv as _E
    _sv, _sg = (obs_noise, obs_noise) if np.isscalar(obs_noise) else tuple(obs_noise)
    _pn = np.random.default_rng(seed + 54321) if (_sv > 0 or _sg > 0) else None
    out = {}
    for tag, deltas, n in [('benign', None, 10), ('weak', [0.1, 0.2], 10), ('strong', [0.6, 0.7], 10)]:
        fps = tns = tps = fns = 0; dels = []
        genv = _E(ep_steps=ep_steps, seed=seed + 9000, on_range=on_range, off_range=off_range)
        for i in range(n):
            genv.reset()
            if deltas is None:
                genv.has_atk = False; genv.atk[:] = False
            else:
                genv.has_atk = True
                d = deltas[i % len(deltas)]
                genv.atk[:] = False; genv.dl[:] = 0; genv.bstart[:] = -1
                t = 40
                while t < ep_steps - 20:
                    e = min(t + 60, ep_steps - 5)
                    genv.atk[t:e] = True; genv.dl[t:e] = d; genv.bstart[t:e] = t
                    t = e + 20
            ob.reset(); prev_a = 0; det = None
            for t in range(ep_steps):
                v, g, atk = genv.nis(prev_a)
                if _pn is not None:
                    v = max(0.0, v + _pn.normal(0.0, _sv))
                    g = max(0.0, g + _pn.normal(0.0, _sg))
                s = ob.push(v, g, prev_a)
                if s is not None:
                    a = agent.act(s, 0.0)
                    if atk:
                        if prev_a == 1:
                            tps += 1
                            if det is None: det = genv.attack_delay()
                        else: fns += 1
                    else:
                        if prev_a == 1: fps += 1
                        else: tns += 1
                    prev_a = a
                if genv.step(): break
            if det is not None: dels.append(det)
        if tag == 'benign':
            out['probe_fp'] = fps / max(fps + tns, 1)
        else:
            out[f'probe_{tag}_rec'] = tps / max(tps + fns, 1)
            out[f'probe_{tag}_delay'] = float(np.median(dels)) if dels else float('nan')
    return out


def summarize(h, tail=20):
    atk = [r for r in h if r['has_atk']]
    last = atk[-tail:] if len(atk) >= tail else atk
    def m(k):
        xs = [r[k] for r in last if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(xs)) if xs else float('nan')
    # 수렴속도: 공격에피 rolling-5 F1 이 0.85 를 처음 지속(2회) 넘는 에피 / 안정성: 마지막 40 공격에피 F1 std
    f1s=[r['f1'] for r in atk]
    conv=np.nan
    for i in range(4,len(f1s)):
        if np.mean(f1s[max(0,i-4):i+1])>=0.85:
            conv=atk[i]['ep']; break
    stab=float(np.std(f1s[-40:])) if len(f1s)>=10 else np.nan
    return dict(F1=m('f1'), P=m('prec'), R=m('rec'), delay=m('delay'), fpr=m('fpr'),
                reward=float(np.mean([r['reward'] for r in h[-10:]])),
                loss=float(np.mean([r['loss'] for r in h[-10:]])),
                crash=float(np.mean([r['crashed'] for r in h[-20:]])),
                conv_ep=float(conv), stab=stab,
                wrec=m('wrec') if 'wrec' in (last[0] if last else {}) else np.nan,
                srec=m('srec') if 'srec' in (last[0] if last else {}) else np.nan)


# 기준 RHUKF config (Isaac set-A: tau0.02·ui4)
BASE_ENV = dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU=0.02, RHUKF_UI=4, RHUKF_R=1.0, RHUKF_PD=0.01)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='calib', choices=['calib', 'obs', 'reward', 'single', 'q234', 'step0'])
    ap.add_argument('--obs', default='raw4')
    ap.add_argument('--nep', type=int, default=100)
    ap.add_argument('--out', default='/tmp/surrogate_out.json')
    args = ap.parse_args()
    ISAAC = dict(F1=0.972, P=0.967, R=0.976, delay=2.1, fpr=0.13, reward=257)

    def line(name, s):
        print(f"  {name:<12s} F1 {s['F1']:.3f}  P {s['P']:.3f}  R {s['R']:.3f}  "
              f"delay {s['delay']:.2f}  fpr {s['fpr']:.3f}  rwd {s['reward']:6.1f}  "
              f"loss {s['loss']:.2f}  crash {s['crash']:.2f}", flush=True)

    out = {}
    if args.mode == 'calib':
        print(f"[calib] raw4 (Isaac baseline 재현) vs 타깃 F1 {ISAAC['F1']} P {ISAAC['P']} R {ISAAC['R']} delay {ISAAC['delay']} fpr {ISAAC['fpr']}")
        t0 = time.time(); h = run_config(BASE_ENV, obs_mode='raw4', n_ep=args.nep)
        s = summarize(h); line('raw4', s); out['raw4'] = h
        print(f"  ({time.time()-t0:.0f}s)  Δ: F1 {s['F1']-ISAAC['F1']:+.3f} P {s['P']-ISAAC['P']:+.3f} delay {s['delay']-ISAAC['delay']:+.2f} fpr {s['fpr']-ISAAC['fpr']:+.3f}")
    elif args.mode == 'obs':
        print("[obs] 관측빌더 비교 (같은 RHUKF config): raw4(Isaac) vs 소프레임 EMA")
        for mode in ['raw4', 'raw2', 'raw1', 'ema2', 'ema1']:
            t0 = time.time(); h = run_config(BASE_ENV, obs_mode=mode, n_ep=args.nep)
            s = summarize(h); line(f'{mode}(d{ObsBuilder(mode).dimS})', s); out[mode] = h
    elif args.mode == 'reward':
        print("[reward] reward 변형 비교 (obs=ema1)")
        variants = {
            'v3_base': {},
            'tp15': dict(r_tp=1.5),
            'fp05': dict(r_fp=-0.5),
            'fp10': dict(r_fp=-1.0),
            'onset25': dict(fn_onset_mult=2.5),
            'tp15_fp10': dict(r_tp=1.5, r_fp=-1.0),
        }
        for name, ro in variants.items():
            t0 = time.time(); h = run_config(BASE_ENV, obs_mode='ema1', reward_over=ro, n_ep=args.nep)
            s = summarize(h); line(name, s); out[name] = h
    elif args.mode == 'q234':
        print("[q234] Q2 관측스케일 / Q3 FP에스컬 / Q4 reward/2 (obs=ema1 기준)")
        exps = [
            ('Q2_ema1_[0,3]', dict(obs_mode='ema1', obs_div=1.0)),
            ('Q2_ema1_[0,1]', dict(obs_mode='ema1', obs_div=3.0)),
            ('Q2_raw4_[0,3]', dict(obs_mode='raw4', obs_div=1.0)),
            ('Q2_raw4_[0,1]', dict(obs_mode='raw4', obs_div=3.0)),
            ('Q3_fp_flat',    dict(obs_mode='ema1', fp_escalate=False)),
            ('Q3_fp_escal',   dict(obs_mode='ema1', fp_escalate=True)),
            ('Q4_rwd_x1',     dict(obs_mode='ema1', reward_mult=1.0)),
            ('Q4_rwd_x0.5',   dict(obs_mode='ema1', reward_mult=0.5)),
        ]
        for name, kw in exps:
            t0 = time.time(); h = run_config(BASE_ENV, n_ep=args.nep, **kw)
            s2 = summarize(h); line(name, s2); out[name] = h
    elif args.mode == 'step0':
        print("[step0] 짧은ON(10,20)/긴OFF(25,40) winnable + RHUKF vs Adam × seed{42,43,44}")
        RHENV=dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3', RHUKF_TAU=0.005, RHUKF_UI=1, RHUKF_R=1.5, RHUKF_PD=0.01)
        ADENV=dict(NET_HIDDEN=16, ADAM_LR='1e-3')
        ONR=(10,21); OFFR=(25,41)
        import numpy as _np
        res={'rhukf':[], 'adam':[]}
        for ag,env in [('rhukf',RHENV),('adam',ADENV)]:
            for sd in [42,43,44]:
                t0=time.time(); h=run_config(env, obs_mode='raw4', n_ep=args.nep, seed=sd, agent_type=ag, on_range=ONR, off_range=OFFR)
                sm=summarize(h); res[ag].append(sm)
                print(f"  {ag:<6s} seed{sd}: F1 {sm['F1']:.3f} P {sm['P']:.3f} R {sm['R']:.3f} delay {sm['delay']:.2f} fpr {sm['fpr']:.3f} rwd {sm['reward']:.1f} ({time.time()-t0:.0f}s)", flush=True)
            out[ag]=h
        print("\n  ── seed 집계 (mean±std) ──")
        for ag in ['rhukf','adam']:
            f1=[r['F1'] for r in res[ag]]; rw=[r['reward'] for r in res[ag]]
            print(f"  {ag:<6s}: F1 {_np.mean(f1):.3f}±{_np.std(f1):.3f} | reward {_np.mean(rw):.1f}±{_np.std(rw):.1f}")
    else:
        t0 = time.time(); h = run_config(BASE_ENV, obs_mode=args.obs, n_ep=args.nep)
        s = summarize(h); line(args.obs, s); out[args.obs] = h
    json.dump(out, open(args.out, 'w'), default=lambda o: float(o))
    print(f"저장: {args.out}")
