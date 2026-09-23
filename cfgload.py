"""cfgload.py — 실험 설정(YAML)을 읽어 모든 모듈이 쓰는 설정 객체로 만든다. env 변수는 쓰지 않는다.

사용
    exp = load_experiment(['configs/cp_regime.yaml'], sets=['reward.scale=0.4', 'agent.swirl.R=2'])
    exp.cfg        swrl_config.Config  (학습기·Isaac 노드가 읽는 평면 설정)
    exp.obs        env.observation.ObsSpec
    exp.reward     env.reward.RewardConfig      (= exp.cfg.reward)
    exp.scenario   env.scenario.ScenarioConfig  (공격·바람·패턴)
    exp.surrogate  sim.surrogate.SurrogateConfig
    exp.run / exp.log / exp.isaac   dict
    exp.resolved   최종 병합 결과(dict) — 결과 폴더에 config.yaml 로 저장해 재현에 쓴다

규칙
  · `extends: 상대경로.yaml` 로 상속(깊은 병합, 리스트는 통째로 교체).
  · `--set a.b.c=값` 은 YAML 문법으로 값을 해석(숫자·리스트·불리언).
  · 스키마에 없는 키는 오류 — 오타가 조용히 무시되는 일을 막는다.
"""
from __future__ import annotations

import copy
import os
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Iterable, List, Optional

import yaml

from env.observation import ObsSpec
from env.reward import RewardConfig
from env.scenario import ScenarioConfig
from sim.surrogate import SurrogateConfig

# ── 스키마: 섹션 → 허용 키 (dataclass 섹션은 필드에서 자동) ─────────────────────────────
_AGENT_KEYS = {'type', 'gamma', 'n_step', 'batch', 'buffer', 'hidden', 'tau', 'update_interval', 'per',
               'hover_dwell', 'eps', 'replay', 'swirl', 'adam'}
_EPS_KEYS = {'start', 'end', 'decay', 'hover_p', 'z_mu', 'z_cap'}
_REPLAY_KEYS = {'mode', 'halflife'}
_SWIRL_KEYS = {'form', 'p_delta', 'p_init', 'huber_c', 'N', 'R', 'q', 'alpha', 'anchor', 'argmax', 'h0', 'spas', 'act', 'eval'}
_ADAM_KEYS = {'lr', 'amsgrad', 'init', 'optimizer', 'loss', 'huber_beta', 'grad_clip'}
_RUN_KEYS = {'name', 'seed', 'episodes', 'ep_steps', 'outdir', 'device'}
_LOG_KEYS = {'probe_every', 'probe_n', 'steps', 'eval_n'}
_ENV_KEYS = {'kind', 'surrogate', 'isaac'}
_ISAAC_KEYS = {'headless', 'speed', 'sim_env', 'launcher', 'compile', 'knobs', 'cfg'}
_TOP = {'extends', 'run', 'env', 'scenario', 'obs', 'reward', 'agent', 'log', 'capture'}


# 병합하지 않고 통째로 교체하는 키 — 확률분포·목록형 dict (하위 설정이 상위 항목을 지울 수 있어야 한다, 09-18 검토)
_REPLACE = {'tiers', 'policies', 'grades', 'grow', 'ploc_tiers', 'dead_tiers'}
_NUM = __import__('re').compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)$')


def _coerce(x):
    """YAML 1.1 은 '1e-3' 을 문자열로 읽는다 → 지수 표기 숫자 문자열을 float 로."""
    if isinstance(x, dict):
        return {k: _coerce(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce(v) for v in x]
    if isinstance(x, str) and _NUM.match(x.strip()):
        return float(x)
    return x


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if k in _REPLACE:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: str, _seen=None) -> dict:
    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f'extends 순환: {path}')
    _seen.add(path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop('extends', None)
    if base:
        d = _deep_merge(_read(os.path.join(os.path.dirname(path), base), _seen), d)
    return d


def _set(d: dict, dotted: str):
    if '=' not in dotted:
        raise ValueError(f'--set 형식은 키=값: {dotted!r}')
    k, v = dotted.split('=', 1)
    cur = d
    parts = k.strip().split('.')
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(v)


def _check(d: dict, allowed: set, where: str):
    bad = set(d or {}) - allowed
    if bad:
        raise KeyError(f'설정 {where} 에 모르는 키 {sorted(bad)} (허용: {sorted(allowed)})')


def _dc_keys(cls) -> set:
    return {f.name for f in fields(cls)}


def _build_dc(cls, d: dict, where: str):
    d = dict(d or {})
    _check(d, _dc_keys(cls), where)
    return cls(**d)


def load_experiment(paths: Iterable[str], sets: Optional[List[str]] = None) -> SimpleNamespace:
    d: dict = {}
    for p in paths:
        d = _deep_merge(d, _read(p))
    for s in sets or []:
        _set(d, s)
    d = _coerce(d)
    _check(d, _TOP - {'extends'}, '최상위')

    run = dict(seed=42, episodes=200, ep_steps=300, outdir='results/claudecodefortest/untitled', device='auto', name='untitled')
    _check(d.get('run'), _RUN_KEYS, 'run'); run.update(d.get('run') or {})
    log = dict(probe_every=1, probe_n=4, steps=False, eval_n=100)   # eval_n: 학습 끝 greedy 최종 평가 에피소드 수(0=끔)   # steps: Isaac 학습 중 스텝 기록(원시 NIS·관측·보상) → <outdir>/steps/
    _check(d.get('log'), _LOG_KEYS, 'log'); log.update(d.get('log') or {})

    obs = _build_dc(ObsSpec, d.get('obs'), 'obs')
    from env.capture import CaptureConfig
    capture = _build_dc(CaptureConfig, d.get('capture'), 'capture')
    reward = _build_dc(RewardConfig, d.get('reward'), 'reward')

    sc = dict(d.get('scenario') or {})
    _check(sc, {'attack', 'wind', 'patterns'}, 'scenario')
    from env.attack import AttackConfig, ProfileClass
    from env.scenario import WindConfig
    atk = dict(sc.get('attack') or {}); _check(atk, _dc_keys(AttackConfig), 'scenario.attack')
    for i, c in enumerate(atk.get('classes') or []):
        _check(c, _dc_keys(ProfileClass), f'scenario.attack.classes[{i}]')
    _check(sc.get('wind'), _dc_keys(WindConfig), 'scenario.wind')
    scenario = ScenarioConfig(attack=atk, wind=dict(sc.get('wind') or {}),
                              **({'patterns': sc['patterns']} if 'patterns' in sc else {}))

    envd = dict(d.get('env') or {}); _check(envd, _ENV_KEYS, 'env')
    kind = envd.get('kind', 'surrogate')
    if kind not in ('surrogate', 'isaac'):
        raise ValueError(f'env.kind={kind!r} (surrogate|isaac)')
    sur = dict(envd.get('surrogate') or {})
    from sim.surrogate import CrashConfig
    _check(sur, _dc_keys(SurrogateConfig), 'env.surrogate')
    _check(sur.get('crash'), _dc_keys(CrashConfig), 'env.surrogate.crash')
    surrogate = SurrogateConfig(**sur)
    isaac = dict(headless=True, speed=1.0, sim_env={}, launcher='isim', compile=True, knobs={}, cfg={})
    _check(envd.get('isaac'), _ISAAC_KEYS, 'env.isaac'); isaac.update(envd.get('isaac') or {})

    # ── 평면 Config (학습기·Isaac 노드) ─────────────────────────────────────────────
    import swrl_config
    cfg = swrl_config.Config()
    ag = dict(d.get('agent') or {}); _check(ag, _AGENT_KEYS, 'agent')
    eps = dict(ag.get('eps') or {}); _check(eps, _EPS_KEYS, 'agent.eps')
    rp = dict(ag.get('replay') or {}); _check(rp, _REPLAY_KEYS, 'agent.replay')
    sw = dict(ag.get('swirl') or {}); _check(sw, _SWIRL_KEYS, 'agent.swirl')
    ad = dict(ag.get('adam') or {}); _check(ad, _ADAM_KEYS, 'agent.adam')

    import torch
    cfg.device = ('cuda' if torch.cuda.is_available() else 'cpu') if run['device'] == 'auto' else run['device']
    cfg.seed = int(run['seed']); cfg.max_episodes = int(run['episodes']); cfg.episode_max_steps = int(run['ep_steps'])
    cfg.outdir = run['outdir']

    atype = ag.get('type', 'swirl')
    if atype not in ('swirl', 'adam', 'ukf', 'ekf'):
        raise ValueError(f'agent.type={atype!r} (swirl|adam|ukf|ekf)')
    cfg.agent_type = 'adam' if atype == 'adam' else 'rhukf'
    cfg.filter_mode = {'swirl': 'rhukf', 'ukf': 'ukf', 'ekf': 'ekf'}.get(atype, 'rhukf')
    M = {'gamma': 'gamma', 'batch': 'batch_size', 'buffer': 'buffer_size', 'tau': 'tau_srrhuif',
         'update_interval': 'update_interval', 'per': 'use_per', 'hover_dwell': 'hover_dwell'}
    for k, a in M.items():
        if k in ag: setattr(cfg, a, ag[k])
    if 'hidden' in ag: cfg.shared_layers = [int(x) for x in ag['hidden']]
    if 'n_step' in ag:
        n = int(ag['n_step']); cfg.use_n_step = n > 1; cfg.n_step_size = max(n, 1)
    if 'start' in eps: cfg.eps_start = float(eps['start'])
    if 'end' in eps: cfg.eps_end = float(eps['end'])
    if 'decay' in eps: cfg.eps_decay_steps = int(eps['decay'])
    if 'hover_p' in eps: cfg.eps_action_probs = [1.0 - float(eps['hover_p']), float(eps['hover_p'])]
    if 'z_mu' in eps: cfg.eps_z_mu = float(eps['z_mu'])
    if 'z_cap' in eps: cfg.eps_z_cap = int(eps['z_cap'])
    if 'mode' in rp: cfg.replay_mode = rp['mode']
    if 'halflife' in rp: cfg.replay_halflife = float(rp['halflife'])
    SW = {'form': 'state_form', 'p_delta': 'p_delta_init', 'p_init': 'p_init', 'huber_c': 'huber_c', 'N': 'N_horizon',
          'alpha': 'alpha', 'anchor': 'anchor_type', 'argmax': 'ddqn_argmax', 'h0': 'h0_online_moving_init', 'spas': 'use_spas',
          'act': 'act_net', 'eval': 'eval_net'}
    for k, a in SW.items():
        if k in sw: setattr(cfg, a, sw[k])
    if 'R' in sw: cfg.r_init = cfg.r_end = float(sw['R'])
    if 'q' in sw: cfg.q_init = cfg.q_end = float(sw['q'])
    AD = {'lr': 'adam_lr', 'amsgrad': 'adam_amsgrad', 'init': 'adam_init', 'optimizer': 'adam_optimizer',
          'loss': 'adam_loss', 'huber_beta': 'adam_huber_beta', 'grad_clip': 'adam_grad_clip'}
    for k, a in AD.items():
        if k in ad: setattr(cfg, a, ad[k])
    cfg.r_inv_sqrt = 1.0 / cfg.r_init; cfg.r_inv = 1.0 / (cfg.r_init ** 2)

    # ★09-23 P2′ 성형 검증: F 의 γ 는 학습기 γ 와 같아야 정책 불변(n-step 망원 포함) · θ 입력이 있어야 한다
    if float(reward.shape_tilt) > 0:
        g = float(cfg.gamma)
        if float(reward.shape_gamma) <= 0:
            reward.shape_gamma = g
        elif abs(float(reward.shape_gamma) - g) > 1e-12:
            raise ValueError(f'reward.shape_gamma={reward.shape_gamma} ≠ agent.gamma={g} — 퍼텐셜 성형은 같은 γ 여야 정책 불변')
        if kind == 'surrogate' and not surrogate.theta:
            raise ValueError('reward.shape_tilt>0 (surrogate) 은 θ 입력이 필요하다 — env.surrogate.theta: true (knn_v4 풀)')

    # 관측·보상은 공용 모듈이 담당 — 평면 Config 에는 차원·보상 객체만 맞춘다(구 reward_scale 는 1 로 고정: 이중 적용 방지)
    cfg.gyro_only = cfg._gyro_only = (obs.features == ['gyro', 'action'])
    cfg.window_size = obs.window; cfg.dimS = obs.dim; cfg.obs_scale = [1.0] * obs.dim
    cfg.reward = reward; cfg.reward_scale = 1.0
    cfg.attack_tq_authority_nm = scenario.attack.authority_nm

    # ── Isaac 전용: 노브(구 env 변수) + Config 필드 직접 설정(구 CLI 플래그: sweep·capture·deadline 등) ──
    from env.knobs import set_knobs
    set_knobs(isaac.get('knobs') or {})
    for k, v in (isaac.get('cfg') or {}).items():
        if not hasattr(cfg, k):
            raise KeyError(f'설정 env.isaac.cfg.{k}: Config 에 없는 필드')
        cur = getattr(cfg, k)
        setattr(cfg, k, tuple(v) if isinstance(cur, tuple) and isinstance(v, list) else v)

    resolved = _deep_merge(d, {'run': run, 'log': log, 'env': {'kind': kind, 'isaac': isaac}})
    if float(reward.shape_tilt) > 0:
        resolved = _deep_merge(resolved, {'reward': {'shape_gamma': float(reward.shape_gamma)}})   # 재현용: 채운 γ 기록
    if capture.enabled and kind != 'isaac':
        raise ValueError('capture.enabled 는 env.kind=isaac 에서만 (surrogate 캡처는 의미 없음)')
    return SimpleNamespace(cfg=cfg, obs=obs, reward=reward, scenario=scenario, surrogate=surrogate, capture=capture,
                           run=run, log=log, isaac=isaac, env_kind=kind, agent_type=atype, resolved=resolved)


def add_cli(ap):
    """argparse 에 --config/--set 을 붙인다."""
    ap.add_argument('--config', action='append', required=True, help='YAML (여러 개면 뒤가 덮어씀)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VAL', help='설정 덮어쓰기 (예: agent.swirl.R=2)')
    return ap


def save_resolved(exp, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(exp.resolved, f, allow_unicode=True, sort_keys=False)
