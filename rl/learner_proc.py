"""rl/learner_proc.py — 학습기를 별도 프로세스로 (LEARNER_PROC=1, Isaac 노드 전용).

왜: 같은 프로세스 스레드 학습은 act 가 락을 기다리는 동안 rclpy 실행기 전체가 멈추고(틱 소실),
락 밖에서도 GIL 을 다툰다. 학습기를 spawn 프로세스로 떼면 노드의 필터 드레인·틱은 학습과 무관하게 돈다.

의미(정석 DQN = surrogate train.py 의 push → learn → act):
  · 메인이 RL 스텝 k 에서 push(s_{k−1}, a_{k−1}, r, s_k) 와 learn 요청 k 를 보낸다(steps_done 동봉 → r-std 스케줄 정합).
  · 학습기는 FIFO 로 push → learn 하고 갱신된 행동용 파라미터 θ_k 와 학습 통계를 돌려준다.
  · 메인은 θ_k 를 받은 뒤에만 a_k 를 고른다(ready_to_act). 갱신이 없는 요청(ui 게이트·버퍼<batch·지평<N)은
    θ_k = θ_{k−1} 이므로 즉시 행동한다 — 갱신 여부는 버퍼 크기·호출 수·지평 길이 거울로 예측하고 회신의 updated 로 검증한다.
  · 행동은 메인의 CPU 사본이 실제 에이전트 클래스의 act 메서드로 계산한다(ε·εz 지속·act_net/eval_net 의미 동일).
    steps_done·ε·εz 상태는 메인이 소유한다.
  · 체크포인트(model_ep*.pt·final_model.pt)는 학습기 프로세스가 같은 경로에 같은 형식으로 저장한다.

통신: multiprocessing Pipe(spawn 문맥 — CUDA 안전). 메인은 poll() 로만 읽는다(블로킹 없음).
초기화·load·compile·close 만 블로킹(스핀 전·종료 때).
"""
from __future__ import annotations

import atexit
import collections
import copy
import os
import time
import traceback

import multiprocessing as mp
import numpy as np

_CTX = mp.get_context('spawn')


class LearnerDied(RuntimeError):
    pass


# ══════════════════════════════════════════════════════════════════════
#  공용: 행동용 파라미터 직렬화
# ══════════════════════════════════════════════════════════════════════
def _strip(sd):
    return {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v.detach().cpu().numpy().copy() for k, v in sd.items()}


def theta_payload(agent, agent_type):
    if agent_type == 'adam':
        return {'net': _strip(agent.net.state_dict()), 'target_net': _strip(agent.target_net.state_dict())}
    return {'theta': agent.theta.detach().cpu().numpy().copy(), 'theta_target': agent.theta_target.detach().cpu().numpy().copy()}


def make_actor(cfg, agent_type, device='cpu'):
    """행동 전용 경량 인스턴스: 버퍼·옵티마이저·필터 캐시 없이 act/get_epsilon/get_q_values 에 필요한 속성만.
    실제 클래스의 메서드를 그대로 쓰므로 ε-greedy·εz 지속·act_net 의미가 학습기 구현과 한 줄도 다르지 않다."""
    import torch
    acfg = copy.copy(cfg); acfg.device = device
    if agent_type == 'adam':
        from rl.agent_adam import OnlineAdamAgent, _DDQNNet
        a = object.__new__(OnlineAdamAgent)
        a.cfg = acfg; a.device = device
        a.net = _DDQNNet(cfg.dimS, cfg.num_actions, cfg.shared_layers, cfg.q_layers, cfg.activation_fn).float().to(device)
        a.target_net = copy.deepcopy(a.net)
    else:
        from rl.agent import OnlineRHUKFAgent
        from rl.network import create_network_info, InputNormalizer
        a = object.__new__(OnlineRHUKFAgent)
        a.cfg = acfg; a.device = device
        a.info = create_network_info(cfg.dimS, cfg.num_actions, acfg)
        a.normalizer = InputNormalizer(device, scale=cfg.obs_scale) if cfg.use_input_norm else None
        n = a.info['total_params']
        a.theta = torch.zeros(n, 1, dtype=torch.float32, device=device)
        a.theta_target = torch.zeros(n, 1, dtype=torch.float32, device=device)
    a.steps_done = 0; a._z_left = 0; a._z_a = 0; a.episode_count = 0
    return a


def load_actor_theta(actor, agent_type, payload):
    import torch
    if agent_type == 'adam':
        actor.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in payload['net'].items()})
        actor.target_net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in payload['target_net'].items()})
    else:
        n = actor.info['total_params']
        actor.theta = torch.from_numpy(np.asarray(payload['theta'], dtype=np.float32).reshape(n, 1).copy())
        actor.theta_target = torch.from_numpy(np.asarray(payload['theta_target'], dtype=np.float32).reshape(n, 1).copy())


def make_agent_like_isaac(cfg, agent_type):
    """online_rl_main 구 경로와 같은 생성(시드는 호출자가 먼저 건다)."""
    if agent_type == 'adam':
        from rl.agent_adam import OnlineAdamAgent
        return OnlineAdamAgent(cfg)
    from rl.agent import OnlineRHUKFAgent
    return OnlineRHUKFAgent(cfg)


# ══════════════════════════════════════════════════════════════════════
#  학습기 프로세스 본체
# ══════════════════════════════════════════════════════════════════════
_STAT_KEYS = (('kgain', '_last_kgain'), ('pmax', '_last_pmax'), ('innov', '_last_innov'),
              ('flip', '_last_argmax_flip'), ('qmax', '_last_qmax'), ('nis', '_last_nis'))


def _child_guard(parent_pid):
    """학습기 자식 프로세스 방어: 종료는 부모의 'stop'(대기 중 save 처리 후 bye) 또는 파이프 EOF 로만.
    · SIGINT/SIGTERM 무시 — 같은 프로세스 그룹의 Ctrl-C·timeout 이 torch.save 도중 자식을 죽여 체크포인트를 깨지 않게
      (부모가 finally 에서 close() 로 순서를 주도한다. 끝내 안 되면 부모가 SIGKILL).
    · PR_SET_PDEATHSIG=SIGKILL — 부모가 SIGKILL 로 죽으면 즉시 따라 죽는다(GPU 를 쥔 고아 방지).
    · stdout/stderr 줄 버퍼링 — spawn 자식은 python -u 를 물려받지 않아 '[Save]' 줄이 블록 버퍼에 갇힌다."""
    import signal
    import sys
    for _st in (sys.stdout, sys.stderr):
        try:
            _st.reconfigure(line_buffering=True)
        except Exception:
            pass
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, signal.SIG_IGN)
        except Exception:
            pass
    try:
        import ctypes
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, int(signal.SIGKILL))   # PR_SET_PDEATHSIG
    except Exception:
        pass
    if parent_pid is not None and os.getppid() != int(parent_pid):             # 설정 전에 부모가 이미 죽었으면
        os._exit(0)


def _atomic_save(agent, path):
    """tmp 에 저장 후 os.replace — 저장 도중 끊겨도 기존 파일이 잘리지 않는다. '[Save] 경로' 줄은 최종 경로로 그대로."""
    import contextlib
    import io
    tmp = f'{path}.tmp'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.save(tmp)
    os.replace(tmp, path)
    out = buf.getvalue().replace(tmp, str(path))
    if out:
        print(out, end='', flush=True)


def _child_main(conn, cfg, agent_type, knobs_d, seed, use_dynamo_quiet, parent_pid=None):
    _child_guard(parent_pid)
    try:
        import random
        import torch
        if knobs_d is not None:
            from env.knobs import set_knobs
            set_knobs(knobs_d)
        if use_dynamo_quiet and hasattr(torch, '_dynamo'):          # run_isaac 와 같은 컴파일 설정
            torch._dynamo.config.suppress_errors = True
            import logging as _lg
            for _n in ("torch._dynamo", "torch._inductor", "torch._functorch",
                       "torch._dynamo.convert_frame", "torch._inductor.compile_fx"):
                _lg.getLogger(_n).setLevel(_lg.ERROR)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from rl.network import apply_tf32_config
        apply_tf32_config(cfg)
        agent = make_agent_like_isaac(cfg, agent_type)
        conn.send(('ready', dict(theta=theta_payload(agent, agent_type), pid=os.getpid(),
                                 steps_done=int(agent.steps_done), episode_count=int(agent.episode_count))))
    except BaseException:                                            # noqa: BLE001
        try: conn.send(('fatal', traceback.format_exc()))
        except Exception: pass
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):
            break
        op = msg[0]
        try:
            if op == 'push':
                agent.push(msg[1], int(msg[2]), float(msg[3]), msg[4], bool(msg[5]))
            elif op == 'learn':
                # ★rev2: 이 분기 어디서 예외가 나도(learn 뒤 통계·θ 직렬화의 CUDA 동기화 오류 포함) 반드시 k 를 단 회신을 보낸다
                #   → 부모의 '갱신 대기'가 풀리고(무한 대기 없음) 연속 오류는 부모가 세어 치명 처리한다.
                k = int(msg[1]); t0 = time.perf_counter()
                try:
                    agent.steps_done = int(msg[2])
                    err = None
                    try:
                        loss, dt_ms, z_var = agent.learn()
                        updated = float(dt_ms) > 0.0             # 갱신 없이 돌아오는 경로는 (0,0,0) 을 준다
                    except Exception:                            # 구 스레드 경로처럼 로그만 남기고 계속
                        err = traceback.format_exc(); loss = dt_ms = z_var = 0.0; updated = True
                    rep = dict(k=k, updated=bool(updated), loss=float(loss), dt_ms=float(dt_ms), z_var=float(z_var),
                               wall_ms=(time.perf_counter() - t0) * 1000.0, size=int(agent.buffer.current_size),
                               calls=int(getattr(agent, '_learn_call_count', 0)), err=err)
                    if updated:
                        for key, attr in _STAT_KEYS:
                            rep[key] = float(getattr(agent, attr, 0.0) or 0.0)
                        try: rep['td_kurt'] = tuple(agent.td_kurtosis())
                        except Exception: rep['td_kurt'] = (0, 0.0, 0.0)
                        rep['theta'] = theta_payload(agent, agent_type)
                except Exception:                                # noqa: BLE001
                    rep = dict(k=k, updated=True, loss=0.0, dt_ms=0.0, z_var=0.0, wall_ms=(time.perf_counter() - t0) * 1000.0,
                               size=-1, calls=-1, err=traceback.format_exc())
                conn.send(('learn', rep))
            elif op == 'end_ep':
                agent.end_episode(float(msg[1]), int(msg[2]))
            elif op == 'reset_n':
                agent.buffer.reset_n_step_cache()
            elif op == 'save':
                if len(msg) > 2 and msg[2] is not None:
                    agent.steps_done = int(msg[2])             # 행동 수(ε 스케줄)는 메인이 소유 — 저장 시점 값으로
                _atomic_save(agent, msg[1]); conn.send(('saved', msg[1]))
            elif op == 'load':
                agent.load(msg[1])
                conn.send(('loaded', dict(theta=theta_payload(agent, agent_type), steps_done=int(agent.steps_done),
                                          episode_count=int(agent.episode_count))))
            elif op == 'compile':
                agent.warmup_compile(); conn.send(('compiled', dict(theta=theta_payload(agent, agent_type))))
            elif op == 'get_theta':
                conn.send(('theta', dict(theta=theta_payload(agent, agent_type), steps_done=int(agent.steps_done),
                                         size=int(agent.buffer.current_size))))
            elif op == 'stop':
                try:                                         # inductor 컴파일 워커 정리(안 하면 atexit 가 늦어 join 이 terminate 로 간다)
                    from torch._inductor.async_compile import shutdown_compile_workers
                    shutdown_compile_workers()
                except Exception:
                    pass
                conn.send(('bye', None)); break
            else:
                conn.send(('error', op, f'unknown op {op!r}'))
        except Exception:                                            # noqa: BLE001
            # ★rev2: 구 코드는 OSError 를 '파이프 끊김'으로 보고 루프를 빠져나갔다 → load 경로 오타(FileNotFoundError ⊂ OSError)
            #   하나로 학습기가 조용히 끝났다. 이제 op 오류는 회신하고, 회신 자체가 실패할 때만(진짜 파이프 끊김) 끝낸다.
            try: conn.send(('error', op, traceback.format_exc()))
            except Exception: break
    try: conn.close()
    except Exception: pass


# ══════════════════════════════════════════════════════════════════════
#  메인 쪽 대리자
# ══════════════════════════════════════════════════════════════════════
class _BufMirror:
    """TensorReplayBuffer 의 크기만 거울로 셈(n-step 캐시 규칙 포함). reset_n_step_cache 는 학습기에도 전달."""
    def __init__(self, cfg, send):
        self.capacity = int(cfg.buffer_size)
        self.use_n = bool(cfg.use_n_step)
        self.n = int(cfg.n_step_size) if self.use_n else 1
        self.count = 0; self.cache = 0; self._send = send

    def push(self, done):
        if not self.use_n:
            self.count += 1; return
        self.cache += 1
        if self.cache == self.n:
            self.count += 1; self.cache -= 1
        if done:
            self.count += self.cache; self.cache = 0

    def reset_n_step_cache(self):
        self.cache = 0
        self._send(('reset_n',))

    @property
    def current_size(self):
        return min(self.count, self.capacity)


class LearnerProxy:
    """노드가 쓰는 에이전트 인터페이스(act/push/end_episode/save/load/get_epsilon/td_kurtosis/buffer…) + 비동기 learn."""

    def __init__(self, cfg, agent_type='rhukf', knobs=None, seed=None, log=None, start_timeout=900.0,
                 actor_device='cpu', dynamo_quiet=True, hang_timeout=120.0, max_err=20):
        self.cfg = cfg
        self.agent_type = 'adam' if agent_type == 'adam' else 'rhukf'
        self._log = log or (lambda m, level='info': print(m, flush=True))
        self._actor = make_actor(cfg, self.agent_type, actor_device)
        parent, child = _CTX.Pipe(duplex=True)
        self._proc = _CTX.Process(target=_child_main, name='rl-learner', daemon=True,
                                  args=(child, cfg, self.agent_type, knobs, int(cfg.seed if seed is None else seed), bool(dynamo_quiet),
                                        os.getpid()))
        self._proc.start(); child.close()
        self._conn = parent
        self._closed = False
        self._hung = False
        self.max_err = int(max_err); self._err_run = 0
        self.buffer = _BufMirror(cfg, self._send)
        self._calls = 0; self._hist = 0
        self._req = 0
        self._outstanding = collections.deque()      # 갱신 예측 요청 중 미회신
        self._sent_at = {}                           # 요청 번호 → 전송 벽시계 (멈춘 학습기 감시)
        self.hang_timeout = float(hang_timeout)
        self._inflight = 0                           # 모든 learn 요청 중 미회신
        self._replies = []
        self.n_mismatch = 0; self.n_errors = 0
        self.last_td_kurt = (0, 0.0, 0.0)
        self.last_reply = None
        atexit.register(self.close)
        try:
            tag, data = self._wait_for(('ready',), start_timeout)
        except BaseException:
            self.close(timeout=5.0)
            raise
        load_actor_theta(self._actor, self.agent_type, data['theta'])
        self._actor.steps_done = int(data['steps_done']); self._actor.episode_count = int(data['episode_count'])
        self.pid = data['pid']
        self._log(f'  [LEARNER] 학습기 프로세스 기동 pid={self.pid} (spawn, agent={self.agent_type}, 행동망=메인 {actor_device})')

    # ── 전송·수신 ─────────────────────────────────────────────────────
    def _send(self, msg):
        if self._closed:
            raise LearnerDied('learner proxy closed')
        try:
            self._conn.send(msg)
        except (BrokenPipeError, EOFError, OSError) as e:
            raise LearnerDied(f'학습기 파이프 끊김: {e}') from e

    def _check_alive(self):
        if not self._proc.is_alive():
            raise LearnerDied(f'학습기 프로세스 종료(exitcode={self._proc.exitcode})')

    def _handle(self, msg):
        tag = msg[0]
        if tag == 'learn':
            rep = msg[1]; k = rep['k']
            self._inflight = max(0, self._inflight - 1)
            self._sent_at.pop(k, None)
            predicted = bool(self._outstanding and self._outstanding[0] == k)
            if predicted:
                self._outstanding.popleft()
            if rep['updated'] != predicted and rep.get('err') is None:
                self.n_mismatch += 1
                self._log(f'  [LEARNER] 갱신 예측 불일치 k={k} 예측={predicted} 실제={rep["updated"]} '
                          f'(size {self.buffer.current_size}/{rep["size"]}, calls {self._calls}/{rep["calls"]})', 'error')
            if rep.get('theta') is not None:
                load_actor_theta(self._actor, self.agent_type, rep['theta'])
            if rep['updated'] and rep.get('td_kurt') is not None:
                self.last_td_kurt = tuple(rep['td_kurt'])
            if rep.get('err'):
                self.n_errors += 1; self._err_run += 1
            else:
                self._err_run = 0
            self.last_reply = rep
            self._replies.append(rep)
            if self.max_err > 0 and self._err_run >= self.max_err:
                raise LearnerDied(f'연속 학습 오류 {self._err_run}회 (LEARNER_MAX_ERR={self.max_err}) — 마지막:\n'
                                  f'{str(rep.get("err")).strip().splitlines()[-1]}')
        elif tag == 'saved':
            self._log(f'  [LEARNER] saved {msg[1]}')     # 저장 완료 시점을 노드 로그 순서에 고정('[Save]' 줄은 학습기가 출력)
        elif tag == 'error':
            self.n_errors += 1
            self._log(f'  [LEARNER ERROR] op={msg[1]}: {str(msg[2]).strip().splitlines()[-1] if msg[2] else ""}', 'error')
            if msg[1] == 'learn' and self._outstanding:   # (방어) k 없는 learn 오류 — 가장 오래된 갱신 대기를 풀어 무한 대기 방지
                self._sent_at.pop(self._outstanding.popleft(), None)
                self._inflight = max(0, self._inflight - 1)
        elif tag == 'fatal':
            raise LearnerDied(f'학습기 초기화 실패:\n{msg[1]}')
        return tag

    _OP_OF = {'ready': None, 'loaded': 'load', 'compiled': 'compile', 'theta': 'get_theta', 'bye': 'stop'}

    def _wait_for(self, tags, timeout):
        """초기화·load·compile·close 전용 블로킹 대기. 사이에 온 learn 회신은 정상 처리.
        기다리는 op 의 오류 회신이 오면 시간 초과를 기다리지 않고 즉시 LearnerDied(원인 포함)."""
        ops = {self._OP_OF.get(t) for t in tags} - {None}
        t_end = time.monotonic() + float(timeout)
        while True:
            try:
                if self._conn.poll(0.05):
                    msg = self._conn.recv()
                    if msg[0] in tags:
                        return msg[0], msg[1]
                    if msg[0] == 'error' and msg[1] in ops:
                        self.n_errors += 1
                        raise LearnerDied(f'학습기 {msg[1]} 실패:\n{msg[2]}')
                    self._handle(msg)
                    continue
            except (EOFError, OSError) as e:
                raise LearnerDied(f'학습기 파이프 끊김: {e}') from e
            self._check_alive()
            if time.monotonic() > t_end:
                raise LearnerDied(f'학습기 응답 시간 초과({timeout}s, 대기 {tags})')

    def poll(self):
        """논블로킹. 도착한 learn 회신 목록을 돌려준다(θ 는 이미 행동망에 반영). 학습기가 죽었으면 LearnerDied."""
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, OSError) as e:
            raise LearnerDied(f'학습기 파이프 끊김: {e}') from e
        self._check_alive()
        if self._outstanding and self.hang_timeout > 0:
            t0 = self._sent_at.get(self._outstanding[0])
            if t0 is not None and time.monotonic() - t0 > self.hang_timeout:
                self._hung = True
                raise LearnerDied(f'학습기 무응답 {time.monotonic() - t0:.0f}s (요청 {self._outstanding[0]}, 한도 {self.hang_timeout:.0f}s)')
        out, self._replies = self._replies, []
        return out

    # ── 에이전트 인터페이스 ──────────────────────────────────────────────
    def act(self, state, eps, greedy=False):
        return self._actor.act(state, eps, greedy=greedy)

    def get_epsilon(self):
        return self._actor.get_epsilon()

    def get_q_values(self, state):
        return self._actor.get_q_values(state)

    @property
    def steps_done(self):
        return self._actor.steps_done

    @steps_done.setter
    def steps_done(self, v):
        self._actor.steps_done = int(v)

    @property
    def episode_count(self):
        return self._actor.episode_count

    def push(self, s, a, r, s_next, done):
        self._send(('push', np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s_next, dtype=np.float32), bool(done)))
        self.buffer.push(bool(done))

    def _predict_update(self):
        """agent.learn() 의 게이트를 그대로 따라 이번 요청이 θ 를 바꾸는지 예측."""
        c = self.cfg
        if self.buffer.current_size < c.batch_size:
            return False
        self._calls += 1
        if c.update_interval > 1 and (self._calls % c.update_interval) != 0:
            return False
        if self.agent_type != 'adam':
            self._hist = min(self._hist + 1, int(c.N_horizon))
            if self._hist < int(c.N_horizon):
                return False
        return True

    def learn_async(self):
        """learn 요청 전송. 반환 (요청 번호, 갱신 예측). 갱신 예측이면 그 회신 전까지 ready_to_act()=False."""
        upd = self._predict_update()
        self._req += 1
        self._send(('learn', self._req, int(self._actor.steps_done)))
        self._inflight += 1
        self._sent_at[self._req] = time.monotonic()
        if upd:
            self._outstanding.append(self._req)
        return self._req, upd

    def ready_to_act(self):
        return not self._outstanding

    def learn(self):
        """동기 호환(테스트·도구용): 요청하고 그 회신을 기다린다. 노드는 learn_async + poll 을 쓴다."""
        k, _ = self.learn_async()
        t_end = time.monotonic() + 600.0
        while True:
            for rep in self.poll():
                if rep['k'] == k:
                    return rep['loss'], rep['dt_ms'], rep['z_var']
            if time.monotonic() > t_end:
                raise LearnerDied('learn 회신 시간 초과')
            self._conn.poll(0.01)

    def end_episode(self, total_reward, episode_length):
        self._actor.episode_count += 1
        self._actor._z_left = 0                        # εz 지속은 에피소드를 넘기지 않는다(에이전트와 동일)
        self.buffer.cache = 0                          # 에이전트 end_episode 가 n-step 캐시를 비운다
        self._send(('end_ep', float(total_reward), int(episode_length)))

    def save(self, path):
        self._send(('save', str(path), int(self._actor.steps_done)))   # FIFO: 앞선 push/learn 이 반영된 θ 로 저장

    def load(self, path):
        self._send(('load', str(path)))
        _, d = self._wait_for(('loaded',), 600.0)
        load_actor_theta(self._actor, self.agent_type, d['theta'])
        self._actor.steps_done = int(d['steps_done']); self._actor.episode_count = int(d['episode_count'])

    def warmup_compile(self):
        if not getattr(self.cfg, 'use_compile', False):
            return
        self._send(('compile',))
        _, d = self._wait_for(('compiled',), 1800.0)
        load_actor_theta(self._actor, self.agent_type, d['theta'])

    def get_theta(self):
        self._send(('get_theta',))
        _, d = self._wait_for(('theta',), 600.0)
        return d

    def td_kurtosis(self):
        return self.last_td_kurt

    def _compute_adaptive_p(self, *args, **kwargs):
        return self.cfg.p_init if self.agent_type != 'adam' else 0.0

    # ── 종료 ────────────────────────────────────────────────────────────
    def close(self, timeout=120.0):
        """stop → 대기 중인 save 까지 처리한 뒤 bye → join. 안 되면 terminate → kill. 여러 번 불러도 안전."""
        if self._closed:
            return
        try:
            if self._proc.is_alive():
                self._conn.send(('stop',))
                try:                                       # 무응답으로 판정된 학습기는 bye 를 오래 기다리지 않는다
                    self._wait_for(('bye',), 5.0 if self._hung else timeout)
                except Exception:
                    pass
        except Exception:
            pass
        self._closed = True
        try:
            self._proc.join(10.0)
            if self._proc.is_alive():                      # 자식은 SIGTERM 을 무시하므로 terminate 는 짧게, 곧 kill
                self._proc.terminate(); self._proc.join(1.0)
            if self._proc.is_alive():
                self._proc.kill(); self._proc.join(5.0)
        except Exception:
            pass
        try: self._conn.close()
        except Exception: pass
        try: atexit.unregister(self.close)
        except Exception: pass
