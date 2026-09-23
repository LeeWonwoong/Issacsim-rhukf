"""LEARNER_PROC 학습기 프로세스 테스트 (CPU·가짜 전이, ROS·Isaac 없음):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m pytest -q tests/test_learner_proc.py

확인하는 것:
  · 동기 경로(train.py 식 push → learn → act, 같은 프로세스)와 같은 전이 열에서 **θ·행동 열이 수치로 동일**
    (SWIRL ui4·Adam ui1). 학습기는 spawn 프로세스, 행동은 메인의 CPU 사본, θ_k 수신 후에만 행동.
  · ui>1: 갱신 없는 스텝은 즉시 행동하고, 갱신 예측이 회신과 한 번도 어긋나지 않는다.
  · 체크포인트 save(FIFO)·load, 학습 예외 → 오류 회신 후 계속, 프로세스 사망 감지, 종료 정리(좀비 없음).
"""
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from cfgload import load_experiment          # noqa: E402
import train                                  # noqa: E402
from rl.learner_proc import LearnerProxy, LearnerDied   # noqa: E402

_TMP = tempfile.mkdtemp(prefix='learner_proc_test_')
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_QUIET = (lambda m, level='info': None)


def _exp(kind, *extra):
    f = 'configs/isaac_v5_swirl.yaml' if kind == 'swirl' else 'configs/isaac_v5_adam.yaml'
    return load_experiment([f], ['run.device=cpu', 'agent.batch=16', f'run.outdir={_TMP}/{kind}', *extra])


def _data(T, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.random((T, 12)).astype(np.float32), rng.normal(size=T).astype(np.float64), rng.random(T) < 0.03)


def _run_sync(exp, T, ep_len=70):
    ag = train.make_agent(exp)
    S, R, D = _data(T)
    np.random.seed(1234)
    acts = []; a_prev = None
    for t in range(T):
        if a_prev is not None:
            ag.push(S[t - 1], a_prev, float(R[t]), S[t], bool(D[t]))
            ag.learn()
        a = ag.act(S[t], ag.get_epsilon()); acts.append(int(a)); a_prev = a
        if t % ep_len == ep_len - 1:
            ag.end_episode(0.0, ep_len); a_prev = None
    return ag, acts


def _run_proc(exp, T, ep_len=70):
    px = LearnerProxy(exp.cfg, exp.cfg.agent_type, seed=exp.cfg.seed, log=_QUIET)
    S, R, D = _data(T)
    np.random.seed(1234)
    acts = []; a_prev = None; immediate = 0; waited = 0
    for t in range(T):
        if a_prev is not None:
            px.push(S[t - 1], a_prev, float(R[t]), S[t], bool(D[t]))
            _, upd = px.learn_async()
            if not upd:
                immediate += 1
            while not px.ready_to_act():                   # 노드는 여기서 막지 않고 드레인·틱을 돌린다(테스트는 폴링 대기)
                px.poll(); time.sleep(0.0005); waited += 1
        a = px.act(S[t], px.get_epsilon()); acts.append(int(a)); a_prev = a
        if t % ep_len == ep_len - 1:
            px.end_episode(0.0, ep_len); a_prev = None
    return px, acts, immediate


def _theta_equal(ag, kind, d):
    if kind == 'adam':
        strip = lambda sd: {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}
        ref = strip(ag.net.state_dict())
        return all(np.array_equal(ref[k].numpy(), d['theta']['net'][k]) for k in ref)
    return (np.array_equal(ag.theta.numpy(), d['theta']['theta'])
            and np.array_equal(ag.theta_target.numpy(), d['theta']['theta_target']))


def _det_check(kind, T):
    exp = _exp(kind)
    ag, acts_ref = _run_sync(exp, T)
    px, acts, immediate = _run_proc(_exp(kind), T)
    try:
        d = px.get_theta()
        assert acts == acts_ref, 'action sequence differs'
        assert _theta_equal(ag, kind, d), 'theta differs'
        assert px.steps_done == ag.steps_done               # 행동 수(ε 스케줄)는 메인 소유. 학습기 값은 마지막 learn 요청 때 값
        assert px.buffer.current_size == ag.buffer.current_size == d['size']
        assert px.n_mismatch == 0 and px.n_errors == 0
        return px, ag, immediate
    finally:
        px.close()


def test_swirl_ui4_matches_sync_path():
    px, ag, immediate = _det_check('swirl', 160)
    assert exp_ui(px) == 4 and immediate > 100          # ui4: 갱신 없는 스텝은 즉시 행동
    assert not px._proc.is_alive() and px._proc.exitcode is not None


def exp_ui(px):
    return int(px.cfg.update_interval)


def test_adam_ui1_matches_sync_path():
    px, ag, immediate = _det_check('adam', 200)
    assert exp_ui(px) == 1
    assert immediate == 15 + 2                           # 버퍼<batch(16) 구간 + n-step 캐시 채움 동안만 즉시


def test_checkpoint_save_load_and_close():
    exp = _exp('adam')
    px, _, _ = _run_proc(exp, 60)
    path = os.path.join(_TMP, 'model_ep1.pt')
    d = px.get_theta()
    px.save(path)                                        # FIFO: 앞선 learn 뒤 θ 로 저장
    sd = px.steps_done
    px.close()
    assert os.path.exists(path) and not px._proc.is_alive()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['steps_done'] == sd
    for k, v in ck['net'].items():
        assert np.array_equal(v.numpy(), d['theta']['net'][k.replace('_orig_mod.', '')])
    px2 = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET)
    try:
        px2.load(path)
        assert px2.steps_done == sd
        s = np.random.default_rng(1).random(12).astype(np.float32)
        q_ref = px2._actor.net(torch.as_tensor(s).unsqueeze(0)).detach().numpy()
        ag = train.make_agent(_exp('adam')); ag.load(path)
        assert np.allclose(q_ref, ag.net(torch.as_tensor(s).unsqueeze(0)).detach().numpy())
    finally:
        px2.close()


def test_learn_error_reported_and_continues():
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET)
    try:
        bad = np.zeros(5, np.float32)                    # 잘못된 차원 → n-step 캐시(3)가 차는 순간 버퍼 기록에서 학습기 예외
        for _ in range(int(px.cfg.n_step_size) if px.cfg.use_n_step else 1):
            px._send(('push', bad, 0, 0.0, bad, False))
        t_end = time.time() + 30
        while px.n_errors == 0 and time.time() < t_end:
            px.poll(); time.sleep(0.01)
        assert px.n_errors == 1 and px._proc.is_alive()
        px.buffer.reset_n_step_cache()                   # 오염된 캐시 비우고
        s = np.zeros(12, np.float32)
        px.push(s, 0, 0.0, s, True); out = px.learn()    # 이후 정상 동작(버퍼<batch 라 갱신 없음 회신)
        assert out == (0.0, 0.0, 0.0) and px.n_errors == 1
    finally:
        px.close()


def test_death_detected():
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET)
    try:
        px._proc.kill(); px._proc.join(5)
        try:
            px.poll()
        except LearnerDied:
            pass
        else:
            raise AssertionError('LearnerDied not raised')
    finally:
        px.close()
    assert not px._proc.is_alive()


def test_hang_detected():
    import signal
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET, hang_timeout=0.5)
    try:
        os.kill(px._proc.pid, signal.SIGSTOP)            # 살아 있지만 응답 없는 학습기
        px._predict_update = lambda: True                 # 이 요청은 갱신 예측 → θ_k 대기
        px.push(np.zeros(12, np.float32), 0, 0.0, np.zeros(12, np.float32), False)
        px.learn_async()
        assert not px.ready_to_act()
        time.sleep(0.7)
        try:
            px.poll()
        except LearnerDied:
            pass
        else:
            raise AssertionError('hang not detected')
    finally:
        try: os.kill(px._proc.pid, signal.SIGCONT)
        except Exception: pass
        px.close(timeout=5)
    assert not px._proc.is_alive()


def test_learn_branch_error_always_replies_with_k():
    """rev2: learn 분기 어디서 예외가 나도(여기선 steps_done 변환 실패) k 를 단 회신 → 갱신 대기가 풀린다. 연속 오류 max_err 회면 LearnerDied."""
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET, max_err=3)
    try:
        _send = px._send
        px._send = lambda m: _send(('learn', m[1], 'not-an-int') if m[0] == 'learn' else m)
        px._predict_update = lambda: True
        k, upd = px.learn_async()
        assert upd and not px.ready_to_act()
        t_end = time.monotonic() + 30
        while not px.ready_to_act() and time.monotonic() < t_end:
            px.poll(); time.sleep(0.005)
        assert px.ready_to_act() and px.n_errors == 1 and px.last_reply['k'] == k and px.last_reply['err']
        px.learn_async(); px.learn_async()
        died = False
        t_end = time.monotonic() + 30
        while time.monotonic() < t_end:
            try:
                px.poll()
            except LearnerDied as e:
                died = '연속 학습 오류' in str(e); break
            time.sleep(0.005)
        assert died
    finally:
        px.close(timeout=10)
    assert not px._proc.is_alive()


def test_wait_for_error_is_immediate():
    """load 실패(경로 오타)는 600 s 시간 초과가 아니라 즉시 LearnerDied(원인 포함)."""
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=_QUIET)
    try:
        t0 = time.monotonic()
        try:
            px.load(os.path.join(_TMP, 'no_such_model.pt'))
        except LearnerDied as e:
            assert 'load' in str(e)
        else:
            raise AssertionError('load error not raised')
        assert time.monotonic() - t0 < 10 and px._proc.is_alive()
    finally:
        px.close()


def test_sigterm_ignored_save_atomic_and_logged():
    """자식은 SIGINT/SIGTERM 을 무시(부모가 stop 으로 순서 주도) · save 는 원자적(tmp → replace) · 저장 완료를 부모 로그에 남긴다."""
    import signal
    logs = []
    px = LearnerProxy(_exp('adam').cfg, 'adam', log=lambda m, level='info': logs.append(m))
    try:
        os.kill(px._proc.pid, signal.SIGTERM); os.kill(px._proc.pid, signal.SIGINT)
        time.sleep(0.5)
        assert px._proc.is_alive()
        path = os.path.join(_TMP, 'atomic.pt')
        px.save(path)
        px.get_theta()                                   # FIFO: save 처리 뒤
        assert os.path.exists(path) and not os.path.exists(path + '.tmp')
        assert any('[LEARNER] saved' in m and path in m for m in logs), logs
        torch.load(path, map_location='cpu', weights_only=False)
    finally:
        px.close()
    assert not px._proc.is_alive()


_ORPHAN = r"""
import os, sys, time
sys.path.insert(0, {root!r}); os.chdir({root!r})
if __name__ == '__main__':
    from cfgload import load_experiment
    from rl.learner_proc import LearnerProxy
    exp = load_experiment(['configs/isaac_v5_adam.yaml'], ['run.device=cpu', 'agent.batch=16', 'run.outdir={out}'])
    px = LearnerProxy(exp.cfg, 'adam', log=lambda m, level='info': None)
    print('LEARNER_PID', px._proc.pid, flush=True)
    time.sleep(60)
"""


def test_parent_sigkill_kills_learner():
    """부모가 SIGKILL 로 죽어도 학습기 자식이 고아로 남지 않는다(PR_SET_PDEATHSIG)."""
    import signal
    import subprocess
    src = os.path.join(_TMP, 'orphan_parent.py')
    with open(src, 'w') as f:
        f.write(_ORPHAN.format(root=ROOT, out=os.path.join(_TMP, 'orphan')))
    p = subprocess.Popen([sys.executable, src], stdout=subprocess.PIPE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    try:
        child = None
        for _ in range(50):
            ln = p.stdout.readline().decode()
            if ln.startswith('LEARNER_PID'):
                child = int(ln.split()[1]); break
        assert child is not None
        os.kill(child, 0)                                # 살아 있음
        p.send_signal(signal.SIGKILL); p.wait(10)
        t_end = time.monotonic() + 10
        gone = False
        while time.monotonic() < t_end:
            try:
                os.kill(child, 0)
                with open(f'/proc/{child}/stat') as f:  # 좀비(부모 없는 init 입양 후 곧 수거)도 사라진 것으로 본다
                    if f.read().split()[2] == 'Z':
                        gone = True; break
            except (ProcessLookupError, FileNotFoundError):
                gone = True; break
            time.sleep(0.1)
        assert gone, f'learner child {child} survived parent SIGKILL'
    finally:
        if p.poll() is None:
            p.kill()


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            t0 = time.time(); f(); print('ok', k, f'{time.time() - t0:.1f}s', flush=True)
