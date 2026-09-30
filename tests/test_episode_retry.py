"""★09-30 Isaac 에피소드 누락 버그: 번호는 _end_episode(학습) 뒤에만 소비, 그 전 리셋이면 같은 번호 재시도(_advance_episode_number)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

def _node():
    try:
        import online_rl_main as M
    except Exception as e:                     # ROS 없는 환경
        pytest.skip(f'online_rl_main import 불가: {e}')
    logs = []
    lg = types.SimpleNamespace(warn=lambda m: logs.append(('W', m)), error=lambda m: logs.append(('E', m)), info=lambda m: None)
    def die(msg):
        n._fatal = msg; raise SystemExit(msg)
    n = types.SimpleNamespace(episode=0, get_logger=lambda: lg, _fatal=None, _sc_die=die)
    adv = lambda **kw: M.OnlineRLController._advance_episode_number(n, **kw) if hasattr(M, 'OnlineRLController') else _find(M)(n, **kw)
    return n, adv, logs

def _find(M):
    for v in vars(M).values():
        if isinstance(v, type) and hasattr(v, '_advance_episode_number'):
            return v._advance_episode_number
    raise AssertionError('_advance_episode_number 없음')

def test_normal_flow_increments_once_per_completed_episode():
    n, adv, logs = _node()
    for k in range(1, 6):
        adv(); assert n.episode == k
        n._ep_consumed = True                  # _end_episode(학습) 완료
    assert not logs

def test_takeoff_failures_retry_same_number():
    n, adv, logs = _node()
    adv(); n._ep_consumed = True               # ep1 완료
    adv(); assert n.episode == 2               # ep2 TAKEOFF
    adv(); adv(); assert n.episode == 2        # 이륙 실패 → WARM → 같은 번호 두 번 재시도
    assert sum(1 for t, _ in logs if t == 'W') == 2
    n._ep_consumed = True; adv(); assert n.episode == 3

def test_retry_cap_fails_loudly_instead_of_skipping():
    n, adv, logs = _node()
    adv()                                      # ep1 첫 시도
    adv(max_retry=3); adv(max_retry=3)         # 2·3번째 시도
    with pytest.raises(SystemExit):
        adv(max_retry=3)                       # 4번째 → 치명 종료(건너뛰지 않음)
    assert n.episode == 1 and n._fatal and 'EP-RETRY' in n._fatal
