"""env/knobs.py — Isaac 노드·UKF 의 세부 노브 저장소 (구 env 변수 대체).

YAML `env.isaac.knobs: {HOVER_ATT: 1, UKF_Q_GYRO: 2e-3, ...}` 가 여기에 들어가고, 코드는 knob('이름', 기본) 으로 읽는다.
os.environ.get 과 같은 문자열 의미를 유지해 기존 파싱(float(...), == '1', not in ('', '0'))이 그대로 동작한다.
프로세스 env 는 읽지 않는다 — 설정 파일에 없는 노브는 기본값이다.
"""
_K: dict = {}


def _s(v) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, (list, tuple)):
        return ','.join(str(x) for x in v)
    return str(v)


def set_knobs(d: dict):
    _K.clear()
    _K.update({str(k): _s(v) for k, v in (d or {}).items()})


def knob(name: str, default=None):
    return _K.get(name, default)


def knobs() -> dict:
    return dict(_K)
