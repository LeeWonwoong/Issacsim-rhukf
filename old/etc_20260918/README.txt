2026-09-18 재구성(refactor/env-split-0918)으로 옮긴 구 실험 코드. 새 코드에서 참조하지 않는다.

scripts/   구 surrogate(env5·7·8·surrogate_run), env_scan_v*, chain_hz, cp_regime_scan, 풀 빌더(build_pool_v*),
           각종 launch_*.sh·agg_*.py·alias_*.py 등. 모두 env 변수로 설정을 넘기던 방식.
           ↳ 대체: train.py + configs/*.yaml (+ scripts/grid.py), 관측·보상·공격 = env/*.py, surrogate 환경 = sim/surrogate.py
           ↳ 회귀 검증(09-18): 구 surrogate_run 과 새 train.py 가 Adam 200ep·SWIRL 3ep 에서 비트 단위 동일.
frozen_v3.env  구 FROZEN-v3 env 변수 묶음 (→ YAML 로 대체)
logs/          구 실행 로그
etc/analysis·etc/docs 는 분석 노트라 그대로 etc/ 에 남겼다.
