# UAV_AttackDetection_SWRL_OnlineRL — RHUKF-FV 통합본

기존 SWRL/SRRHUIF 파이프라인을 **RHUKF-FV(Receding-Horizon UKF, Full-Vector, Covariance form)** 로
교체한 전체 프로젝트입니다. 온라인(`online_rl_main.py`)·오프라인 검증(`custom_env.py`) 모두 통합·검증 완료.

## 무엇이 바뀌었나
- `rl/srrhuif_core.py` **삭제** → `rl/rhukf_core.py` **신규** (covariance form, error/absolute 스위치)
- `rl/agent.py` : `OnlineRHUKFAgent` (learn() → 3-튜플, soft target, PER β-annealing, n-step)
- `rl/network.py`, `rl/memory.py` : **FP32 + TF32**, PER(우선순위/IS-weight)·n-step 버퍼
- `env/reward.py` : `RewardConfig` 모듈화 (+ 기존 `calculate_reward` 하위호환)
- `swrl_config.py` : RHUKF 필드(state_form/measurement_mode/anchor/N_horizon/q·r·p_init 등),
  `reward: RewardConfig`, `done_steps`(기존 `done_step` 오타 수정), `obs_scale=[1.0]*12`
- `online_rl_main.py` : import/agent 3곳 RHUKF로 교체, 보상 호출에 `rc=cfg.reward`
- `custom_env.py` : RHUKF-FV vs Adam D3QN 오프라인 비교(FP32) + 4-Context Q-landscape

> 참고: `calibration/calibrate_online_today.py`의 저장부에 있던 `json.setdefault(...)` 크래시 라인을
> 제거했습니다(정상 `json.dump(..., indent=4)`만 유지). 그 외 calibration 로직은 원본 그대로입니다.

## 핵심 설정 (swrl_config.py)
- 필터: RHUKF-FV / covariance / `state_form='error'`(기본, `'absolute'` 전환 가능)
- `measurement_mode='q_target'`, activation `silu`, soft update `tau=0.005`
- `use_n_step=True`(n=3), `use_per=True`(IS-weight를 R 대각에 `/w` 적용), twin off
- UT: `alpha=0.9, beta=2.0, kappa=0.0` / `q_init=0.01, r_init=1.5, p_init=0.03, p_delta_init=0.05`
- `N_horizon=5`, `huber_c=1000`, `tikhonov_lambda=1e-8`
- `update_interval=1` : learn 빈도 게이트 (N 스텝마다 1번만 업데이트, 1=매 스텝). 온라인·오프라인·Adam 모두 동일 적용

## 오프라인 검증
```bash
python custom_env.py --episodes 200 --state_form error     # RHUKF-FV(error) vs Adam
python custom_env.py --episodes 200 --state_form absolute   # absolute 모드
python custom_env.py --episodes 200 --skip_adam             # RHUKF 단독
```
→ `./results_drone/<param_str>/` 에 reward/loss 곡선 + 4-Context Q-landscape PNG.

## 온라인 (실시간, 서버)
```bash
# 터미널 1: Isaac Sim + PX4 SITL
isim run_sim.py --no-headless
# 터미널 2: ROS2 RL 노드 (calibration.json이 실행 디렉터리에 있어야 함)
isim online_rl_main.py
```
PX4 최초 1회 (pxh):
```
param set COM_LOW_BAT_ACT 0
param set COM_DISARM_LAND -1
param save
```

## 구조
```
online_rl_main.py        # ROS2 RL 제어/평가 노드 (50Hz tick, 비동기 learn)
run_sim.py               # Isaac Sim + Pegasus + PX4 SITL
swrl_config.py           # 통합 설정 + 시나리오 샘플러 + 공격 유틸
custom_env.py            # 오프라인 RHUKF-FV vs Adam 벤치
env/
  ukf_filter.py          # 12-state Dynamics UKF + NIS
  reward.py              # RewardConfig + calculate_reward
rl/
  agent.py               # OnlineRHUKFAgent
  rhukf_core.py          # RHUKF-FV step (absolute/error) + PER priorities
  network.py             # flat-theta forward (FP32)
  memory.py              # PER + n-step 버퍼
calibration/             # SysID(OLS) + 당일 RLS 보정 + calibration.json
utils/plotting.py
```
