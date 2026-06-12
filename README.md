# UAV_AttackDetection_SWRL_OnlineRL — RHUKF-FV 통합본

## 이번 개정 (online 실험 세팅: burst LoE / Adam baseline / PER off / done 정책 / 리셋 수정)
- **agent_type 선택**: `cfg.agent_type = 'rhukf'`(제안) | `'adam'`(Adam+Huber baseline).
  `rl/agent_adam.py`(신규) = `OnlineRHUKFAgent`와 동일 인터페이스·동일 네트워크 구조,
  손실만 Huber(smooth_l1) + Adam + soft target update. → FIR(receding-horizon) **구조 순기여 isolate**.
- **버스트 LoE 공격**: `attack_mode='burst'`. 에피소드 내 on-off-on 반복(`attack_burst_*_range`).
  `sample_episode_scenario`가 `scenario['attack_bursts']=[(s,e),...]` 생성, 컨트롤러가 경계에서 토글.
  LoE 자체(`compute_attack_forces`)는 on/off 그대로 — 비정상성을 시간 구조로만 강조(FIR이 이기는 regime).
- **PER off → Huber-R 단독**: `use_per=False`. RHUKF의 outlier 방어를 Huber-adaptive R 하나로 격리.
  (Adam baseline도 동일하게 uniform 샘플링 → 공정 비교.)
- **done 정책**: `use_logical_done=False`(기본) → **물리적 crash만 종료**. 논리적 종료(미탐/복귀/오탐)는 끔.
  push의 terminal 마스크는 물리 crash에만 True (timeout·논리종료는 truncation → 부트스트랩 유지).
- **리셋 수정**: circle 시작 갭 제거(중심 (-R,0)), `crash_drift` 유예(`drift_patience`),
  eval crash도 reason별 SOFT/WARM/HARD 라우팅, SOFT_RECOVERY `soft_recovery_timeout` 초과 시 WARM 에스컬레이션.

## 정밀도/구조 (rhukf.py 기준 정렬)
- 전역 FP32 고정(`allow_tf32=False`), NN forward만 `@tf32_forward`로 호출 동안 TF32 허용(`use_tf32_forward`).
- 순수 DDQN(dueling 제거): `shared_layers → q_layers → nA`.
- `load_calibration()` 자동 탐색: 준 경로 → `calibration/<파일>` → 레포 루트 → cwd → glob.

## 비교 실험 (에이전트만 바꿔 두 번)
```python
cfg.agent_type = 'rhukf'    # 제안
# cfg.agent_type = 'adam'   # baseline (Adam + Huber)
cfg.use_per = False         # 공통: Huber-R 단독
cfg.attack_mode = 'burst'   # 공통: 버스트 LoE
```

## 핵심 설정 (swrl_config.py)
- 필터: RHUKF-FV / covariance / `state_form='error'`(기본, `'absolute'` 전환 가능)
- `measurement_mode='q_target'`, activation `silu`, soft update `tau=0.005`
- `use_n_step=True`(n=3), `use_per=False`(이번 실험), twin off
- UT: `alpha=0.9, beta=2.0, kappa=0.0` / `q_init=0.01, r_init=1.5, p_init=0.03, p_delta_init=0.05`
- `N_horizon=5`, `huber_c=1000`, `tikhonov_lambda=1e-8`, `update_interval=1`
- done: `use_logical_done=False`, `drift_patience=5`, `soft_recovery_timeout=15.0`
- attack: `attack_mode='burst'`, `attack_burst_count_range=(2,4)`, `_on_range=(15,35)`, `_off_range=(20,50)`

## 오프라인 검증 (custom_env — 이번 온라인 실험엔 미사용)
```bash
python custom_env.py --episodes 200 --state_form error
python custom_env.py --episodes 200 --skip_adam
```

## 온라인 (실시간, 서버)
```bash
# 터미널 1: Isaac Sim + PX4 SITL
isim run_sim.py --no-headless
# 터미널 2: ROS2 RL 노드 (calibration.json 자동 탐색)
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
swrl_config.py           # 통합 설정 + 시나리오 샘플러(버스트) + 공격 유틸
custom_env.py            # 오프라인 RHUKF-FV vs Adam 벤치
env/
  ukf_filter.py          # 12-state Dynamics UKF + NIS
  reward.py              # RewardConfig + calculate_reward
rl/
  agent.py               # OnlineRHUKFAgent
  agent_adam.py          # OnlineAdamAgent (Adam + Huber baseline)
  rhukf_core.py          # RHUKF-FV step (absolute/error) + PER priorities
  network.py             # flat-theta forward (FP32)
  memory.py              # PER + n-step 버퍼
calibration/             # SysID(OLS) + 당일 RLS 보정 + calibration.json
utils/plotting.py
```
