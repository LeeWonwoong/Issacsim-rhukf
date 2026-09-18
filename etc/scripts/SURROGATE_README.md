# NIS Surrogate 환경 — 하이퍼파라미터 스캔용 (Isaac 없이)

실제 `rl/agent.py`(RHUKF-FV) + `env/reward.py`를 그대로 재사용하고 **환경(NIS 관측)만 대체**한다.
Isaac 대기(sim time)를 제거 → GPU에서 빠르게 하이퍼파라미터를 훑는다.
**⚠ 용도는 스캔 전용. 기여 검증·최종 config 선정은 반드시 Isaac.**

## 파일 (전부 `etc/scripts/`)
- `surrogate_env.py`  — NIS 생성기 (실측 분포 + 정책의존성 + burst 시간구조)
- `surrogate_run.py`  — 러너 + CONFIGS 딕셔너리 (여기서 하이퍼파라미터 수정)
- `nis_conds.json`    — 실측 NIS 분포(config E). Isaac 데이터에서 뽑은 것. 재생성은 아래.

## 실행 (GPU 기본)
```bash
cd <repo루트>              # rl/, env/, swrl_config.py 가 있는 곳
python3 etc/scripts/surrogate_run.py
# 출력: config별  F1 / reward / loss / 소요시간
```
- `cfg.device`가 자동 cuda (GPU 전용이면 수 초/ep). CPU 강제하려면 `CUDA_VISIBLE_DEVICES="" python3 ...`.
- 로컬 GPU 전용 권장 (Isaac과 공유 시 컨텍스트 경쟁 가능).

## config 수정 (surrogate_run.py 하단 CONFIGS)
```python
CONFIGS = {
  '이름': dict(NET_HIDDEN=16, RHUKF_N=5, RHUKF_Q='1e-3',
              RHUKF_TAU=0.02, RHUKF_UI=4, RHUKF_R=1.0, RHUKF_PD=0.01),
  # ui=update_interval, N=N_horizon(세트), tau=필터시상수, R=측정노이즈, PD=p_init_delta, Q=프로세스노이즈
  # ★관측제약: batch(128) × N ≥ params([16,16]=514) → N≥5
  # 아직 안 돌린 축 스캔: RHUKF_Q 1e-2/1e-4, RHUKF_ALPHA 0.1/0.5/0.9, RHUKF_R 2/2.5/3
}
```
`n_ep`(run_config 기본 60), `seed`도 인자로 조절.

## ★ 반드시 할 검증 (충실도 담보)
Isaac 격자(results_g1_*)와 **같은 config를 surrogate에서도 돌려 랭킹이 일치하는지** 확인.
- 일치(F1·reward 순서 유사) → surrogate 신뢰 → 새 config 스캔.
- 불일치 → 폐기. (지금 Isaac set-A는 F1~0.97·둔감. surrogate도 그래야 정상)

## 충실도 설계 (surrogate_env.py)
```
① NIS 조건별 분포  : nis_conds.json 실측 empirical 샘플 (공격2.1 / 기동스파이크 / 바람 / 평시)
② 정책의존성       : 호버→기동aliasing 제거 · 공격→행동무관 지속 · track+기동→스파이크
③ 시간구조         : burst ON~U(40,80)/OFF~U(10,30) · 온셋 상승 · (오프셋)
④ reward/TD노이즈  : calculate_reward 동일 + 같은 DDQN → 자동 재현
```
한계: ②의 counterfactual(안 해본 행동의 NIS)은 근사. 펄스 시그니처·오프셋 동역학은 단순화됨.
→ **검증된 이웃 안에서만** 스캔하고, 외삽 금지.

## nis_conds.json 재생성 (플랜트/config 바뀌면)
현재 파일은 results_final2/results_flip(config E)에서 6조건 NIS를 뽑은 것.
플랜트가 바뀌면 새 Isaac 캡처로 다시 뽑아야 한다(조건별 nis_g_scaled/nis_v_scaled 샘플 → json).
