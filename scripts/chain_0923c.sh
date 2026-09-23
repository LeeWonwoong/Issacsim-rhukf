#!/usr/bin/env bash
# 09-23 01:35 사용자 확정 파이프라인:
#   ① n55 보상×R (SWIRL R 은 보상·TD 에 맞춘 2점) ② n54 정합신뢰도 ③ P(=K gain) 스윕 ④ N 스윕 8/10/12 ⑤ Isaac 연결
#   선택 기준(사용자): 오탐↓·지연↓·온셋 즉시탐지·유지·종료 즉시복귀·약공격·**Adam 대비 격차 최대**.
#   판정: SWIRL 절대 품질 하한(F1 ≥ 0.60 ∧ 사건탐지 ≥ 0.90) 안에서 ΔF1(SWIRL−Adam) 최대, 동률이면 ΔFA·Δ지연.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; S=/tmp/claude-1001/-home-acsl-projects-Issacsim-rhukf/c2b43851-6cf0-4b54-a537-da44ef7e978d/scratchpad
until [ -f results/claudecodefortest/n55_rho_sweep/GRID_DONE ] && [ -f results/claudecodefortest/n54_fidelity/GRID_DONE ]; do sleep 180; done
echo "- $(date '+%H:%M') chain_0923c: n55·n54 완료 → 보상·R 확정" >> $L
PICK=$(python3 $S/pick_regime.py 2>>$L); echo "$PICK" >> $L
set -- $PICK; CFA=$1; CD=$2; BO=$3; RV=$4
[ -z "$RV" ] && { echo "- $(date '+%H:%M') chain_0923c: 판정 불가 → 중단" >> $L; exit 0; }
RW="--set reward.c_fa=$CFA --set reward.c_d=$CD --set reward.bonus=$BO"
BASE="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=128 --set agent.swirl.R=$RV --set agent.swirl.q=1.0e-3 --set agent.swirl.act=target --set agent.swirl.argmax=spas --set agent.swirl.huber_c=0 --set agent.hidden=[24,24] --set agent.buffer=50000 --set log.probe_every=20 --set log.probe_n=8 --set log.eval_n=100"
# ③ P(=K gain) 스윕: 사용자 지시로 **0.01 이상만** — 0.01(기준) / 0.05 / 0.1.
#    α 는 안정대역 규칙 α√pΔ = 0.012 로 같이 움직이고, R 은 같은 비(R/P_zz_σ)를 유지하도록 pΔ^0.53 배 한다(실측 지수).
#    ★ A′(pΔ0.05)가 Isaac 에서 진 건 pΔ 탓이 아니라 R 을 2 로 같이 올려 비가 1.80(나쁜 대역)이 된 탓 — 여기서 분리한다.
R05=$(python3 -c "print(round($RV*2.5,2))"); R10=$(python3 -c "print(round($RV*3.5,2))")   # 사용자 지정 배율 2.5 / 3.5
cat > configs/grids/n57_P_sweep.yaml <<YML
# 09-23 01:45 P(K gain) 스윕. 보상 $CFA/$CD/$BO · 기준 R $RV(pΔ0.01). pΔ 0.05 → R $R05 · pΔ 0.1 → R $R10 (같은 R/P_zz_σ 비)
base: [configs/newenv_v5.yaml, configs/overlays/surrogate_knn_v2.yaml]
outdir: results/claudecodefortest/n57_P_sweep
seeds: [42, 43]
stages: {a: {}}
learners:
  s: {device: gpu, set: {agent.type: swirl, run.device: cuda, log.eval_n: 100, log.probe_every: 20, log.probe_n: 8, agent.swirl.q: 1.0e-3, agent.swirl.huber_c: 0, agent.swirl.argmax: spas, agent.swirl.act: target, agent.swirl.N: 10, agent.batch: 128, agent.hidden: [24, 24], agent.tau: 0.005, agent.update_interval: 1, agent.buffer: 50000, reward.c_fa: $CFA, reward.c_d: $CD, reward.bonus: $BO},
      axes: {p: [{_name: P005, agent.swirl.p_delta: 0.05, agent.swirl.alpha: 0.05, agent.swirl.R: $R05},
                 {_name: P010, agent.swirl.p_delta: 0.10, agent.swirl.alpha: 0.03, agent.swirl.R: $R10}]}}
YML
echo "- $(date '+%H:%M') chain_0923c: P 스윕 시작 (pΔ0.05·α0.05·R $R05 / pΔ0.1·α0.03·R $R10, 기준 pΔ0.01·R $RV 은 ①에 있음)" >> $L
python3 scripts/grid.py run configs/grids/n57_P_sweep.yaml --gpu 4 --cpu 0 --omp-cpu 3 >> results/claudecodefortest/n57.launch.log 2>&1
PB=$(python3 $S/pick_P.py "$RV" "$R05" "$R10" 2>>$L); echo "- $(date '+%H:%M') chain_0923c: P 확정 = $PB" >> $L
set -- $PB; PD=$1; AL=$2; RV=$3
# ④ N 스윕 8 / 12 (10 은 앞 단계에 있음)
cat > configs/grids/n58_N_sweep.yaml <<YML
base: [configs/newenv_v5.yaml, configs/overlays/surrogate_knn_v2.yaml]
outdir: results/claudecodefortest/n58_N_sweep
seeds: [42, 43]
stages: {a: {}}
learners:
  s: {device: gpu, set: {agent.type: swirl, run.device: cuda, log.eval_n: 100, log.probe_every: 20, log.probe_n: 8, agent.swirl.R: $RV, agent.swirl.q: 1.0e-3, agent.swirl.huber_c: 0, agent.swirl.argmax: spas, agent.swirl.act: target, agent.swirl.p_delta: $PD, agent.swirl.alpha: $AL, agent.batch: 128, agent.hidden: [24, 24], agent.tau: 0.005, agent.update_interval: 1, agent.buffer: 50000, reward.c_fa: $CFA, reward.c_d: $CD, reward.bonus: $BO},
      axes: {agent.swirl.N: [8, 12]}}
YML
echo "- $(date '+%H:%M') chain_0923c: N 스윕 8/12 시작 (P $PD·α $AL)" >> $L
python3 scripts/grid.py run configs/grids/n58_N_sweep.yaml --gpu 4 --cpu 0 --omp-cpu 3 >> results/claudecodefortest/n58.launch.log 2>&1
NB=$(python3 $S/pick_N.py "$RV" "$PD" 2>>$L); echo "- $(date '+%H:%M') chain_0923c: N 확정 = $NB" >> $L
# ⑤ Isaac 연결
until [ "$(ps -eo args | grep -c '[t]rain.py --config')" -eq 0 ]; do sleep 60; done
B=results/claudecodefortest/isaac_v9; mkdir -p $B
echo "- $(date '+%H:%M') chain_0923c: Isaac s42 짝 기동 — 보상 $CFA/$CD/$BO · R $RV · P $PD · α $AL · N $NB" >> $L
bash scripts/isaac_run.sh $B/swirl_final_s42 --config configs/isaac_v5_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/swirl_final_s42 \
     --set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=128 $RW --set agent.swirl.R=$RV --set agent.swirl.p_delta=$PD --set agent.swirl.alpha=$AL --set agent.swirl.N=$NB
python3 $S/report_run.py $B/swirl_final_s42 >> results/claudecodefortest/TUNE_REPORT.md 2>&1 || true
bash scripts/isaac_run.sh $B/adam_final_s42 --config configs/isaac_v5_adam.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/adam_final_s42 --set agent.batch=128 $RW
python3 $S/report_run.py $B/adam_final_s42 >> results/claudecodefortest/TUNE_REPORT.md 2>&1 || true
echo "- $(date '+%H:%M') chain_0923c 완료" >> $L; touch $B/CHAIN_DONE
