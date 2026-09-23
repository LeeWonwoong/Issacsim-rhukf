#!/usr/bin/env bash
# 09-22 12:45 확정 무대 Isaac 본선(사용자 조건: 정수 보상 (1,2,3,10)·R1·B256·U(0,7)·v4 공격·관측÷4).
#   SWIRL 노브는 ③(v2 노브) vs ②(현 노브) Isaac s42 결과로 자동 선택(pick_knobs_0922.py). 칼만-TD 표준판 = SWIRL 과 같은 노브·act=active·P 지속. Adam 3e-4·β5·clip10·ui1·τ0.005.
#   시드 42: SWIRL → Adam(∥ surrogate n44 격자 GPU) → EKF → UKF, 이어 43·44. 런당 ≈1 h.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v6; mkdir -p $B
until [ -f results/claudecodefortest/isaac_v5/swirl_v2knobs_s42/RUN_DONE ]; do sleep 60; done
PICK=$(python3 scripts/pick_knobs_0922.py 2>>$B/pick.log) || PICK=lowK
if [ "$PICK" = lowK ]; then KN="--set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.tau=0.02 --set agent.update_interval=4"; KQ="--set agent.swirl.q=0.001";
else KN="--set agent.swirl.p_delta=0.1 --set agent.swirl.alpha=0.05 --set agent.tau=0.005 --set agent.update_interval=1"; KQ="--set agent.swirl.q=0.0001"; fi
echo "$PICK" > $B/SWIRL_KNOBS; echo "$KN" >> $B/SWIRL_KNOBS
echo "- $(date '+%H:%M') chain_0922d: ③ 판정 → SWIRL 노브 = $PICK ($KN). 확정 무대 Isaac 본선 시작 (정수보상·R1·B256·U(0,7))" >> $L
COMMON=(--config configs/overlays/isaac.yaml "--set" "scenario.wind.range=[0.0, 7.0]" --set agent.batch=256)
run(){ local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" "${COMMON[@]}" --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922d: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1; }
for s in 42 43 44; do
  run $B/swirl_s$s configs/isaac_v4_swirl.yaml --set run.seed=$s --set agent.swirl.N=6 $KN
  if [ $s = 42 ] && [ ! -f results/claudecodefortest/n44_locked_stage/GRID_DONE ]; then
    python3 scripts/grid.py run configs/grids/n44_locked_stage.yaml --gpu 4 --cpu 2 --omp-cpu 4 > results/claudecodefortest/n44.launch.log 2>&1 &
    echo "- $(date '+%H:%M') chain_0922d: surrogate n44(확정 무대 4학습기 3시드) Adam 창에서 병행" >> $L
  fi
  run $B/adam_s$s configs/isaac_v4_adam.yaml --set run.seed=$s
  until [ -f results/claudecodefortest/n44_locked_stage/GRID_DONE ]; do sleep 60; done   # GPU 비운 뒤 칼만-TD
  run $B/ekf_s$s configs/isaac_v4_swirl.yaml --set run.seed=$s --set agent.type=ekf --set agent.swirl.act=active $KN $KQ
  run $B/ukf_s$s configs/isaac_v4_swirl.yaml --set run.seed=$s --set agent.type=ukf --set agent.swirl.act=active $KN $KQ
  echo "- $(date '+%H:%M') chain_0922d: 시드 $s 4학습기 완료" >> $L
done
echo "- $(date '+%H:%M') chain_0922d 완료" >> $L; touch $B/CHAIN_DONE
