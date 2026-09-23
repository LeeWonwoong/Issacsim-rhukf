#!/usr/bin/env bash
# 09-21 18:50 사용자 지시: Isaac 우선. surrogate GPU 격자(n16b·n35b·n36) 끝나면 시드 42→43→44 순으로
#   Isaac SWIRL(GPU 단독, learn-step 예산) → [Isaac Adam(CPU) ∥ n29b 시드별 분할 격자(GPU 4)] 반복. n29b 는 Adam 창에서만 GPU 를 쓴다.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
# 09-22 02:45 바람 범위: n40 2시드 판정(U(0,9) 격차 F1 +0.055·FA −0.28, SWIRL F1 0.848·FPR 0.009) → 기본 U(0,9). U(0,6) 로 바꾸려면 WIND="[0.0, 6.0]" 로 재기동.
WIND="${WIND:-[0.0, 9.0]}"
SW_SET="${SW_SET:-}"   # 예: "--set agent.batch=128 --set agent.swirl.N=10" (n41 판정 결과, chain_0922.sh 가 채움)
AD_SET="${AD_SET:-}"
L=results/claudecodefortest/NIGHT_0919.md
for g in n16b_obsdiv_v4 n35b_axisB_pd01 n36_windpulse; do until [ -f results/claudecodefortest/$g/GRID_DONE ]; do sleep 120; done; done
until [ "$(ps -eo args | grep -v grep | grep -c 'train.py')" -eq 0 ]; do sleep 60; done
echo "- $(date '+%H:%M') chain_isaac_v4: GPU 비움 → 시작" >> $L
for s in 42 43 44; do
  OUT=results/claudecodefortest/isaac_v4/swirl_s$s
  if [ ! -f "$OUT/RUN_DONE" ]; then
    bash scripts/isaac_run.sh "$OUT" --config configs/isaac_v4_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=$s --set run.outdir=$OUT --set "scenario.wind.range=$WIND" $SW_SET
    echo "- $(date '+%H:%M') chain_isaac_v4: swirl s$s rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
    [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
  fi
  OUT=results/claudecodefortest/isaac_v4/adam_s$s
  if [ ! -f "$OUT/RUN_DONE" ]; then
    ADF=results/claudecodefortest/isaac_v4/ADAM_SET
    if [ "${AD_WAIT:-0}" = 1 ]; then until [ -f $ADF ]; do sleep 120; done; fi
    [ -f $ADF ] && AD_SET="$(cat $ADF)"
    # n29b 분할 격자(GPU)를 Adam(CPU) 런과 병행
    if [ ! -f results/claudecodefortest/n29b_ktd_q_v4_s$s/GRID_DONE ]; then
      python3 scripts/grid.py run configs/grids/n29b_s$s.yaml --gpu 4 --cpu 0 --omp-cpu 2 > results/claudecodefortest/n29b_s$s.launch.log 2>&1 &
      echo "- $(date '+%H:%M') chain_isaac_v4: n29b s$s 격자 병행 시작" >> $L
    fi
    bash scripts/isaac_run.sh "$OUT" --config configs/isaac_v4_adam.yaml --config configs/overlays/isaac.yaml --set run.seed=$s --set run.outdir=$OUT --set "scenario.wind.range=$WIND" $AD_SET
    echo "- $(date '+%H:%M') chain_isaac_v4: adam s$s rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
    [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
    until [ -f results/claudecodefortest/n29b_ktd_q_v4_s$s/GRID_DONE ]; do sleep 120; done   # 다음 SWIRL 전에 GPU 비움
  fi
done
echo "- $(date '+%H:%M') chain_isaac_v4 완료" >> $L
