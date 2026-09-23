#!/usr/bin/env bash
# 09-22 21:40 사용자: R(TD var) → P·q(gain) → N(지평, sweet spot) 순서. chain_0922k 완료 후 R∈{0.5,0.35,0.25} 중 후반 공격F1+리턴 최선을 골라 P{0.005,0.02} → q{3e-4,3e-3} → N{4,8,10}
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v8
until [ -f $B/CHAIN_DONE ]; do sleep 120; done
RB=$(python3 - <<'PY'
import csv, numpy as np, os
B='results/claudecodefortest/isaac_v8/'; best=None
for R,d in ((0.5,'swirl_ui1_P01_R05_s42'),(0.35,'swirl_ui1_P01_R035_s42'),(0.25,'swirl_ui1_P01_R025_s42')):
    p=B+d+'/metrics_rhukf.csv'
    if not os.path.exists(p): continue
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f1=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
    m=(ep>=150)&(b>0); score=f1[m].mean()+ (rw[ep>=150].mean()-280)/100
    if best is None or score>best[0]: best=(score,R)
print(best[1] if best else 0.5)
PY
)
echo "- $(date '+%H:%M') chain_0922l: R 확정 = $RB → P·q·N 순" >> $L
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=256"
run(){ local OUT=$1; shift 1
  [ -f "$OUT/RUN_DONE" ] && return
  echo "- $(date '+%H:%M') chain_0922l: $(basename $OUT) 시작" >> $L
  bash scripts/isaac_run.sh "$OUT" --config configs/isaac_v5_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT $UI --set agent.swirl.R=$RB "$@"
  echo "- $(date '+%H:%M') chain_0922l: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1; }
run $B/swirl_ui1_P005_s42  --set agent.swirl.p_delta=0.005 --set agent.swirl.alpha=0.14 --set agent.swirl.N=6
run $B/swirl_ui1_P02_s42   --set agent.swirl.p_delta=0.02  --set agent.swirl.alpha=0.07 --set agent.swirl.N=6
run $B/swirl_ui1_q3e4_s42  --set agent.swirl.p_delta=0.01  --set agent.swirl.alpha=0.1  --set agent.swirl.q=3.0e-4 --set agent.swirl.N=6
run $B/swirl_ui1_q3e3_s42  --set agent.swirl.p_delta=0.01  --set agent.swirl.alpha=0.1  --set agent.swirl.q=3.0e-3 --set agent.swirl.N=6
run $B/swirl_ui1_N4_s42    --set agent.swirl.p_delta=0.01  --set agent.swirl.alpha=0.1  --set agent.swirl.N=4
run $B/swirl_ui1_N8_s42    --set agent.swirl.p_delta=0.01  --set agent.swirl.alpha=0.1  --set agent.swirl.N=8
run $B/swirl_ui1_N10_s42   --set agent.swirl.p_delta=0.01  --set agent.swirl.alpha=0.1  --set agent.swirl.N=10
echo "- $(date '+%H:%M') chain_0922l 완료 (P·q·N)" >> $L; touch $B/TUNE_DONE
