#!/usr/bin/env bash
# 09-22 22:25 사용자: ③b 를 (1,1,1,6) 로 검증 → 성립하면 그 보상으로 R(측정 기반 계산) → P → N 순 튜닝, 그 뒤 ④ 공격 v5a 짝.
#   SWIRL 스케줄은 Adam 과 동일(ui1·τ0.005·B256) 고정. 모든 런 s42, 판독 report_run.py.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v8
S=/tmp/claude-1001/-home-acsl-projects-Issacsim-rhukf/c2b43851-6cf0-4b54-a537-da44ef7e978d/scratchpad
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=256"
R6="--set reward.c_fa=1.0 --set reward.c_d=1.0 --set reward.bonus=6.0"
run(){ local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return 0
  echo "- $(date '+%H:%M') chain_0922m: $(basename $OUT) 시작" >> $L
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922m: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
  python3 $S/report_run.py "$OUT" >> results/claudecodefortest/TUNE_REPORT.md 2>&1; }
# ── ③ Adam FN1.0 (이미 실행 중) 완료 대기
until [ -f $B/adam_FN10_s42/RUN_DONE ]; do sleep 60; done
python3 $S/report_run.py $B/adam_FN10_s42 >> results/claudecodefortest/TUNE_REPORT.md 2>&1
# ── ③b (1,1,1,6) 짝: SWIRL R0.5 + Adam
run $B/swirl_1116_R05_s42 configs/isaac_v5_swirl.yaml $UI $R6 --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.R=0.5 --set agent.swirl.N=6
run $B/adam_1116_s42      configs/isaac_v5_adam.yaml  $R6
# ── 보상 채택 판정: SWIRL 유지(≥ v5·R0.5 −0.05) ∧ Adam 하락(≤ v5 Adam −0.02)
ADOPT=$(python3 - <<'PY'
import csv, numpy as np, os
B='results/claudecodefortest/isaac_v8/'; V='results/claudecodefortest/isaac_v7/'
def f1(p):
    if not os.path.exists(p): return None
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); a=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r])
    m=(ep>=150)&(b>0); return float(a[m].mean())
s_new=f1(B+'swirl_1116_R05_s42/metrics_rhukf.csv'); s_old=f1(B+'swirl_ui1_P01_R05_s42/metrics_rhukf.csv')
a_new=f1(B+'adam_1116_s42/metrics_adam.csv');       a_old=f1(V+'adam_s42/metrics_adam.csv')
ok = (s_new is not None and a_new is not None and s_new >= s_old-0.05 and a_new <= a_old-0.02)
print('1116' if ok else 'v5', s_new, s_old, a_new, a_old)
PY
)
set -- $ADOPT; MODE=$1
echo "- $(date '+%H:%M') chain_0922m: 보상 판정 = $MODE (SWIRL $2 vs $3 / Adam $4 vs $5)" >> $L
if [ "$MODE" = "1116" ]; then RW="$R6"; TAG=1116; else RW=""; TAG=v5; fi
# ── R: 측정 기반 목표 계산(소수 1자리) + 브래킷 1점
RES=$(python3 scripts/pick_R.py $B/swirl_${TAG}_R05_s42 $B/swirl_ui1_P01_R05_s42 0.5 2>>$L || echo "0.5 0 0 0")
set -- $RES; RSTAR=$1
echo "- $(date '+%H:%M') chain_0922m: R* = $RSTAR (Zvar $2 / 기준 $3, Kg $4)" >> $L
RALT=$(python3 -c "r=$RSTAR; print(round(max(0.1, r-0.3),1) if r>=0.6 else round(r+0.3,1))")
run $B/swirl_${TAG}_R${RSTAR}_s42 configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.R=$RSTAR --set agent.swirl.N=6
run $B/swirl_${TAG}_R${RALT}_s42  configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.R=$RALT  --set agent.swirl.N=6
RBEST=$(python3 - <<PY
import csv, numpy as np, os, glob
B='results/claudecodefortest/isaac_v8/'; best=None
for R in (0.5, $RSTAR, $RALT):
    for p in glob.glob(B+f'swirl_${TAG}_R{R}_s42/metrics_rhukf.csv')+ (glob.glob(B+'swirl_ui1_P01_R05_s42/metrics_rhukf.csv') if (R==0.5 and '${TAG}'=='v5') else []):
        r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
        m=(ep>=150)&(b>0); s=f[m].mean()+(rw[ep>=150].mean()-280)/100
        if best is None or s>best[0]: best=(s,R)
print(best[1] if best else 0.5)
PY
)
echo "- $(date '+%H:%M') chain_0922m: R 확정 = $RBEST → P 스캔" >> $L
# ── P(gain): 0.005 / 0.02 (α√pΔ ≈ 0.012 규칙)
run $B/swirl_${TAG}_P0005_s42 configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.005 --set agent.swirl.alpha=0.17 --set agent.swirl.R=$RBEST --set agent.swirl.N=6
run $B/swirl_${TAG}_P002_s42  configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.02  --set agent.swirl.alpha=0.085 --set agent.swirl.R=$RBEST --set agent.swirl.N=6
PBEST=$(python3 - <<PY
import csv, numpy as np, os
B='results/claudecodefortest/isaac_v8/'; best=None
for P,d in ((0.01,'swirl_${TAG}_R${RBEST}_s42'),(0.01,'swirl_${TAG}_R05_s42'),(0.005,'swirl_${TAG}_P0005_s42'),(0.02,'swirl_${TAG}_P002_s42')):
    p=B+d+'/metrics_rhukf.csv'
    if not os.path.exists(p): continue
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
    m=(ep>=150)&(b>0); s=f[m].mean()+(rw[ep>=150].mean()-280)/100
    if best is None or s>best[0]: best=(s,P)
print(best[1] if best else 0.01)
PY
)
A_OF=$(python3 -c "import math; p=$PBEST; print(round(0.012/math.sqrt(p),3))")
echo "- $(date '+%H:%M') chain_0922m: P 확정 = $PBEST (α $A_OF) → N 스캔" >> $L
# ── N(지평 sweet spot): ui1 이면 창이 N 스텝 = ui4 의 1/4 → 위쪽을 넓게 본다. learn-step 예산 40 ms 확인 필요
for N in 4 10 14; do
  run $B/swirl_${TAG}_N${N}_s42 configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=$PBEST --set agent.swirl.alpha=$A_OF --set agent.swirl.R=$RBEST --set agent.swirl.N=$N
done
echo "- $(date '+%H:%M') chain_0922m: 튜닝 완료(R $RBEST · P $PBEST · N 스캔) → 공격 v5a 짝" >> $L
run $B/swirl_${TAG}_v5a_s42 configs/isaac_v5a_swirl.yaml $UI $RW --set agent.swirl.p_delta=$PBEST --set agent.swirl.alpha=$A_OF --set agent.swirl.R=$RBEST --set agent.swirl.N=6
run $B/adam_${TAG}_v5a_s42  configs/isaac_v5a_adam.yaml  $RW
echo "- $(date '+%H:%M') chain_0922m 완료" >> $L; touch $B/TUNE_DONE
