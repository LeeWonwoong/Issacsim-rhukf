#!/usr/bin/env bash
# 09-22 22:50 (R 규칙 갱신: 필터 일관성 NIS 로 P_zz_σ 역산 → R = ρ·P_zz_σ, 실측상 ρ≤1.0 이 좋음. 새 보상 3종 시작 R=0.6)
# 09-22 22:35 사용자: ③ FN1.0 짝으로 격차 확인 → ③b (1,1,1,6) → ③c (1,2,2,5, R 재튜닝) → [③d (1,2,1,6) 분석 후보] → 보상 확정 → R*→P→N 튜닝.
#   SWIRL 스케줄 Adam 과 동일(ui1·τ0.005·B256) 고정. 각 보상의 R 은 결정항 rms 비로 예측(v5 R0.5 기준 → 모두 0.8), 확정 뒤 측정 Zvar 로 1회 보정.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v8
S=/tmp/claude-1001/-home-acsl-projects-Issacsim-rhukf/c2b43851-6cf0-4b54-a537-da44ef7e978d/scratchpad
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=256"
P0="--set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.N=6"
run(){ local OUT=$1; local CFG=$2; shift 2
  [ -f "$OUT/RUN_DONE" ] && return 0
  echo "- $(date '+%H:%M') chain_0922n: $(basename $OUT) 시작" >> $L
  bash scripts/isaac_run.sh "$OUT" --config "$CFG" --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT "$@"
  echo "- $(date '+%H:%M') chain_0922n: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  [ -f "$OUT/final_model.pt" ] && python3 scripts/eval_model.py "$OUT" --n 100 --device cpu >> "$OUT/eval_surrogate.log" 2>&1
  python3 $S/report_run.py "$OUT" >> results/claudecodefortest/TUNE_REPORT.md 2>&1; }
until [ -f $B/adam_FN10_s42/RUN_DONE ]; do sleep 60; done
python3 $S/report_run.py $B/adam_FN10_s42 >> results/claudecodefortest/TUNE_REPORT.md 2>&1
# ── 보상 후보 3종 짝 (SWIRL 은 예측 R0.8) ──────────────────────────
R_1116="--set reward.c_fa=1.0 --set reward.c_d=1.0 --set reward.bonus=6.0"
R_1225="--set reward.c_fa=2.0 --set reward.c_d=2.0 --set reward.bonus=5.0"
R_1216="--set reward.c_fa=2.0 --set reward.c_d=1.0 --set reward.bonus=6.0"
run $B/swirl_1116_R06_s42 configs/isaac_v5_swirl.yaml $UI $P0 $R_1116 --set agent.swirl.R=0.6
run $B/adam_1116_s42      configs/isaac_v5_adam.yaml  $R_1116
run $B/swirl_1225_R06_s42 configs/isaac_v5_swirl.yaml $UI $P0 $R_1225 --set agent.swirl.R=0.6
run $B/adam_1225_s42      configs/isaac_v5_adam.yaml  $R_1225
[ -f $B/SKIP_1216 ] || { run $B/swirl_1216_R06_s42 configs/isaac_v5_swirl.yaml $UI $P0 $R_1216 --set agent.swirl.R=0.6
                         run $B/adam_1216_s42      configs/isaac_v5_adam.yaml  $R_1216; }
# ── 보상 확정: 같은 시드 짝 격차(공격에피 ΔF1 + Δ리턴/100) 최대, 단 SWIRL 절대 F1 ≥ 0.70 ─────
PICK=$(python3 - <<'PY'
import csv, numpy as np, os
B='results/claudecodefortest/isaac_v8/'; V='results/claudecodefortest/isaac_v7/'
def M(p):
    if not os.path.exists(p): return None
    r=list(csv.DictReader(open(p))); return {k:np.array([float(x[k]) for x in r]) for k in ('episode','reward','f1','bias_scale')}
def gap(s,a):
    S,A=M(s),M(a)
    if S is None or A is None: return None
    si={int(e):i for i,e in enumerate(S['episode'])}; ai={int(e):i for i,e in enumerate(A['episode'])}; c=[e for e in si if e in ai and e>=100]
    if len(c)<50: return None
    atk=np.array([S['bias_scale'][si[e]]>0 for e in c])
    df=np.array([S['f1'][si[e]]-A['f1'][ai[e]] for e in c])[atk].mean()
    dr=np.array([S['reward'][si[e]]-A['reward'][ai[e]] for e in c]).mean()
    m=(S['episode']>=150)&(S['bias_scale']>0); f=S['f1'][m].mean()
    return df+dr/100.0, f, df, dr
C={'v5': (B+'swirl_ui1_P01_R05_s42/metrics_rhukf.csv', V+'adam_s42/metrics_adam.csv'),
   'FN10': (B+'swirl_ui1_P01_R1_FN10_s42/metrics_rhukf.csv', B+'adam_FN10_s42/metrics_adam.csv'),
   '1116': (B+'swirl_1116_R06_s42/metrics_rhukf.csv', B+'adam_1116_s42/metrics_adam.csv'),
   '1225': (B+'swirl_1225_R06_s42/metrics_rhukf.csv', B+'adam_1225_s42/metrics_adam.csv'),
   '1216': (B+'swirl_1216_R06_s42/metrics_rhukf.csv', B+'adam_1216_s42/metrics_adam.csv')}
best=None
for k,(s,a) in C.items():
    g=gap(s,a)
    if g is None: continue
    print(f'{k}: 점수 {g[0]:+.3f} (ΔF1 {g[2]:+.3f} Δ리턴 {g[3]:+.1f}) SWIRL F1 {g[1]:.3f}', flush=True)
    if g[1] >= 0.70 and (best is None or g[0] > best[0]): best=(g[0], k)
print('PICK', best[1] if best else 'v5')
PY
)
echo "$PICK" >> $L; TAG=$(echo "$PICK" | awk '/^PICK/{print $2}')
case $TAG in 1116) RW="$R_1116"; R0=0.6;; 1225) RW="$R_1225"; R0=0.6;; 1216) RW="$R_1216"; R0=0.6;; FN10) RW="--set reward.c_d=1.0"; R0=0.4;; *) TAG=v5; RW=""; R0=0.5;; esac
echo "- $(date '+%H:%M') chain_0922n: 보상 확정 = $TAG (시작 R $R0)" >> $L
# ── R*: 측정 Zvar 로 1회 보정 ────────────────────────────────────
SRC=$(ls -d $B/swirl_${TAG}_R0*_s42 2>/dev/null | head -1); [ -z "$SRC" ] && SRC=$B/swirl_ui1_P01_R05_s42
RES=$(python3 scripts/pick_R.py $SRC 1.0 2>>$L || echo "$R0 0 0 0"); set -- $RES; RSTAR=$1
RLOW=$(python3 -c "print(round(max(0.1, 0.6*$2),1))")   # 비 0.6 (v2 가 0.51 이었다)
echo "- $(date '+%H:%M') chain_0922n: R* = $RSTAR (Zvar $2 / 기준 $3, Kg $4)" >> $L
if [ "$RSTAR" != "$R0" ]; then run $B/swirl_${TAG}_R${RSTAR}_s42 configs/isaac_v5_swirl.yaml $UI $P0 $RW --set agent.swirl.R=$RSTAR; fi
if [ "$RLOW" != "$RSTAR" ] && [ "$RLOW" != "$R0" ]; then run $B/swirl_${TAG}_R${RLOW}_s42 configs/isaac_v5_swirl.yaml $UI $P0 $RW --set agent.swirl.R=$RLOW; fi
RBEST=$(python3 - <<PY
import csv, numpy as np, glob, os
best=None
for p in glob.glob('results/claudecodefortest/isaac_v8/swirl_${TAG}_R*_s42/metrics_rhukf.csv'):
    R=float(p.split('_R')[-1].split('_')[0].replace('0','0.',1)) if False else None
    import re; m=re.search(r'_R([0-9.]+)_s42', p); R=float(m.group(1)) if m else None
    if R is None: continue
    if R>3: R=R/10.0
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
    mm=(ep>=150)&(b>0); s=f[mm].mean()+(rw[ep>=150].mean()-280)/100
    if best is None or s>best[0]: best=(s,R)
print(best[1] if best else $R0)
PY
)
echo "- $(date '+%H:%M') chain_0922n: R 확정 = $RBEST → P 스캔" >> $L
run $B/swirl_${TAG}_P0005_s42 configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.005 --set agent.swirl.alpha=0.17  --set agent.swirl.R=$RBEST --set agent.swirl.N=6
run $B/swirl_${TAG}_P002_s42  configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=0.02  --set agent.swirl.alpha=0.085 --set agent.swirl.R=$RBEST --set agent.swirl.N=6
PN=$(python3 - <<PY
import csv, numpy as np, glob, re
best=None
for p in glob.glob('results/claudecodefortest/isaac_v8/swirl_${TAG}_*_s42/metrics_rhukf.csv'):
    P = 0.005 if 'P0005' in p else (0.02 if 'P002' in p else 0.01)
    r=list(csv.DictReader(open(p))); ep=np.array([float(x['episode']) for x in r]); f=np.array([float(x['f1']) for x in r]); b=np.array([float(x['bias_scale']) for x in r]); rw=np.array([float(x['reward']) for x in r])
    mm=(ep>=150)&(b>0)
    if mm.sum()<10: continue
    s=f[mm].mean()+(rw[ep>=150].mean()-280)/100
    if best is None or s>best[0]: best=(s,P)
import math; P=best[1] if best else 0.01; print(P, round(0.012/math.sqrt(P),3))
PY
)
set -- $PN; PBEST=$1; ABEST=$2
echo "- $(date '+%H:%M') chain_0922n: P 확정 = $PBEST (α $ABEST) → N sweet spot" >> $L
for N in 4 10 14; do
  run $B/swirl_${TAG}_N${N}_s42 configs/isaac_v5_swirl.yaml $UI $RW --set agent.swirl.p_delta=$PBEST --set agent.swirl.alpha=$ABEST --set agent.swirl.R=$RBEST --set agent.swirl.N=$N
done
echo "- $(date '+%H:%M') chain_0922n 튜닝 완료: 보상 $TAG · R $RBEST · P $PBEST · N 스캔" >> $L; touch $B/TUNE_DONE
