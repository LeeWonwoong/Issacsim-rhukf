#!/usr/bin/env bash
# 09-23 01:15 밤샘 자동 연결: n55(ρ 스윕) 완료 → SWIRL−Adam 격차 최대 ρ 확정 → 그 보상으로 Isaac s42 짝(SWIRL 최선 R + Adam) 자동 기동.
#   surrogate 가 GPU 를 비운 뒤에만 Isaac 을 띄운다(learn-step 예산 40 ms).
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v9; S=/tmp/claude-1001/-home-acsl-projects-Issacsim-rhukf/c2b43851-6cf0-4b54-a537-da44ef7e978d/scratchpad
mkdir -p $B
until [ -f results/claudecodefortest/n55_rho_sweep/GRID_DONE ]; do sleep 180; done
python3 $S/n53_report.py >> results/claudecodefortest/TUNE_REPORT.md 2>&1 || true
until [ "$(ps -eo args | grep -c '[t]rain.py --config')" -eq 0 ]; do sleep 120; done
PICK=$(python3 - <<'PY'
import json, os, numpy as np
R='results/claudecodefortest/n55_rho_sweep/'
def m(p):
    o=[]
    for s in (42,43,44):
        d=f'{R}{p}_s{s}'
        if os.path.exists(d+'/eval.json'):
            e=json.load(open(d+'/eval.json')); o.append([e['f1'],e['fa_episode_rate'],e['delay']])
    return np.array(o).mean(0) if o else None
best=None
for tag,(sa,sb,ad,cfa,cd,bo) in {
 'rho1':( 'a_s_wr1v5Ra','a_s_wr1v5Rb','a_s_wr1v5Ra',1.5,1.5,4),      # v5 는 Adam 셀이 n55 에 없음 → 아래에서 예외 처리
 'rho2':( 'a_s_wr2_1216Ra','a_s_wr2_1216Rb','a_a_wr2_1216',2.0,1.0,6.0),
 'rho3':( 'a_s_wr3_1316Ra','a_s_wr3_1316Rb','a_a_wr3_1316',3.0,1.0,6.0)}.items():
    A=m(ad)
    if tag=='rho1' or A is None: continue
    cand=[(m(x),x) for x in (sa,sb)]; cand=[(v,x) for v,x in cand if v is not None]
    if not cand: continue
    v,x=max(cand, key=lambda t:t[0][0]); gap=v[0]-A[0]
    print(f'{tag}: SWIRL {v[0]:.3f}({x}) vs Adam {A[0]:.3f} → gap {gap:+.3f}', flush=True)
    if v[0]>=0.60 and (best is None or gap>best[0]): best=(gap, tag, x, cfa, cd, bo)
print('PICK', ' '.join(map(str,best[1:])) if best else 'none')
PY
)
echo "$PICK" >> $L; set -- $(echo "$PICK" | awk '/^PICK/{print $2,$3,$4,$5,$6}')
TAG=$1; CELL=$2; CFA=$3; CD=$4; BO=$5
[ -z "$TAG" ] || [ "$TAG" = none ] && { echo "- $(date '+%H:%M') chain_0923: 판정 불가 → Isaac 보류" >> $L; exit 0; }
RBEST=$(python3 -c "
import re; c='$CELL'
print(re.search(r'R([ab])$', c).group(1))" 2>/dev/null || echo a)
case "$CELL" in *Ra) RV=$( [ "$TAG" = rho2 ] && echo 0.4 || echo 0.5 );; *) RV=$( [ "$TAG" = rho2 ] && echo 0.7 || echo 0.9 );; esac
echo "- $(date '+%H:%M') chain_0923: 보상 $TAG (c_fa $CFA·c_d $CD·bonus $BO), SWIRL R $RV → Isaac s42 짝 기동" >> $L
UI="--set agent.tau=0.005 --set agent.update_interval=1 --set agent.batch=128"
RW="--set reward.c_fa=$CFA --set reward.c_d=$CD --set reward.bonus=$BO"
for X in swirl adam; do
  OUT=$B/${X}_${TAG}_s42; [ -f $OUT/RUN_DONE ] && continue
  if [ $X = swirl ]; then EX="$UI $RW --set agent.swirl.p_delta=0.01 --set agent.swirl.alpha=0.1 --set agent.swirl.N=10 --set agent.swirl.R=$RV"; CFG=configs/isaac_v5_swirl.yaml
  else EX="$RW --set agent.batch=128"; CFG=configs/isaac_v5_adam.yaml; fi
  bash scripts/isaac_run.sh $OUT --config $CFG --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$OUT $EX
  echo "- $(date '+%H:%M') chain_0923: $(basename $OUT) rc=$(cat $OUT/RUN_DONE 2>/dev/null)" >> $L
  python3 $S/report_run.py $OUT >> results/claudecodefortest/TUNE_REPORT.md 2>&1 || true
done
echo "- $(date '+%H:%M') chain_0923 완료" >> $L
