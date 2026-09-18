#!/usr/bin/env bash
# reward_variants — batch3b(1~9) 완료 대기 → 최적 RHUKF config 선정 → 3 변형 Isaac 측정 (2026-08-29)
#   V1_fpescal: FP 에스컬레이션(첫 -0.2 → 지속 -1.2)   [FP_ESCAL=1]
#   V2_rhalf : reward 전항 ÷2 (비율 동일·크기 절반)     [--reward-scale 0.5]
#   V3_clip35: 관측 클립 상한 3.0 → 3.5 (나눗셈 없음)   [OBS_CLIP=3.5]
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=reward_variants.log
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
echo "[$(date '+%m-%d %H:%M')] 변형체인 기동 — batch3b 완료 대기" | tee -a $LOG
while [ ! -f TUNE_BATCH3_DONE ]; do sleep 300; done
sleep 30
echo "[$(date '+%m-%d %H:%M')] batch3b 완료 감지 → 최적 config 선정" | tee -a $LOG

# 최적 RHUKF config 선정 (F1 우선, reward 타이브레이크) → best_env.txt
python3 - <<'PY' | tee -a $LOG
import glob,csv,os,numpy as np
ENVMAP={
 'A_r10_pd001':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.01',
 'A_r10_pd005':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.0 RHUKF_PD=0.05',
 'A_r15_pd001':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.5 RHUKF_PD=0.01',
 'A_r15_pd005':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.02 RHUKF_UI=4 RHUKF_R=1.5 RHUKF_PD=0.05',
 'B_r10_pd001':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.0 RHUKF_PD=0.01',
 'B_r10_pd005':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.0 RHUKF_PD=0.05',
 'B_r15_pd001':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.01',
 'B_r15_pd005':'NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.05',
}
best=None
for name,envs in ENVMAP.items():
    f=f'results_g1_{name}/metrics_rhukf.csv'
    if not os.path.exists(f): continue
    rows=list(csv.DictReader(open(f)))
    atk=[r for r in rows if float(r.get('tp',0))+float(r.get('fn',0))>0]
    if len(atk)<10: continue
    f1=np.mean([float(r['f1']) for r in atk[-20:]])
    rwd=np.mean([float(r['reward']) for r in rows[-10:]])
    print(f"  {name}: F1={f1:.3f} rwd={rwd:.1f} ep={len(rows)}")
    key=(round(f1,3), rwd)
    if best is None or key>best[0]: best=(key,name,envs)
print(f"★ 최적: {best[1]} (F1 {best[0][0]:.3f} rwd {best[0][1]:.1f})")
open('best_env.txt','w').write(best[2]+'\n'+best[1]+'\n')
PY
BESTENV=$(head -1 best_env.txt); BESTNAME=$(sed -n 2p best_env.txt)
echo "[$(date '+%m-%d %H:%M')] 최적=$BESTNAME → 3 변형 시작" | tee -a $LOG

kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }
run(){ # $1=이름 $2=추가env $3=추가cli
  local OUT=results_v_$1
  if [ -f $OUT/RUN_DONE ]; then echo "[$(date '+%m-%d %H:%M')] $1 SKIP" | tee -a $LOG; return; fi
  echo "[$(date '+%m-%d %H:%M')] $1 START env='$BESTENV $2' cli='$3'" | tee -a $LOG
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $BESTENV $2 timeout 12600 python3 -u online_rl_main.py --headless --agent rhukf \
    --max-ep 200 --speed 1 --seed 42 --outdir $OUT $3 > $OUT/train.log 2>&1
  local rc=$?; local ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)
  echo "[$(date '+%m-%d %H:%M')] $1 END rc=$rc ep=$ep" | tee -a $LOG
  if [ "$ep" -ge 150 ] || [ "$rc" = "0" ]; then touch $OUT/RUN_DONE; fi
}
run fpescal "FP_ESCAL=1" ""
run rhalf   ""           "--reward-scale 0.5"
run clip35  "OBS_CLIP=3.5" ""
kill_isaac
echo "════ 변형 3런 완료 $(date '+%m-%d %H:%M') ════" | tee -a $LOG
touch REWARD_VARIANTS_DONE
