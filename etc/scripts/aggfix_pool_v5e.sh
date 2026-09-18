#!/usr/bin/env bash
# aggfix_pool_v5e (2026-09-15 저녁): aggressive 설정점 위상 수정 → ① Isaac cert_AGG 재캡처 ② 풀 v5e(구 캡처의 aggressive 제외 + 새 캡처) ③ 검증 ④ 사슬 전환 표식
#   cert_AGG: aggressive × δ{0,0.1,…,0.6,0.7,0.76,0.80,0.84}(11) × ws{0,7,10} × {track,dhover3} × 2ep = 132 ep (~50 min 추정)
#   실행 중 chain_hz/ISAAC_BUSY 로 사슬 단계 착수를 막는다. 실패하면 POOL_V5E_READY 를 만들지 않아 사슬은 v5d 로 진행.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
R=results/claudecodefortest; N=$R/night; C=$N/chain_hz; OUT=$R/cert_AGG
LOG=$R/isaac_commit.log; log(){ echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG | tee -a $C/chain.log >/dev/null; echo "$1"; }
cnt(){ ps aux | grep -cE "$1"; }
exec 8>$C/.aggpool.lock; flock -n 8 || { echo "이미 실행 중"; exit 1; }
[ "$(cnt '[c]hain_hz.py work')" -eq 0 ] || { log "aggfix_pool_v5e: 사슬 워커가 이미 GPU 사용 중 — 중단"; exit 1; }
touch $C/ISAAC_BUSY; trap 'rm -f $C/ISAAC_BUSY' EXIT
while [ "$(cnt '[o]nline_rl_main.py')" -gt 0 ] || [ "$(cnt '[e]nv_scan_v33.py')" -gt 0 ]; do sleep 30; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN ATK_ON_LO ATK_ON_HI ATK_OFF_LO ATK_OFF_HI
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SPEED_MOD_AMP=0 FLIP_TERMINAL=0 HOVER_DRIFT_SYM=1
export ATK_FAMILY=v5 ATK_DELTA_LO=0.10 ATK_SPLIT=0.72 ATK_DELTA_HI=0.84 ATK_P_UPPER=0.5 ATK_ON_LO=25 ATK_ON_HI=40 ATK_START_LO=60 ATK_START_HI=200 PROB_NO_ATTACK=0.5
export UKF_Q_GYRO=2e-3 UKF_R_GYRO=0.2 UKF_Q_EULER=2e-3 UKF_Q_VEL=1e-3 UKF_R_VEL=0.1 NIS_CLIP=4.0
export BUFFER_SIZE=50000 NET_HIDDEN=16 TERMINAL_PEN=0
export FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8"
[ -f $R/FS_PARAMS.env ] && source $R/FS_PARAMS.env
kill_all(){ pkill -x MicroXRCEAgent 2>/dev/null||true; pkill -x px4 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
nvidia-smi >/dev/null 2>&1 || { log "GPU 불가 — 중단"; exit 1; }
if [ ! -f $OUT/sweep_summary.csv ] || [ "$(($(wc -l < $OUT/sweep_summary.csv)-1))" -lt 120 ]; then
  rm -rf $OUT; mkdir -p $OUT; kill_all; log "cert_AGG START (aggressive 수정 코드 × δ11 × ws0/7/10 × track,dhover3 × 2ep)"
  env SWEEP_ATTACK_TYPE=tilt WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
  timeout 9000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack --log-zu \
    --capture-patterns aggressive --capture-disturbances none:0,wind_turbulence:7,wind_turbulence:10 \
    --capture-biases 0.0,0.436,0.872,1.308,1.744,2.18,2.616,3.052,3.314,3.488,3.662 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  kill_all
fi
rows=$(($(wc -l < $OUT/sweep_summary.csv 2>/dev/null || echo 1)-1)); errs=$(grep -c -E 'Traceback|HARD_RESET' $OUT/run.log 2>/dev/null || echo 0)
log "cert_AGG END rows=$rows err=$errs"
[ "$rows" -ge 120 ] && [ -f $OUT/zu_log.npz ] || { log "⚠ cert_AGG 불완전(rows $rows) — 풀 v5e 미구축, 사슬은 v5d 유지"; exit 1; }
log "풀 v5e 구축 시작 (구 캡처 aggressive 제외 + cert_AGG·aggfix_verify)"
POOL_EXCLUDE_PATTERN=aggressive POOL_EXCLUDE_KEEP=cert_AGG,aggfix_verify POOL_EXTRA_DIRS=cert_AGG,aggfix_verify \
  python3 etc/scripts/build_pool_v5c.py --out $N/train_pool_v5e.npz --procs 20 > $N/pool_v5e_build.log 2>&1 || { log "⚠ 풀 v5e 구축 실패 — 사슬은 v5d 유지"; exit 1; }
python3 - <<'PY' > $N/POOL_V5E_CHECK.md 2>&1
import numpy as np
N='results/claudecodefortest/night'; a=np.load(f'{N}/train_pool_v5d.npz'); b=np.load(f'{N}/train_pool_v5e.npz')
print('# 풀 v5d vs v5e (aggressive 위상 수정) — 자동 생성 aggfix_pool_v5e.sh'); print()
print('| 키 | v5d n | v5e n | v5d gyro p50/p95/p99 | v5e gyro p50/p95/p99 | v5d vel p95 | v5e vel p95 |'); print('|---|---|---|---|---|---|---|')
bad = []
for k in sorted(set(x[2:] for x in a.files if x.startswith('g_'))):
    ga, gb, va, vb = a['g_'+k], b.get('g_'+k), a['v_'+k], b.get('v_'+k)
    if gb is None: bad.append(k); continue
    q=lambda x: '/'.join('%.2f'%np.percentile(x,p) for p in (50,95,99))
    print(f'| {k} | {len(ga)} | {len(gb)} | {q(ga)} | {q(gb)} | {np.percentile(va,95):.2f} | {np.percentile(vb,95):.2f} |')
print(); print('누락 키:', bad or '없음')
PY
grep -q "누락 키: 없음" $N/POOL_V5E_CHECK.md || { log "⚠ 풀 v5e 키 누락 — POOL_V5E_CHECK.md 확인, 사슬은 v5d 유지"; exit 1; }
touch $C/POOL_V5E_READY; log "풀 v5e 준비 완료 → 사슬 S1 부터 v5e 사용 (night/POOL_V5E_CHECK.md)"
