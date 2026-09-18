#!/usr/bin/env bash
# speed_ab_verify v2 (2026-09-01): sim-time 타이머 완전수정 검증 — 양 옵티마이저.
#   타이머 수정은 speed1 에서 주기 무변경 → 기존 batch3b speed-1 런이 그대로 기준선.
#   ∴ speed3 두 런(swirl·adam)만 돌려 기존과 per-에피 대조 + RTF 달성치 + learn-skip 확인.
#   ★ 핵심: RHUKF learn(18~24ms) 이 Adam(2~4ms) 보다 무거워 per-optimizer 지속속도가 다름.
#     speed3 예산=33ms > RHUKF 24ms → 이론상 fit. 실측으로 확인.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
LOG=speed_ab_verify.log
# g1 env (기존 batch3b 기준선과 동일 — 재현 대조 성립 조건)
export SENSOR_NOISE_SCALE=1.0 WIND_MOMENT_ARM=0.02 EP_MAX_STEPS=400
export SPEED_MOD_AMP=0.5 SPEED_MOD_FREQ=0.5 COM_BIAS_STD=0.05
EP=40
# B_r15_pd001 (batch3b 최적 RHUKF) 과 정확히 동일 config
CB_SWIRL="NET_HIDDEN=16 RHUKF_N=5 RHUKF_Q=1e-3 RHUKF_TAU=0.005 RHUKF_UI=1 RHUKF_R=1.5 RHUKF_PD=0.01"
CA_ADAM="NET_HIDDEN=16"    # adam_n16 과 동일 (AMSGrad 기본 ON)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

# ── stealth_sweep 완전 종료 대기 (Isaac 충돌 방지) ──
say "대기: stealth_sweep(s05_adam) 종료 + online_rl 소멸..."
while true; do
  running=$(ps -eo args | grep online_rl_main | grep -v grep | wc -l)
  [ -f results_ss_s05_adam/RUN_DONE ] && [ "$running" -eq 0 ] && break
  sleep 30
done
say "sweep 종료 확인 → A/B 시작."

kill_isaac(){ for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 5; }

run(){ local OUT=$1 AG=$2 ENVC=$3 SPD=$4
  say "$OUT START (agent=$AG speed=$SPD seed42 ep=$EP)"
  kill_isaac; rm -rf $OUT; mkdir -p $OUT
  env $ENVC timeout 7200 python3 -u online_rl_main.py --headless --agent $AG \
      --max-ep $EP --speed $SPD --seed 42 --outdir $OUT > $OUT/train.log 2>&1
  say "$OUT END rc=$? ep=$(grep -acE 'TRAIN Ep [0-9]+/' $OUT/train.log 2>/dev/null||echo 0)"
  say "  RTF 달성: $(grep -E '\[RTF\]' $OUT/train.log 2>/dev/null | tail -2 | tr '\n' ' ')"
  say "  learn: $(grep -E 'learn-step:' $OUT/train.log 2>/dev/null | tail -1 | sed 's/^ *//')"
}

# ★ 신 코드로 speed1·speed3 both 직접 (apples-to-apples). RHUKF 먼저(learn-bound 바인딩).
run results_spdtest_swirl_spd1 rhukf "$CB_SWIRL" 1
run results_spdtest_swirl_spd3 rhukf "$CB_SWIRL" 3
run results_spdtest_adam_spd1  adam  "$CA_ADAM"  1
run results_spdtest_adam_spd3  adam  "$CA_ADAM"  3
say "A/B 완료. 비교: python3 etc/scripts/speed_ab_compare.py"
kill_isaac
touch SPEED_AB_DONE
