#!/usr/bin/env bash
# 09-22 14:52 사용자 '고': ② Isaac s42 SWIRL v5 후보 B → export → 헤드리스 리허설 → ③ 후보 A′(pΔ0.05·R2) → export → 짝비교
cd /home/acsl/projects/Issacsim-rhukf || exit 1
L=results/claudecodefortest/NIGHT_0919.md; B=results/claudecodefortest/isaac_v7; mkdir -p $B px4_field/models results/claudecodefortest/f13_rehearsal/figs
echo "- $(date '+%H:%M') chain_0922e 시작: Isaac s42 SWIRL v5 후보 B" >> $L
bash scripts/isaac_run.sh $B/swirlB_s42 --config configs/isaac_v5_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/swirlB_s42
echo "- $(date '+%H:%M') chain_0922e: swirlB_s42 rc=$(cat $B/swirlB_s42/RUN_DONE)" >> $L
if [ -f $B/swirlB_s42/final_model.pt ]; then
  python3 px4_field/export_policy.py $B/swirlB_s42/final_model.pt px4_field/models/swirl_v5B_s42.npz >> $B/export.log 2>&1 && echo "- $(date '+%H:%M') chain_0922e: export → px4_field/models/swirl_v5B_s42.npz" >> $L
  # 헤드리스 자동 리허설(δ0.5 고정 사건, circle) — 약 6분. 런처는 잔재가 없어야 뜬다(isaac_run 이 END 에서 정리함)
  sleep 5
  HEADLESS=1 timeout 900 bash px4_field/launch_isaac_manual.sh f13 1.0 --model models/swirl_v5B_s42.npz --attack dds --attack-fixed 0.5,20,40 --sitl-auto 2.5 > $B/rehearsal_B.log 2>&1
  echo "- $(date '+%H:%M') chain_0922e: 리허설 B rc=$? ($(ls -t px4_field/field_logs/f13_policy_*.csv | head -1))" >> $L
  F=$(ls -t px4_field/field_logs/f13_policy_*.csv | head -1)
  python3 px4_field/plot_obs_live.py "$F" --out results/claudecodefortest/f13_rehearsal/figs/fig_v5B_obs.png >> $B/rehearsal_B.log 2>&1 || true
  python3 px4_field/plot_policy_log.py "$F" --out results/claudecodefortest/f13_rehearsal/figs/fig_v5B_traj.png >> $B/rehearsal_B.log 2>&1 || true
  pkill -9 -x px4 2>/dev/null; sleep 3
fi
echo "- $(date '+%H:%M') chain_0922e: Isaac s42 SWIRL v5 후보 A′(pΔ0.05·R2·α0.05·ui1·τ0.005) 시작" >> $L
bash scripts/isaac_run.sh $B/swirlA2_s42 --config configs/isaac_v5_swirl.yaml --config configs/overlays/isaac.yaml --set run.seed=42 --set run.outdir=$B/swirlA2_s42 --set agent.swirl.p_delta=0.05 --set agent.swirl.R=2.0 --set agent.swirl.alpha=0.05 --set agent.tau=0.005 --set agent.update_interval=1
echo "- $(date '+%H:%M') chain_0922e: swirlA2_s42 rc=$(cat $B/swirlA2_s42/RUN_DONE)" >> $L
[ -f $B/swirlA2_s42/final_model.pt ] && python3 px4_field/export_policy.py $B/swirlA2_s42/final_model.pt px4_field/models/swirl_v5A2_s42.npz >> $B/export.log 2>&1
echo "- $(date '+%H:%M') chain_0922e 완료" >> $L; touch $B/CHAIN_DONE
