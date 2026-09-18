#!/usr/bin/env bash
# ③ 뒤 체이닝: burst duty별 결과성 vs 회피성 (가장 중요한 질문)
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
REPORT=burstduty_report.txt
# 시간창 env 안 씀 → 공격 step30~끝(긴 캠페인, 누적 관찰). 바람 없음(격리). δ0.7.
BIAS="0.0,3.05"; PATS="hover,aggressive"; EP=8
until grep -q "COUPLING TEST DONE" coupling_test_report.txt 2>/dev/null; do sleep 60; done
echo "########## BURSTDUTY START $(date +%H:%M:%S) ##########" | tee "$REPORT"
run_b () {  # name burst out
  local name="$1" burst="$2" out="$3"
  echo | tee -a "$REPORT"; echo "==== [$name] $(date +%H:%M:%S) burst='${burst:-상수}' ====" | tee -a "$REPORT"
  mkdir -p "$out"
  local E="SWEEP_ATTACK_TYPE=loe_combined"
  [ -n "$burst" ] && E="$E SWEEP_BURST=$burst"
  env $E setsid python3 online_rl_main.py --sweep --headless \
    --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
    --capture-disturbances "none:0" --capture-biases "$BIAS" --capture-patterns "$PATS" \
    --episodes "$EP" --ramp 0.0 --speed 10 --outdir "$out" > "$out/run.log" 2>&1 &
  wait $!; echo "  [done] $out $(date +%H:%M:%S)" | tee -a "$REPORT"
}
run_b "ON4·OFF8 (33%)"  "4,8"   results_burst_4_8
run_b "ON8·OFF8 (50%)"  "8,8"   results_burst_8_8
run_b "ON12·OFF8 (60%)" "12,8"  results_burst_12_8
run_b "ON20·OFF8 (71%)" "20,8"  results_burst_20_8
run_b "상수 (100%)"      ""      results_burst_const
echo | tee -a "$REPORT"; echo "==== [비교] ====" | tee -a "$REPORT"
~/isaacsim/python.sh burstduty_analysis.py \
  results_burst_4_8 "ON4/OFF8(33%)" results_burst_8_8 "ON8/OFF8(50%)" \
  results_burst_12_8 "ON12/OFF8(60%)" results_burst_20_8 "ON20/OFF8(71%)" \
  results_burst_const "상수(100%)" 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo "########## BURSTDUTY DONE $(date +%H:%M:%S) ##########" | tee -a "$REPORT"
