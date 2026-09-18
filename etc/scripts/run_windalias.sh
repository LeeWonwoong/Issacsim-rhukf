#!/usr/bin/env bash
# 체이닝: windthreshold 끝난 뒤 → 공격강도×바람 aliasing 맵 (상수 + 펄스)
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
REPORT=windalias_report.txt
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260   # 속도 1×
DIST="none:0,wind_turbulence:9,wind_turbulence:15"
BIAS="0.0,1.31,2.18,3.05"        # 평시 + δ0.3/0.5/0.7
PATS="hover,aggressive,circle"
EP=6
# 선행(windthreshold 및 그 sweep) 끝날 때까지 대기
until grep -q "WIND THRESHOLD DONE" windthreshold_report.txt 2>/dev/null; do sleep 60; done
echo "########## WINDALIAS START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"

run_al () {  # name burst out
  local name="$1" burst="$2" out="$3"
  echo | tee -a "$REPORT"; echo "==== [$name] $(date +%H:%M:%S) ====" | tee -a "$REPORT"
  mkdir -p "$out"
  local ENV="SWEEP_ATTACK_TYPE=loe_combined"
  [ -n "$burst" ] && ENV="$ENV SWEEP_BURST=$burst"
  env $ENV setsid python3 online_rl_main.py --sweep --headless \
    --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
    --capture-disturbances "$DIST" --capture-biases "$BIAS" --capture-patterns "$PATS" \
    --episodes "$EP" --ramp 0.0 --speed 10 --outdir "$out" > "$out/run.log" 2>&1 &
  local pid=$!; wait $pid
  echo "  [done] $out rows=$(( $(wc -l < "$out/sweep_summary.csv" 2>/dev/null||echo 1)-1 )) $(date +%H:%M:%S)" | tee -a "$REPORT"
  ~/isaacsim/python.sh windalias_analysis.py "$out" 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
}

run_al "상수 aliasing맵" "" results_windalias_const
run_al "펄스 aliasing맵" "rand:2,6,6,14" results_windalias_pulse
echo | tee -a "$REPORT"; echo "########## WINDALIAS DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
