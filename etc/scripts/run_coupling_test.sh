#!/usr/bin/env bash
# ② 뒤 체이닝: 커플링/속도 3버전 gyro aliasing 검증
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
REPORT=coupling_test_report.txt
export WIND_START=110 WIND_END=200 SWEEP_ATK_START=160 SWEEP_ATK_END=250
DIST="none:0,wind_turbulence:12"; BIAS="0.0,1.74,3.05"; PATS="hover,aggressive"; EP=6
until grep -q "WINDALIAS DONE" windalias_report.txt 2>/dev/null; do sleep 60; done
echo "########## COUPLING TEST START $(date +%H:%M:%S) ##########" | tee "$REPORT"
run_c () {  # name env out
  local name="$1" envv="$2" out="$3"
  echo | tee -a "$REPORT"; echo "==== [$name] $(date +%H:%M:%S) ====" | tee -a "$REPORT"
  mkdir -p "$out"
  env $envv SWEEP_ATTACK_TYPE=loe_combined setsid python3 online_rl_main.py --sweep --headless \
    --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
    --capture-disturbances "$DIST" --capture-biases "$BIAS" --capture-patterns "$PATS" \
    --episodes "$EP" --ramp 0.0 --speed 10 --outdir "$out" > "$out/run.log" 2>&1 &
  wait $!; echo "  [done] $out $(date +%H:%M:%S)" | tee -a "$REPORT"
}
run_c "A 커플ON·1x"     ""                              results_coupling_on
run_c "B 커플OFF·1x"    "UKF_NO_COUPLING=1"             results_coupling_off
run_c "C 커플OFF·1.5x"  "UKF_NO_COUPLING=1 SPEED_SCALE=1.5" results_coupling_off15
echo | tee -a "$REPORT"; echo "==== [비교] ====" | tee -a "$REPORT"
~/isaacsim/python.sh coupling_test_analysis.py results_coupling_on results_coupling_off results_coupling_off15 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo "########## COUPLING TEST DONE $(date +%H:%M:%S) ##########" | tee -a "$REPORT"
