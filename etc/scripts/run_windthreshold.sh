#!/usr/bin/env bash
# 밤샘: 강풍 하한선 5~15 m/s (turbulence 정밀 + gust 대조) — innovation 이 노이즈 바닥 넘는 지점
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
REPORT=windthreshold_report.txt
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260   # 속도 1× (SPEED_SCALE 미설정)
# turbulence 5~15 정밀(하한선 탐색) + gust 5/9/15(대조) + none
DIST="none:0,wind_turbulence:5,wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_turbulence:9,wind_turbulence:10,wind_turbulence:12,wind_turbulence:15,wind_gust:5,wind_gust:9,wind_gust:15"
echo "########## WIND THRESHOLD START $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee "$REPORT"
mkdir -p results_windthreshold
SWEEP_ATTACK_TYPE=loe_combined setsid python3 online_rl_main.py --sweep --headless \
  --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
  --capture-disturbances "$DIST" --capture-biases "0.0,3.05" \
  --capture-patterns "hover,waypoint,circle,figure8,aggressive" \
  --episodes 4 --ramp 0.0 --speed 10 --outdir results_windthreshold \
  > results_windthreshold/run.log 2>&1 &
pid=$!; wait $pid
rows=$(( $(wc -l < results_windthreshold/sweep_summary.csv 2>/dev/null||echo 1) - 1 ))
echo "  [done] rows=$rows $(date +%H:%M:%S)" | tee -a "$REPORT"
echo | tee -a "$REPORT"; echo "==== [전궤적] ====" | tee -a "$REPORT"
~/isaacsim/python.sh strongwind_analysis.py results_windthreshold 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo | tee -a "$REPORT"; echo "==== [hover만] ====" | tee -a "$REPORT"
~/isaacsim/python.sh strongwind_analysis.py results_windthreshold --hover 2>/dev/null | grep -v "Warn\|Deprecat" | tee -a "$REPORT"
echo "########## WIND THRESHOLD DONE $(date +%Y-%m-%d_%H:%M:%S) ##########" | tee -a "$REPORT"
