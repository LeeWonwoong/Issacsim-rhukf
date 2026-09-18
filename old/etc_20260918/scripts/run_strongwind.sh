#!/usr/bin/env bash
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260
# 속도는 1× (SPEED_SCALE 미설정). 강풍 9~15 m/s, gust+turbulence 둘 다.
DIST="none:0,wind_turbulence:9,wind_turbulence:12,wind_turbulence:15,wind_gust:9,wind_gust:12,wind_gust:15"
mkdir -p results_strongwind
SWEEP_ATTACK_TYPE=loe_combined setsid python3 online_rl_main.py --sweep --headless \
  --sweep-mode torque --torque-yaw-ratio 0.0 --capture-mode hijack \
  --capture-disturbances "$DIST" --capture-biases "0.0,3.05" \
  --capture-patterns "hover,waypoint,circle,figure8,aggressive" \
  --episodes 4 --ramp 0.0 --speed 10 --outdir results_strongwind \
  > results_strongwind/run.log 2>&1 &
echo "strongwind pid $!"
