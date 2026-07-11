#!/usr/bin/env bash
# adam 학습 (speed 10) — 바람 config 고정(turbulence ws6~8 보조) + 공격 combined ft1.5 동결
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
mkdir -p results_adam
exec python3 -u online_rl_main.py --headless --agent adam --speed 10 --outdir results_adam
