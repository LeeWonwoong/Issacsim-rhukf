#!/usr/bin/env bash
# surrogate v5 스윕 재개 (재부팅 후): 기존 JSON 결과는 유지, 미완 런만 실행. DONE 마커만 제거.
cd /home/acsl/projects/Issacsim-rhukf
rm -f results/claudecodefortest/night/V5S*_DONE results/claudecodefortest/night/V5SWEEPS_*_DONE
for w in 0 1 2 3 4 5 6 7; do setsid nohup python3 etc/scripts/night_sweeps_v5.py $w 8 >> results/claudecodefortest/night/v5_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v5 sweep RESUME 8 workers" >> results/claudecodefortest/night/v5_launch.log
