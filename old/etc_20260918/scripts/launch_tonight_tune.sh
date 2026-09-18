#!/usr/bin/env bash
# SWIRL 재튜닝(tonight_hm TONIGHT_SET=tune, GPU 18런) — tonight 1라운드(SW 10런) 완료 후 6 워커로 착수(GPU 경합·Isaac 리셋 완화)
cd /home/acsl/projects/Issacsim-rhukf; O=results/claudecodefortest/night/tonight
while :; do
  n=$(python3 -c "import json,glob; print(sum(len(json.load(open(f))) for f in glob.glob('$O/Tgpu_w*.json')))" 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 10 ] && break; sleep 120
done
for w in $(seq 0 5); do TONIGHT_SET=tune JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/tonight_hm.py work $w 6 > $O/Ugpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] tonight SWIRL 재튜닝(P₀0.1·N5 × blk·hm·mix × s42–44) GPU 6 워커 18 런 START" >> results/claudecodefortest/night/v5_launch.log
