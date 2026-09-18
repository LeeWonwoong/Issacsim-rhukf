#!/usr/bin/env bash
# ext_D(가치함수 몫 실험) GPU 작업이 끝나면 그림 4 캡처를 띄운다. Isaac 은 GPU 를 크게 쓰므로 겹치지 않게.
cd /home/acsl/projects/Issacsim-rhukf
while :; do
  left=$(JOBSET=gpu EXT_SW=SW EXT_ADAM=Adam1e-3 python3 -c "
import os,sys; sys.path.insert(0,'etc/scripts'); import tonight_queue as Q
print(sum(1 for j in Q.ext_jobs('gpu','ext_D') if not os.path.exists(f'{Q.Q}/{j[0]}.json')))" 2>/dev/null)
  [ "${left:-1}" -eq 0 ] && break; sleep 120
done
echo "[$(date '+%m-%d %H:%M')] ext_D 완료 → fig4_nis 캡처 착수" >> results/claudecodefortest/night/tonight/ext_launch.log
bash etc/scripts/capture_fig4_nis.sh
