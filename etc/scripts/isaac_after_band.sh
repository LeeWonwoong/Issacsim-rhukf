#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f BANDCHK_DONE ]; do sleep 30; done
# night0908 의 ②③④ 만 실행 (① 밴드는 bandchk 로 대체) — 원 스크립트에서 ② 부터 재사용 불가하므로 직접
bash etc/scripts/isaac_night0908_rest.sh
