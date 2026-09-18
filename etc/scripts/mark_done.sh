#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
ORDER="A_r10_pd001 A_r10_pd005 A_r15_pd001 A_r15_pd005 B_r10_pd001 B_r10_pd005 B_r15_pd001 B_r15_pd005 adam_n16"
while true; do
  cur=$(ls -dt results_g1_*/ 2>/dev/null|head -1|sed 's|results_g1_||;s|/||')
  for r in $ORDER; do
    [ "$r" = "$cur" ] && break   # 현재 실행 런까지만(그 이전 = 완료/superseded)
    d=results_g1_$r
    if [ -d "$d" ] && [ ! -f "$d/RUN_DONE" ]; then
      ep=$(grep -aE 'TRAIN Ep [0-9]+/' $d/train.log 2>/dev/null|tail -1|grep -oE '^ *[0-9]+'|tr -d ' ')
      [ "${ep:-0}" -ge 150 ] && touch $d/RUN_DONE
    fi
  done
  [ -f results_g1_adam_n16/RUN_DONE ] && break
  sleep 120
done
