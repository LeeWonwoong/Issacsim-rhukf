#!/usr/bin/env bash
# cp_regime 격자 구동기 (09-18): 단계 lin(FN 선형) → 판정 → 단계 const(FN 상수 −0.4) → 판정 → CP_DONE
# 실행: setsid nohup bash etc/scripts/launch_cp_regime.sh > results/claudecodefortest/cp_regime/driver.out 2>&1 < /dev/null &
cd /home/acsl/projects/Issacsim-rhukf || exit 1
C=results/claudecodefortest/cp_regime; mkdir -p $C/logs
exec 9>$C/.lock; flock -n 9 || { echo "이미 실행 중"; exit 1; }
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $C/cp.log; }
NG=${CP_NG:-10}; NC=${CP_NC:-4}
for S in lin const; do
  log "단계 $S 시작 (GPU $NG · CPU $NC 워커)"
  for w in $(seq 0 $((NG-1))); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 nice -n 8 python3 etc/scripts/cp_regime_scan.py work $S $w $NG 9>&- > $C/logs/w_${S}_gpu$w.out 2>&1 < /dev/null & done
  for w in $(seq 0 $((NC-1))); do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" nice -n 19 python3 etc/scripts/cp_regime_scan.py work $S $w $NC 9>&- > $C/logs/w_${S}_cpu$w.out 2>&1 < /dev/null & done
  wait
  log "단계 $S 워커 종료 — 완료 $(ls $C/${S}_*.json 2>/dev/null | wc -l)/44"
  python3 etc/scripts/cp_regime_scan.py judge > $C/judge_$S.out 2>&1; log "판정 갱신 (REPORT.md, curves_*.png)"
done
touch $C/CP_DONE; log "전체 종료"
