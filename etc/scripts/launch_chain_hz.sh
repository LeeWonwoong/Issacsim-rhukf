#!/usr/bin/env bash
# chain_hz 사슬 구동기 (09-15 저녁): v33 종료 대기 → D0 → S1 → D1 → S2 → D2 → S3 → D3. 각 단계 무대는 앞 판정 JSON 이 정한다.
# v34(보류)·v35(무대 탐색 초안)를 대체한다. 결과: results/claudecodefortest/night/chain_hz/ (chain.log · DECISIONS.md · REPORT.md)
# 실행: setsid nohup bash etc/scripts/launch_chain_hz.sh > results/claudecodefortest/night/chain_hz/driver.out 2>&1 < /dev/null &
# 재실행 안전: flock 단일 인스턴스, 판정 JSON 이 있으면 그 단계는 건너뜀, 워커는 json 이름 기준으로 이어 돎.
cd /home/acsl/projects/Issacsim-rhukf || exit 1
N=results/claudecodefortest/night; C=$N/chain_hz; mkdir -p $C
exec 9>$C/.lock; flock -n 9 || { echo "[$(date '+%m-%d %H:%M')] 이미 실행 중인 chain_hz 구동기 있음 — 중단"; exit 1; }
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $C/chain.log; }
cnt() { ps aux | grep -cE "$1"; }
NG=${CHAIN_NG:-10}; NC=${CHAIN_NC:-8}

log "구동기 시작 (GPU 워커 $NG, CPU 워커 $NC)"
while :; do
  n33=$(cnt '[e]nv_scan_v33.py'); n34=$(cnt '[e]nv_scan_v34.py'); n35=$(cnt '[e]nv_scan_v35.py'); nis=$(cnt '[o]nline_rl_main.py')
  [ "$n33" -eq 0 ] && [ "$n34" -eq 0 ] && [ "$n35" -eq 0 ] && [ "$nis" -eq 0 ] && break
  sleep 120
done
nd=$(ls $N | grep -c 'V33gpu_w.*_DONE'); log "GPU 비었음 (V33gpu DONE $nd/10)"
# ★ 사용자 요청(09-15 16:55): 사슬 런 전에 v33 분석·보고·아티팩트 갱신을 먼저 끝낸다.
#   V33_ENDED 를 남기면 Claude 점검 cron 이 분석을 수행하고 ANALYSIS_V33_DONE 을 만든다.
#   세션이 없어 GPU 가 밤새 노는 일을 막기 위해 최대 CHAIN_HOLD_MIN(기본 120)분만 기다린다.
touch $C/V33_ENDED
HOLD=${CHAIN_HOLD_MIN:-120}; waited=0
if [ ! -f $C/ANALYSIS_V33_DONE ]; then log "v33 분석·보고·아티팩트 갱신 대기 (최대 ${HOLD}분)"; fi
while [ ! -f $C/ANALYSIS_V33_DONE ] && [ $waited -lt $HOLD ]; do sleep 60; waited=$((waited+1)); done
if [ -f $C/ANALYSIS_V33_DONE ]; then log "v33 분석 완료 확인 (${waited}분 대기) → 사슬 진행"; else log "⚠ v33 분석 플래그 없음 ${HOLD}분 경과 → 사슬 진행(분석은 사후)"; fi

run_stage() {   # $1 = S1|S2|S3
  local S=$1 ng nc w
  ng=$(python3 etc/scripts/chain_hz.py count $S gpu) || { log "$S 잡 구성 실패 — 중단"; exit 1; }
  nc=$(python3 etc/scripts/chain_hz.py count $S cpu) || { log "$S 잡 구성 실패 — 중단"; exit 1; }
  local wg=$(( ng < NG ? ng : NG )) wc=$(( nc < NC ? nc : NC ))
  # ★09-15 20:00: Isaac(online_rl_main)이 GPU 를 쓰는 동안은 착수하지 않는다(aggressive 수정 검증 캡처 등)
  local wi=0; while [ "$(cnt '[o]nline_rl_main.py')" -gt 0 ] || [ -f $C/ISAAC_BUSY ]; do [ $wi -eq 0 ] && log "$S 착수 전 Isaac 실행 중 — 종료 대기"; wi=$((wi+1)); sleep 60; done
  [ $wi -gt 0 ] && log "Isaac 종료 확인 (${wi}분 대기) → $S 진행"
  if [ "$(cnt "[c]hain_hz.py work $S ")" -gt 0 ]; then
    log "$S 워커가 이미 실행 중 — 재착수 없이 완료만 기다림"      # 리뷰 수정: 구동기 재실행 시 중복 착수 방지
  else
    rm -f $C/${S}*_DONE
    log "$S 착수: GPU $ng 런 / $wg 워커, CPU $nc 런 / $wc 워커 (저장된 런은 워커가 건너뜀)"
    python3 etc/scripts/chain_hz.py count $S >> $C/chain.log 2>&1
    # 리뷰 수정: 9>&- 로 flock fd 를 워커에 물려주지 않는다(구동기가 죽은 뒤 재실행이 잠금에 막히지 않게)
    for w in $(seq 0 $((wg-1))); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/chain_hz.py work $S $w $wg 9>&- > $C/${S}gpu_w$w.log 2>&1 < /dev/null & done
    for w in $(seq 0 $((wc-1))); do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup nice -n 19 python3 etc/scripts/chain_hz.py work $S $w $wc 9>&- > $C/${S}cpu_w$w.log 2>&1 < /dev/null & done
    echo "[$(date '+%m-%d %H:%M')] chain_hz $S (GPU $ng 런 + CPU $nc 런) START" >> $N/v5_launch.log
    sleep 60
  fi
  while :; do
    local done_n=$(ls $C | grep -cE "^${S}(gpu|cpu)_w[0-9]+_DONE$") alive=$(cnt "[c]hain_hz.py work $S ")
    [ "$done_n" -ge $((wg+wc)) ] && break
    if [ "$alive" -eq 0 ]; then log "⚠ $S 워커 없음인데 DONE $done_n/$((wg+wc)) — 외부 종료 추정, 있는 결과로 판정 진행"; break; fi
    sleep 180
  done
  local fails=$(cat $C/${S}*_w*.log 2>/dev/null | grep -c '!! ')
  log "$S 완료 (실패 런 $fails)"
}

decide() {   # $1 = D1|D2|D3, $2 = 앞 단계 S1|S2|S3. 종료코드 3(원자료 불완전)이면 그 단계 워커를 1회 재착수해 누락 런만 채운 뒤 재판정
  local rc; python3 etc/scripts/chain_hz.py decide $1 >> $C/chain.log 2>&1; rc=$?
  if [ $rc -eq 3 ]; then log "$1 원자료 불완전 → $2 워커 재착수(누락 런만) 1회"; run_stage $2; python3 etc/scripts/chain_hz.py decide $1 >> $C/chain.log 2>&1; rc=$?; fi
  [ $rc -eq 0 ] || { log "$1 실패(rc $rc) — 사슬 중단. DECISIONS.md·워커 로그 확인 후 구동기 재실행"; exit 1; }
}
[ -f $C/D0.json ] || { python3 etc/scripts/chain_hz.py decide D0 >> $C/chain.log 2>&1 || { log "D0 실패(v33 원자료 불완전 가능) — 사슬 중단"; exit 1; }; }
log "D0: $(cat $C/D0.json | python3 -c 'import json,sys;print("k*=",json.load(sys.stdin)["k"])')"
[ -f $C/D1.json ] || { run_stage S1; decide D1 S1; }
log "D1: $(python3 -c "import json;d=json.load(open('$C/D1.json'));print('c*=',d['cell'])")"
[ -f $C/D2.json ] || { run_stage S2; decide D2 S2; }
log "D2: $(python3 -c "import json;d=json.load(open('$C/D2.json'));print({k:d[k] for k in ('SW','UKF','EKF','Adam')})")"
[ -f $C/REPORT.md ] || { run_stage S3; decide D3 S3; }
log "프레임워크 확정 — $C/REPORT.md (Claude 점검 cron 이 최종 분석·보고·아티팩트 갱신 후 FINAL_REPORTED 생성)"
touch $C/CHAIN_DONE
# ★ 사용자 요청(09-15 19:25): 확정 무대에서 배치 64 절제 (SW* 창 N 10·14·2N*, UKF*·EKF*·Adam 배치 64, 시드 101–110 ↔ S3 배치 128 짝)
[ -f $C/REPORT_B64.md ] || { run_stage S4; decide D4 S4; }
log "배치 64 절제 종료 — $C/REPORT_B64.md (cron 이 분석·보고·아티팩트 갱신 후 B64_REPORTED 생성)"
touch $C/B64_DONE
