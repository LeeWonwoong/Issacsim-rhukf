#!/usr/bin/env bash
# ============================================================================
# run_overnight_qr.sh — roll 공격 스윕 → 임계 선정 → Q·R 오프라인 튜닝 (2026-08-13)
#
#   nohup ./run_overnight_qr.sh > overnight_qr.log 2>&1 &
#
# 절차 (각 단계 결과를 QR_TUNING_PROGRESS.txt 에 실시간 기록)
#   [1] roll 공격 크래시 스윕 — 5패턴 × bias{0.2,0.3,0.4,0.5} × {track,hover}
#         track 생존율로 "추락 직전" 세기를 찾는다. NIS(공격 vs 평시)도 함께.
#   [2] 임계 선정 — pick_qr_threshold.py 가 sub-crash 세기를 자동 결정
#   [3] Q·R 데이터 캡처 — 그 세기에서 track, --log-zu 로 (z,u) 저장
#   [4] Q·R 오프라인 튜닝 — qr_tune_offline.py 가 격자 재구동, d′·persistence 계산
#   [5] 문서화 — 절차·결과 전부 QR_TUNING_PROGRESS.txt + qr_tune_result.json
# ============================================================================
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/colcon_ws/install/setup.bash 2>/dev/null || true
set -u
cd /home/acsl/projects/Issacsim-rhukf
PY=python3
ISAAC=~/isaacsim/python.sh
DOC=QR_TUNING_PROGRESS.txt
BIASES="0.872,1.308,1.744,2.180"   # 정규화 roll δ = 0.2,0.3,0.4,0.5 (× authority 4.36)
EP_CRASH=4
EP_QR=4

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DOC"; }

cat > "$DOC" <<HDR
════════════════════════════════════════════════════════════════════════
 Q·R 튜닝 진행 기록 — roll 공격 기반          시작 $(date '+%Y-%m-%d %H:%M')
════════════════════════════════════════════════════════════════════════

목표
  roll 축에 정규화 δ=0.2~0.5(제어권한의 20~50%) 공격을 주입하고, **추락하지 않는
  직전 세기**에서 UKF 의 Q·R 을 튜닝한다.

튜닝 원칙 (사용자)
  · 필터에 고집을 준다 = 공격받은 센서를 흡수하지 않게 K 를 낮춘다 (R↑ 또는 Q↓).
  · 단 기동·바람에서도 추정이 발산하면 안 되고(평시 NIS 낮게), 공격이 끝나면 복귀해야 한다.
  · R 은 센서값을 그대로 쓰지 않는다 — 크게 부풀려야 흡수가 느려진다(고집).
  · 판정은 χ² 일관성이 아니라 **d′(공격 vs 평시)** + persistence(공격 중 유지).

메커니즘 (왜 흡수가 문제인가)
  predict 는 다음 스텝에 이번 update 결과 x̂(k) 에서 출발한다. K 가 크면 x̂ 가 공격받은
  센서 z 쪽으로 끌려가고, 다음 predict 의 h(x̂⁻) 가 z 에 가까워져 innovation→0 (흡수).
  R↑·Q↓ 로 K 를 낮추면 흡수가 느려져 공격 중에도 innovation 이 유지된다.

공격 매핑
  SWEEP_ATTACK_TYPE=loe_roll,  torque 모드,  value = δ × authority(4.36)
  → 정규화 nx = value/authority = δ.  bias 목록 = $BIASES  (δ 0.2/0.3/0.4/0.5)
════════════════════════════════════════════════════════════════════════

HDR

# ── [1] 크래시 스윕 ────────────────────────────────────────────────────
log "[1] roll 공격 크래시 스윕 시작 (5패턴 × 4세기 × {track,hover} × ${EP_CRASH}ep)"
OUT1=results_roll_crash_0813
rm -rf "$OUT1"; mkdir -p "$OUT1"
SWEEP_ATTACK_TYPE=loe_roll YAW_SLEW_RATE=1.57 setsid $PY online_rl_main.py --sweep --headless --speed 10 \
    --sweep-mode torque \
    --capture-mode deadline \
    --deadline-patterns hover,waypoint,circle,figure8,aggressive \
    --deadline-biases "$BIASES" \
    --episodes $EP_CRASH \
    --outdir "$OUT1" > "$OUT1/run.log" 2>&1 &
PID1=$!
# 배너 확인
for i in $(seq 1 90); do sleep 2; grep -qE "CAPTURE:deadline|Traceback" "$OUT1/run.log" 2>/dev/null && break; done
if grep -q "Traceback" "$OUT1/run.log" 2>/dev/null; then
  log "✗ [1] 예외 발생 — 중단"; tail -30 "$OUT1/run.log" | tee -a "$DOC"; kill -- -$PID1 2>/dev/null; exit 1
fi
grep -m1 "CAPTURE:deadline" "$OUT1/run.log" | tee -a "$DOC"
# 완료 대기
W=0; while kill -0 $PID1 2>/dev/null; do sleep 20; W=$((W+20)); [ $W -ge 7200 ] && { log "⚠ [1] 2h 초과 강제종료"; kill -- -$PID1 2>/dev/null; break; }; done
log "[1] 크래시 스윕 완료 (경과 ${W}s, $(wc -l < $OUT1/sweep_detail.csv) 행)"

# ── [2] 임계 선정 ──────────────────────────────────────────────────────
log "[2] 임계 선정 — pick_qr_threshold.py"
$ISAAC pick_qr_threshold.py "$OUT1" | tee -a "$DOC"
CHOSEN=$(cat "$OUT1/chosen_bias.txt" 2>/dev/null)
if [ -z "$CHOSEN" ]; then log "✗ [2] 임계 선정 실패 — 중단"; exit 1; fi
log "[2] 선정된 sub-crash 세기: bias=$CHOSEN"

# ── [3] Q·R 데이터 캡처 (log-zu) ───────────────────────────────────────
log "[3] Q·R 데이터 캡처 — bias=$CHOSEN, track, --log-zu (${EP_QR}ep × 5패턴)"
OUT3=results_qr_data_0813
rm -rf "$OUT3"; mkdir -p "$OUT3"
SWEEP_ATTACK_TYPE=loe_roll YAW_SLEW_RATE=1.57 setsid $PY online_rl_main.py --sweep --headless --speed 10 \
    --sweep-mode torque \
    --capture-mode deadline \
    --deadline-patterns hover,waypoint,circle,figure8,aggressive \
    --deadline-biases "$CHOSEN" \
    --episodes $EP_QR --log-zu \
    --outdir "$OUT3" > "$OUT3/run.log" 2>&1 &
PID3=$!
for i in $(seq 1 90); do sleep 2; grep -qE "CAPTURE:deadline|Traceback" "$OUT3/run.log" 2>/dev/null && break; done
if grep -q "Traceback" "$OUT3/run.log" 2>/dev/null; then
  log "✗ [3] 예외 — 중단"; tail -30 "$OUT3/run.log" | tee -a "$DOC"; kill -- -$PID3 2>/dev/null; exit 1
fi
W=0; while kill -0 $PID3 2>/dev/null; do sleep 20; W=$((W+20)); [ $W -ge 5400 ] && { log "⚠ [3] 초과 강제종료"; kill -- -$PID3 2>/dev/null; break; }; done
ZU="$OUT3/zu_log.npz"
if [ ! -f "$ZU" ]; then log "✗ [3] zu_log.npz 없음 — 중단"; exit 1; fi
log "[3] Q·R 데이터 캡처 완료 (zu_log $(du -h $ZU | cut -f1))"

# ── [4] Q·R 오프라인 튜닝 ──────────────────────────────────────────────
log "[4] Q·R 오프라인 격자 튜닝 — qr_tune_offline.py"
{ echo; echo "════ [4] Q·R 격자 (gyro 채널) ════"; } >> "$DOC"
$ISAAC qr_tune_offline.py "$ZU" -o "$OUT3" 2>&1 | grep -viE "warn|deprecat" | tee -a "$DOC"

# ── [5] 마무리 ─────────────────────────────────────────────────────────
{
echo
echo "════════════════════════════════════════════════════════════════════════"
echo " 완료 $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════════════════════════════════"
echo "산출물:"
echo "  $OUT1/          크래시 스윕 (생존율·NIS)"
echo "  $OUT1/roll_crash_analysis.txt   패턴×세기 표"
echo "  $OUT3/zu_log.npz                Q·R 튜닝 입력 (z,u)"
echo "  $OUT3/qr_tune_result.json       격자 결과 (d′·persistence 전체)"
echo
echo "다음: 위 ★최적 R_gyro·Q_gyro 를 env/ukf_filter.py 에 반영할지 아침에 판단."
echo "  ⚠ 반영 전 백업. 복귀(공격 종료 후 innovation 감소)는 이번에 안 쟀다 —"
echo "     이 스윕은 공격이 끝까지 지속(attack_end 99999). 복귀 시상수는 별도 burst 캡처 필요."
} | tee -a "$DOC"
log "전체 완료."
