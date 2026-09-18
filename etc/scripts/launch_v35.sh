#!/usr/bin/env bash
# v35 (09-15 밤, 무대 탐색 STAGE_SEARCH_0915): v33 GPU 완료(V33gpu DONE 10) ∧ Isaac(online_rl_main) 유휴 ∧ v34 미실행 대기 → 착수.
# 하한 0.10 · 시드 42–46 · blk50k γ0.9/γ0.95 · 콜드스타트 ε500 ws10h 30 ep · SWIRL 블록 튜닝(P₀0.1, N5). GPU 60 런 10 워커 + CPU Adam lr×3 45 런 4 워커.
# v34(하한 0.05) 를 대체한다 — launch_v34.sh 와 동시에 걸지 말 것(아래에서 감지 시 중단).
# 실행: setsid nohup bash etc/scripts/launch_v35.sh > results/claudecodefortest/night/v35_autolaunch.log 2>&1 < /dev/null &
# 점검 수정(09-15 16:30): ① launch_v34.sh 대기 셸·v35 워커가 이미 있으면 중단(같은 폴링에서 둘이 동시 착수하는 경합 방지)
#   ② v33 워커가 외부 종료돼 DONE 이 10 개가 안 되어도, v33 프로세스가 모두 사라지면 경고 후 착수(무한 대기 방지)
#   ③ 결과 json 은 지우지 않는다(DONE 만 삭제) — 런처 재실행 시 워커가 이름 기준으로 이어 돈다
cd /home/acsl/projects/Issacsim-rhukf; N=results/claudecodefortest/night
cnt() { ps aux | grep -cE "$1"; }
[ "$(cnt '[l]aunch_v34.sh')" -eq 0 ] || { echo "[$(date '+%m-%d %H:%M')] launch_v34.sh 대기/실행 중 — v35 와 동시 착수 금지, 중단"; exit 1; }
[ "$(cnt '[e]nv_scan_v35.py')" -eq 0 ] || { echo "[$(date '+%m-%d %H:%M')] env_scan_v35 워커가 이미 실행 중 — 중복 착수 금지, 중단"; exit 1; }
echo "[$(date '+%m-%d %H:%M')] v35 대기 시작 (V33gpu DONE $(ls $N | grep -c 'V33gpu_w.*_DONE')/10)"
while :; do
  nd=$(ls $N | grep -c 'V33gpu_w.*_DONE'); n33=$(cnt '[e]nv_scan_v33.py'); n34=$(cnt '[e]nv_scan_v34.py'); nis=$(cnt '[o]nline_rl_main.py')
  if [ "$nis" -eq 0 ] && [ "$n34" -eq 0 ] && [ "$n33" -eq 0 ]; then
    [ "$nd" -lt 10 ] && echo "[$(date '+%m-%d %H:%M')] ⚠ v33 프로세스 없음인데 V33gpu DONE $nd/10 — v33 워커 외부 종료 추정, v35 착수는 진행"
    break
  fi
  sleep 120
done
echo "[$(date '+%m-%d %H:%M')] v33 완료 → v35 착수"
rm -f $N/V35*_DONE
for w in $(seq 0 9); do JOBSET=gpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 8 python3 etc/scripts/env_scan_v35.py $w 10 > $N/v35gpu_w$w.log 2>&1 < /dev/null & done
for w in 0 1 2 3; do JOBSET=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup nice -n 19 python3 etc/scripts/env_scan_v35.py $w 4 > $N/v35cpu_w$w.log 2>&1 < /dev/null & done
echo "[$(date '+%m-%d %H:%M')] v35 무대 탐색(하한 0.10; blk50k g90/g95·cold30 w10h·SW 튜닝) (GPU 10 워커 60 런 + CPU 4 워커 45 런) START" >> $N/v5_launch.log
