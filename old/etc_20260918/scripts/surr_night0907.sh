#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
LOG=surr_night0907.log
rm -f V6TUNE_S1_W0_DONE V6TUNE_S1_W1_DONE V6TUNE_S2_W0_DONE V6TUNE_S2_W1_DONE
echo "[$(date '+%m-%d %H:%M')] v6 tune stage1 시작 (30 config, 2 workers)" | tee $LOG
python3 etc/scripts/v6_tune.py 0 1 > /tmp/v6tune_w0.log 2>&1 &
python3 etc/scripts/v6_tune.py 1 1 > /tmp/v6tune_w1.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] stage1 완료 → stage2 (top10 × N6/seed43)" | tee -a $LOG
python3 etc/scripts/v6_tune.py 0 2 > /tmp/v6tune2_w0.log 2>&1 &
python3 etc/scripts/v6_tune.py 1 2 > /tmp/v6tune2_w1.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M')] v6 tune 전체 완료" | tee -a $LOG
touch V6TUNE_DONE
