#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f V7HNS_W0_DONE ] || [ ! -f V7HNS_W1_DONE ]; do sleep 60; done
rm -f V7LTH_W0_DONE V7LTH_W1_DONE
python3 etc/scripts/v7_lethal.py 0 > /tmp/v7lth_w0.log 2>&1 &
python3 etc/scripts/v7_lethal.py 1 > /tmp/v7lth_w1.log 2>&1 &
wait
