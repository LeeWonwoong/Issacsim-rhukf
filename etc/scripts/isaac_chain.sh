#!/usr/bin/env bash
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f results_o3_calwin_spas_s42/RUN_DONE ]; do sleep 120; done
exec bash etc/scripts/isaac_spasab.sh
